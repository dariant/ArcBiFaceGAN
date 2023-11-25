import torch
import dnnlib
from DB_SG2_with_id_features import legacy as legacy
import numpy as np 
import os 
import random 
from Arcface_files.ArcFace_functions import prepare_locked_ArcFace_model
import json 
import collections 
from tqdm import tqdm 
import click 
from torchvision import utils, transforms
from facenet_pytorch import MTCNN

#########################################
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

#########################################
def cosine_similarity_torch(x, y):
    dot = torch.sum(torch.multiply(x, y), axis=1)
    norm = torch.norm(x) * torch.norm(y)
    similarity = torch.clip(dot/norm, -1., 1.)
    return similarity

#########################################
def generate_image_from_z(G, z_latent, arcface_latent, general_truncation_psi):
    w_latent = G.mapping(z_latent, arcface_latent, truncation_psi=general_truncation_psi)
    img, img_NIR, _ = G.synthesis(w_latent)
    return img, img_NIR 

#########################################
def preprocess_generated_image_general(img):
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img 

#########################################
def prepare_for_arcface_model_torch(img): 
    img = transforms.functional.resize(img, (112, 112))
    img = img.float()
    img = ((img / 255) - 0.5) / 0.5 
    return img 

#########################################
def preprocess_generated_image_for_ArcFace_torch(img): 
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = transforms.functional.resize(img, (112, 112))#img.reshape((112, 112))
    img = img.float()
    return img 

#########################################
def load_training_arcfaces(fname, device):
    with open(fname) as f:
        labels = json.load(f)

    labels = collections.OrderedDict(labels)
    
    most_sim_arcs = dict()
    for k in labels.keys():
        id_key = k.split("_")[0]
        if id_key in most_sim_arcs: 
            break

        most_sim_arcs[id_key] = torch.from_numpy(np.array(labels[k]))[None, :].to(device)

    labels_names = [fname.replace('\\', '/') for fname in labels.keys()]
    labels = [labels[fname.replace('\\', '/')] for fname in labels.keys()]
    
    labels = np.array(labels)
    labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
    return labels, labels_names, most_sim_arcs

#########################################
def is_unique_id(tmp_arc, eval_arcs, too_similar_threshold, logging):
    for arc_key, arc_val in eval_arcs.items(): 
        sim = cosine_similarity_torch(tmp_arc, arc_val)

        if sim.abs() > too_similar_threshold:
            if logging: print("\nNot new, it's:", arc_key, "  sim:", sim)
            # print("Sim to previous ids:", sim)
            return False 
    
    return True 

#########################################

def check_if_face_centered(landmarks):
    # distance left eye (or right eye) to nose
    dist_leye_to_nose =  landmarks[0][0][0] - landmarks[0][2][0]
    dist_reye_to_nose =  landmarks[0][1][0] - landmarks[0][2][0]

    # we want this close to 0, or at least than the average distance
    if torch.abs(dist_leye_to_nose + dist_reye_to_nose) < 1/3 * (torch.abs(dist_leye_to_nose) + torch.abs(dist_reye_to_nose)): 
        return True 
    
    return False 

#########################################
def save_images(img_list, NIR_img_list, folder, id):

    folder = os.path.join(folder, f"id_{id}")
    folder_VIS = os.path.join(folder, "VIS")
    folder_NIR = os.path.join(folder, "NIR")
    os.makedirs(folder_VIS, exist_ok=True)
    os.makedirs(folder_NIR, exist_ok=True)

    for i, img in enumerate(img_list):
        NIR_img = NIR_img_list[i]
        utils.save_image(img, os.path.join(folder_VIS, f"sample_{i}_VIS.png"), normalize=True, range=(-1, 1))
        utils.save_image(NIR_img, os.path.join(folder_NIR, f"sample_{i}_NIR.png"), normalize=True, range=(-1, 1))

#########################################
def compare_to_samples_of_same_id(tmp_arcface_feat, first_arc, sample_arcs_list, threshold_for_same_ID, threshold_for_too_similar, logging):
    # compute similarity
    sim = cosine_similarity_torch(tmp_arcface_feat, first_arc)
    
    # if similarity less than similarity of (avg - 2* std) ..  then it is not the same ID! ... DO NOT SAVE 
    if sim.abs() < threshold_for_same_ID: 
        if logging: print("Not enough similar sample")
        return False 
        
    # Check also that the similarity is not too high! ... We do not want repeated images 
    for tmp_prev_ind, prev_arc in enumerate(sample_arcs_list): 
        prev_sim = cosine_similarity_torch(tmp_arcface_feat, prev_arc)
        if logging: print("... Sim to prev:", tmp_prev_ind, "=", prev_sim)
        if prev_sim > threshold_for_too_similar:
            return False 
        
    return True 

###########################################
def detect_face(tmp_img, mtcnn_model, device, logging):
    
    boxes, probs, landmarks = mtcnn_model.detect(tmp_img.permute(0, 2, 3, 1), landmarks=True)
    
    # TODO FIX NE RABIS TOK PREGLEDAT ..
    # if no face detected or too many faces
    if len(landmarks) == 0 or len(landmarks) > 1 or landmarks[0] is None:
        if logging: print("Either no face or too many faces", len(landmarks))
        return False 
    
    landmarks = landmarks[0].astype(np.float32)
    landmarks = torch.from_numpy(landmarks).to(device)
    
    # or if face detection probabilty is low
    if probs[0][0] < 0.97: # TODO za ta kriterij tudi test na real slikah in vzami povprečje ! ... povprečje je 1.0 !
        if logging: print("Probability low:", probs[0])
        return False  
    
    # check if first face is centered (i.e. the distance between the nose and both eyes is similar)
    is_face_centered = check_if_face_centered(landmarks)
    if not is_face_centered: 
        if logging: print("Not centered")
        return False  

    return True 


###########################################

@click.command()

# Configuration 
@click.option('--gen_model', help='Path to pretrained identity-conditioned StyleGAN2 model (pkl file)', required=True)
@click.option('--rec_model', help='Path to recognition model (pth file)', required=True)
@click.option('--training_ids', help='Path to the .json file of identity features of the training dataset', metavar='PATH', required=True)
@click.option('--outdir', help='Path to output folder', metavar='PATH', required=True)
@click.option('--gpu_device_number', help='Which CUDA gpu to use [default: 0]', type=int, default=0, metavar='INT')
@click.option('--ids', help='How many identities to generate [default: 100]', type=int, default=100, metavar='INT')
@click.option('--samples_per_id', help='How many samples to generate for each identity [default: 32]', type=int, default=32, metavar='INT')
@click.option('--truncation', help='What truncation factor to use during sampling', type=float, default=0.7, metavar='FLOAT')
@click.option('--seed', help='Select a seed to use during sampling', type=int, default=0, metavar='INT')

def main(**args):
    print("Config:", args)
    print("Current directory:", os.getcwd())

    torch.set_grad_enabled(False)
    device = f"cuda:{args['gpu_device_number']}"    
    seed = args["seed"]
    num_ids = args["ids"] 
    samples_per_id = args["samples_per_id"]    
    general_truncation_psi = args["truncation"]
    
    logging = False

    nrow = 4 # number of rows in grid images  
    
    arcface_multiplier = 4
    avg_similarity = 0.776
    std_similarity = 0.077
    threshold_for_same_ID = avg_similarity - 2 * std_similarity
    threshold_for_same_ID = torch.from_numpy(np.array([threshold_for_same_ID])).to(device) 
    threshold_for_too_similar = avg_similarity + std_similarity
    
    network_pkl = args["gen_model"]
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    arcface_model = prepare_locked_ArcFace_model(args["rec_model"]).to(device)     
    mtcnn_model = MTCNN(select_largest=False, device=device)
    _, _, all_arcs = load_training_arcfaces(args["training_ids"], device)

    set_all_seeds(seed)
    
    os.makedirs(args['outdir'], exist_ok=True)
    print("Existing identities:", len(all_arcs.keys()))
    
    max_attempts = 2500

    id = 0 
    pbar = tqdm(total=num_ids)

    while id < num_ids:
        arcface_start = torch.randn(1, 512, device=device)
        arcface_start = torch.nn.functional.normalize(arcface_start, p=2, dim=1)
        arcface_feat = arcface_multiplier * arcface_start 

        img_list = []; NIR_img_list = []; sample_arcs_list = []
        
        current_sample = 0
        attempts = 0
        
        is_face_detected = False 
        enough_samples_per_id = True

        while current_sample < samples_per_id:
            attempts += 1 
            if attempts > max_attempts: 
                print("\nToo many failed samples... Skipping to new ID.")
                enough_samples_per_id = False
                break 
            
            z_feat = torch.randn(1, 512, device=device) 
            
            img, img_NIR = generate_image_from_z(G, z_feat, arcface_feat, general_truncation_psi)    
            tmp_img = preprocess_generated_image_general(img)

            # check if a face is detected in the first sample 
            if  current_sample == 0: 
                is_face_detected = detect_face(tmp_img, mtcnn_model, device, logging)

                # if not, generate new id            
                if not is_face_detected:
                    break 

            # extract arcface feature from image and normalize 
            img_tmp = prepare_for_arcface_model_torch(tmp_img)#preprocess_generated_image_for_ArcFace_torch(img)
            tmp_arcface_feat = arcface_model(img_tmp)
            tmp_arcface_feat = torch.nn.functional.normalize(tmp_arcface_feat,  p=2, dim=1 )

            # if first sample isn't unique, find new ID
            new_unique_id = is_unique_id(tmp_arcface_feat, all_arcs, threshold_for_same_ID, logging)
            if  current_sample == 0:
                if not new_unique_id: break 
                
            # check if other samples are also unique, but don't break entire loop if not   
            else: 
                if not new_unique_id: continue 
            ######################

            # Compare between samples 
            if current_sample == 0: 
                first_arc = tmp_arcface_feat
            else:
                similar_to_samples_of_same_id = compare_to_samples_of_same_id(tmp_arcface_feat,first_arc, sample_arcs_list,  threshold_for_same_ID, threshold_for_too_similar, logging)
                if not similar_to_samples_of_same_id: continue 
       
            # append to lists
            img_list.append(img)
            sample_arcs_list.append(tmp_arcface_feat)

            NIR_img_list.append(img_NIR)

            current_sample+=1 


        if not is_face_detected or not new_unique_id or not enough_samples_per_id:
            #print("Do not save, skip to next id.")
            continue 
        
        else:
            all_arcs["sample_" + str(id)] = first_arc #tmp_arcface_feat # TODO should be first arc
        

        # save images 
        utils.save_image(
                torch.cat(img_list, dim=0),
                f"{args['outdir']}/id_{id}_samples_VIS.png",#_{ind}.png",
                normalize=True,
                range=(-1, 1),
                nrow=int(int(samples_per_id)/nrow),
            )
        
        utils.save_image(
            torch.cat(NIR_img_list, dim=0),
            f"{args['outdir']}/id_{id}_samples_NIR.png",#_{ind}.png",
            normalize=True,
            range=(-1, 1),
            nrow=int(int(samples_per_id)/nrow)#int(int(n_sample)/5),
        )
    
        save_images(img_list, NIR_img_list, args['outdir'], id)
        
        id+=1
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    main()
