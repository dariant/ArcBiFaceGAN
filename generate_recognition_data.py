import argparse

import torch
from torchvision import utils, transforms
import dnnlib
#from model import Generator
from DB_SG2_with_id_features import legacy as legacy
import numpy as np 
import os 
import random 
from PIL import Image
import torch.nn.functional as F
from Arcface_files.ArcFace_functions import preprocess_image_for_ArcFace, prepare_locked_ArcFace_model
import json 
import collections 
from facenet_pytorch import MTCNN
from tqdm import tqdm 

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

#########################################
def cosine_similarity(x, y):
    dot = np.sum(np.multiply(x, y), axis=1)
    norm = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    similarity = np.clip(dot/norm, -1., 1.)
    return similarity
#########################################

#########################################
def cosine_similarity_torch(x, y):
    dot = torch.sum(torch.multiply(x, y), axis=1)
    norm = torch.norm(x) * torch.norm(y)
    similarity = torch.clip(dot/norm, -1., 1.)
    return similarity
#########################################



#########################################
def random_similar_cos(v, cos_similarity_threshold):
    # Form the unit vector parallel to v:
    u = (v / torch.norm(v))[0]
    # Pick a random vector:
    r = torch.randn(len(u)).to(device)
    # Form a vector perpendicular to v:
    u = u.float()
    uperp = r - r.dot(u)*u
    # Make it a unit vector:
    uperp = uperp / torch.norm(uperp)
    # w is the linear combination of u and uperp with coefficients costheta and sin(theta) = sqrt(1 - costheta**2), respectively:
    w = cos_similarity_threshold * u + torch.sqrt(1 - cos_similarity_threshold**2)*uperp
    w = w[None, :]

    return w
#########################################

def generate_image_from_z(z_latent, arcface_latent):
    w_latent = G.mapping(z_latent, arcface_latent, truncation_psi=general_truncation_psi)
    img, img_NIR, _ = G.synthesis(w_latent)
    #img = img #print(img.shape)
    #img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img, img_NIR 

#########################################
def preprocess_image_for_ArcFace_torch(img): 
    img = transforms.functional.resize(img, (112, 112))#img.reshape((112, 112))
    img = img.float()
    return img 

#########################################
def preprocess_generated_image_general(img):
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img 

#########################################
def prepare_for_arcface_model_torch(img): 
    # img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8) # NOTE already done for MTCNN
    img = transforms.functional.resize(img, (112, 112))#img.reshape((112, 112))
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
def preprocess_generated_image_for_ArcFace_pytorch(img): 
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = transforms.functional.resize(img, (112, 112))#img.reshape((112, 112))
    
    img = img.float()
    img = ((img / 255) - 0.5) / 0.5 

    # img.div_(255).sub_(0.5).div_(0.5)
    
    return img 


#########################################
def load_training_arcfaces(fname):

    with open(fname) as f:
        labels = json.load(f)
        
    #print(labels)
    labels = collections.OrderedDict(labels)
    
    most_sim_arcs = dict()
    for k in labels.keys():
        id_key = k.split("_")[0]
        if id_key in most_sim_arcs: 
            break

        most_sim_arcs[id_key] = torch.from_numpy(np.array(labels[k]))[None, :].to(device)

    labels_names = [fname.replace('\\', '/') for fname in labels.keys()]
    labels = [labels[fname.replace('\\', '/')] for fname in labels.keys()]
    
    

    #return 
    #labels = np.array(list(labels.values()))
    labels = np.array(labels)
    labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
    return labels, labels_names, most_sim_arcs


#########################################
# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = torch.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        
        
    vectors.append(v)
    
    #vectors = np.array(vectors)
    return torch.cat(vectors, dim=0)

#########################################
def is_unique_id(tmp_arc, eval_arcs, too_similar_threshold):
    for arc_key, arc_val in eval_arcs.items(): 
        sim = cosine_similarity_torch(tmp_arc, arc_val)

        if sim.abs() > too_similar_threshold:
            if logging: print("\nNot new, it's:", arc_key, "  sim:", sim)
            # print("Sim to previous ids:", sim)
            return False 
    
    return True 

#########################################
def compute_euclidean_distance(a, b):
    euclid_dist = sum(((a - b)**2).reshape(512))
    return euclid_dist

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

from torchvision.utils import save_image

def save_images(img_list, NIR_img_list, folder, id):

    folder = os.path.join(folder, f"id_{id}")
    folder_VIS = os.path.join(folder, "VIS")
    folder_NIR = os.path.join(folder, "NIR")
    os.makedirs(folder_VIS, exist_ok=True)
    os.makedirs(folder_NIR, exist_ok=True)

    for i, img in enumerate(img_list):
        NIR_img = NIR_img_list[i]
        save_image(img, os.path.join(folder_VIS, f"sample_{i}_VIS.png"), normalize=True, range=(-1, 1))
        save_image(NIR_img, os.path.join(folder_NIR, f"sample_{i}_NIR.png"), normalize=True, range=(-1, 1))

#########################################
#    

def compare_to_samples_of_same_id(tmp_arcface_feat, first_arc, sample_arcs_list):
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
            #bad_sample =  True 
            return False 
        
    return True 

###########################################
def detect_face(tmp_img):
    
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
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    print("Current directory", os.getcwd())
    # NEW MODEL 
    
    # name = "07_08_NORM_ARC_label_NIR_weight_01_CONT_05_tufts_256_poses_1-7_HALF_IDs-000480" 
    name = "17_08_NORM_ARC_MOST_SIM_label_CONTINUE_NIR_weight_05_tufts_256_poses_1-7_HALF_IDs-001042-000160"# TODO was "31_05_ID_split_Norm_ARC_most_sim_CONT_NIR_w_05_tufts_256_poses_1-7__801"  #"24_05_NORM_ARC_MOST_SIM_label_CONT_05_561"
    # TODO was "31_05_ID_split_Norm_ARC_most_sim_CONT_NIR_w_05_tufts_256_poses_1"
    # name = "31_05_with_loss_w_01_ID_split_Norm_ARC_most_sim_CONT_NIR_w_05_tufts_256_poses_1-7_001362-7-000561"
    #name = "24_05_NORM_ARC_label_CONT_05_801"

    ckpt = "EXPERIMENTS_ArcBiFaceGAN/"+ name  +".pkl"

    

    seed = 0
    device = "cuda:1"
    
    num_ids = 1000 #1000 #250 #95 #95
    samples_per_id = 32 #25
    
    output_folder = "SYNTHETIC_DATASETS/Fast_improved_ids_" + str(num_ids) + "_samples_" + str(samples_per_id) + "_" + name

    arcface_multiplier = 4
    
    general_truncation_psi = 0.7 #0.7  


    save_grids = True
    save_NIR = True

    normal_sample = False
    training_arc = False
    
    change_slightly = False 
    
    compare_between_each_other = False 
    experiment_z_sim = False 


    logging = False

    # za dva standard deviationa razlike ... torej skor večina populacije .... .... 0.589
    # za tri standard deviatione ... je pa že insane ... samo 1% 
    # NOTE without blurry images ... and only a single set 
    avg_similarity = 0.7764 
    std_similarity = 0.0773
    
    threshold_for_same_ID = avg_similarity - 2 * std_similarity
    threshold_for_same_ID = torch.from_numpy(np.array([threshold_for_same_ID])).to(device) 
    threshold_for_too_similar = avg_similarity + std_similarity


    print("Seed:", seed)
    #set_all_seeds(seed)

    print(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    network_pkl = ckpt
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)


    
    
    
    arcface_model = prepare_locked_ArcFace_model().to(device)     

    mtcnn_model = MTCNN(select_largest=False, device=device)

    
    avg_arc_sim = []

    img_list = []

    # TODO most similar
    arcface_dataset_file = 'DATASETS/tufts_256_poses_1-7_aligned/train_labels_arcface_most_sim/dataset.json'    
    #arcface_dataset_file = 'DATASETS/tufts_256_poses_1-7_aligned/train_labels_arcface/dataset.json'    
    arcface_train_list, arc_file_names, all_arcs = load_training_arcfaces(arcface_dataset_file)

    
    #print(arcface_train_list)
    print("Training arcs most sim:", all_arcs.keys())
    set_all_seeds(seed)
    nrow = 4 
    
    
    first_id = None
    id = 0 

    pbar = tqdm(total=num_ids)

    #first_arc = [] 
    while id < num_ids:
        
       
        #arcface_start = torch.nn.functional.normalize(torch.randn(1, 512, device=device), p=2, dim=1 )
        arcface_start = torch.randn(1, 512, device=device)
        if logging: print("Initial max arc:", arcface_start.max(), " Min:", arcface_start.min(),)

        arcface_start = torch.nn.functional.normalize(arcface_start, p=2, dim=1)

        arcface_start = arcface_multiplier * arcface_start #TODO # was 8

        if logging: print("NEW Max arc:", arcface_start.max(), " Min:", arcface_start.min(),)
    
        
        z_start = torch.randn(1, 512, device=device)
        arcface_feat = arcface_start # +  torch.nn.functional.normalize( torch.randn(1, 512, device=device) , p=2, dim=1) #arcface_start


        img_list = []; NIR_img_list = []; avg_arc_sim = []; avg_arc_sim_to_train = []; tmp_arcs_for_comparison = []
        samples_to_compare = random.sample(range(samples_per_id), 1)

        sample_arcs_list = []

        current_sample = 0
        first_z = None
        attempts = 0
        max_attempts = 2500
        
        is_face_detected = False 

        is_first_unique = False 
        enough_samples_per_id = True

        while current_sample < samples_per_id:
            #print(".", end="")
            attempts += 1 
            if attempts > max_attempts: 
                print("\nToo many failed samples... Skipping to new ID.")
                enough_samples_per_id = False
                break 
            
            random_change = torch.randn(1, 512, device=device) #torch.randn_like(arcface_start, device=device)
            z_feat = random_change #0.5 * z_start + 0.5 * random_change #0.7 * z_start + 0.3 * random_change # z_start + random_change
            
            img, img_NIR = generate_image_from_z(z_feat, arcface_feat)    
            tmp_img = preprocess_generated_image_general(img)

            # check if a face is detected in the first sample 
            if  current_sample == 0: 
                is_face_detected = detect_face(tmp_img)

                # if not, generate new id            
                if not is_face_detected:
                    break 

            # extract arcface feature from image and normalize 
            img_tmp = prepare_for_arcface_model_torch(tmp_img)#preprocess_generated_image_for_ArcFace_torch(img)
            tmp_arcface_feat = arcface_model(img_tmp)
            tmp_arcface_feat = torch.nn.functional.normalize(tmp_arcface_feat,  p=2, dim=1 )

            # check if the first sample is a unique id
            # TODO druga ideja:  if first sample of ID is unique ... then also check if others are (if not then generate new sample) let the counter quit it
            # previous problem was we quit them even if only one was not good
            
            # if first sample isn't unique, find new ID
            new_unique_id = is_unique_id(tmp_arcface_feat, all_arcs, threshold_for_same_ID)
            if  current_sample == 0:
                if not new_unique_id: break 
                else: is_first_unique = True 

            # check if other samples are also unique, but don't break entire loop if not   
            else: 
                if not new_unique_id: continue 
            ######################

            # Compare between samples 
            if current_sample == 0: 
                first_arc = tmp_arcface_feat
            else:
                similar_to_samples_of_same_id = compare_to_samples_of_same_id(tmp_arcface_feat,first_arc, sample_arcs_list)
                if not similar_to_samples_of_same_id: continue 

            # TODO ... cleanup stuff
            #if current_sample in samples_to_compare: 
            tmp_arcs_for_comparison.append(tmp_arcface_feat)

            
            # append to lists
            img_list.append(img)
            sample_arcs_list.append(tmp_arcface_feat)

            if save_NIR: NIR_img_list.append(img_NIR)

            current_sample+=1 


        if not is_face_detected or not new_unique_id or not enough_samples_per_id:
            #print("Do not save, skip to next id.")
            continue 
        
        else:
            all_arcs["sample_" + str(id)] = first_arc #tmp_arcface_feat # TODO should be first arc
        

        # save images 
        grid = utils.save_image(
                torch.cat(img_list, dim=0),
                #torch.cat(full_list, 0),
                f"{output_folder}/id_{id}_samples_VIS.png",#_{ind}.png",
                normalize=True,
                range=(-1, 1),
                nrow=int(int(samples_per_id)/nrow),
            )

        if save_NIR: 
            grid = utils.save_image(
                torch.cat(NIR_img_list, dim=0),
                f"{output_folder}/id_{id}_samples_NIR.png",#_{ind}.png",
                normalize=True,
                range=(-1, 1),
                nrow=int(int(samples_per_id)/nrow)#int(int(n_sample)/5),
            )
        
        save_images(img_list, NIR_img_list, output_folder, id)

        # print("\n")
        # print("===" * 20)

        
        id+=1
        # print("Generate new id", id)
        pbar.update(1)

    pbar.close()