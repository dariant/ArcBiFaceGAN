import json 
import os
import re
import numpy as np 
import click
import collections
from PIL import Image 
from tqdm import tqdm
from Arcface_files.ArcFace_functions import preprocess_image_for_ArcFace, prepare_locked_ArcFace_model

#########################################
def atoi(text):
    return int(text) if text.isdigit() else text

#########################################
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


#########################################
def cosine_similarity(x, y, ax):
    dot = np.sum(np.multiply(x, y), axis=ax)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    similarity = np.clip(dot/norm, -1., 1.)
    return similarity


#########################################
def compare_mean_to_group(mean_arc, group_arcs):
    sims = []
    
    for i in group_arcs:
        tmp_sim = cosine_similarity(mean_arc, group_arcs, ax=1)
        sims.append(tmp_sim)

        
        
    avg_sim = np.mean(sims)
    std_sim = np.std(sims)
    return avg_sim, std_sim

#########################################
def compare_all_combinations(group_arcs):
    sims = []
    avg_for_each = []
    for i, a in enumerate(group_arcs):
        sim_each_img = []
        for j, b in enumerate(group_arcs):        
            if i ==j:
                continue    
            #print(len(i))
            tmp_sim = cosine_similarity(a, b, ax=0)
            sims.append(tmp_sim)
            sim_each_img.append(tmp_sim)
            
        avg_for_each.append(np.mean(sim_each_img))
        #print(avg_for_each)
    avg_sim = np.mean(sims)
    std_sim = np.std(sims)

    max_ind = np.argmax(avg_for_each)
    most_sim = group_arcs[max_ind]
    #print("Most similar img:", max_ind, "--", avg_for_each[max_ind])
    
    return avg_sim, std_sim, most_sim


#########################################


@click.command()

# Configuration 
@click.option('--data_folder', help='Training data (directory or zip)', metavar='PATH', required=True)
@click.option('--rec_model', help='Path to recognition model (pth file)', required=True)
@click.option('--gpu_device_number', help='Which CUDA gpu to use [default: 0]', type=int, default=0, metavar='INT')
@click.option('--all_or_one', help='Create one feature for each image [all], or one feature for each identity [one] [default: all]', default="all")

def main(**args):
    print("Config:", args)

    data_folder = args['data_folder']
    device = f"cuda:{args['gpu_device_number']}"
    path_to_VIS_folder = os.path.join(data_folder, "VIS")

    fnames = os.listdir(path_to_VIS_folder)
    
    arcface_model = prepare_locked_ArcFace_model(args["rec_model"]).to(device)
    print(fnames)
    
    fnames.sort(key=natural_keys)
    # Create arcface features for each image 
    print("Create arcface features")
    group_dict = dict()
    for fname in tqdm(fnames):
        id = int(fname.split("_")[0])
        
        if id not in group_dict:
            group_dict[id] = dict()

        img_path = os.path.join(path_to_VIS_folder, fname)
        img = Image.open(img_path)
        
        img = preprocess_image_for_ArcFace(img).to(device)
        arcface_feature = arcface_model(img).cpu().detach()
        arcface_feature = np.array(arcface_feature)[np.newaxis, ...]    
        arcface_feature = np.transpose(arcface_feature, axes=[2, 1, 0])
        arcface_feature = arcface_feature[:, 0, 0]
        
        group_dict[id][fname] = arcface_feature


    label_map = collections.OrderedDict() 

    print("Select all or one:", args["all_or_one"])

    # Find most similar identity feature for each identity 
    if args["all_or_one"] == "one":
        map_id_to_arc = dict()
        for id in tqdm(group_dict):
            all_avg_sim, std_all_sim, most_sim = compare_all_combinations(list(group_dict[id].values()))
            map_id_to_arc[id] = most_sim

        # Construct .json file 
        for fname in tqdm(fnames):
            id = int(fname.split("_")[0])
            arc_tmp = map_id_to_arc[id]
            label_map[fname] = arc_tmp.tolist()
            
    # save an identity feature for each image
    else: 
        for fname in fnames:
            id = int(fname.split("_")[0])
            #arc_tmp = np.load(data_folder + "arcface_features/" + fname.split(".")[0] + ".npy")
            arc_tmp = group_dict[id][fname]
            label_map[fname] = arc_tmp.tolist()

    print("Save to .json")
    with open(data_folder + "identity_features.json", "w") as outfile:
        json.dump(label_map, outfile)


if __name__ == "__main__":
    main()

