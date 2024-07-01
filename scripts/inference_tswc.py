import os
import sys
import json
import glob
import yaml
import torch
import zipfile
import argparse
import warnings
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as f

from tqdm import tqdm
from PIL import Image

#sys.path.append("../")
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    complete_keypoints, coords_to_dict
from utils.utils_keypoints import KeypointsDB
from utils.utils_lines import LineKeypointsDB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to the (kp model) configuration file")
    parser.add_argument("--cfg_l", type=str, required=True,
                        help="Path to the (line model) configuration file")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory")
    parser.add_argument("--split", type=str, required=True,
                        help="Dataset split")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Root directory")
    parser.add_argument("--weights_kp", type=str, required=True,
                        help="Model (keypoints) weigths to use")
    parser.add_argument("--weights_line", type=str, required=True,
                        help="Model (lines) weigths to use")
    parser.add_argument("--cuda", type=str, default="cuda:0",
                        help="CUDA device index (default: 'cuda:0')")
    parser.add_argument("--kp_th", type=float, default=0.3457, help="Keypoint threshold")
    parser.add_argument("--line_th", type=float, default=0.370, help="Line threshold")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")


    args = parser.parse_args()
    return args


def get_files(file_paths):
    jpg_files = []
    for file_path in file_paths:
        directory_path = os.path.join(os.path.join(args.root_dir, "Dataset/80_95"), file_path)
        if os.path.exists(directory_path):
            files = os.listdir(directory_path)
            jpg_files.extend([os.path.join(directory_path, file) for file in files if file.endswith('.jpg')])

    jpg_files = sorted(jpg_files)
    return jpg_files

def get_homographies(file_paths):
    npy_files = []
    for file_path in file_paths:
        directory_path = os.path.join(os.path.join(args.root_dir, "Annotations/80_95"), file_path)
        if os.path.exists(directory_path):
            files = os.listdir(directory_path)
            npy_files.extend([os.path.join(directory_path, file) for file in files if file.endswith('.npy')])

    npy_files = sorted(npy_files)
    return npy_files


def make_file_name(file):
    file =  "TS-WorldCup/" + file.split("TS-WorldCup/")[-1]
    splits = file.split('/')
    side = splits[3]
    match = splits[4]
    image = splits[5]
    frame = 'IMG_' + image.split('.')[0].split('_')[-1]
    return side + '-' + match + '-' + frame


if __name__ == "__main__":
    args = parse_args()

    with open(args.root_dir + args.split + '.txt', 'r') as file:
        # Read lines from the file and remove trailing newline characters
        seqs = [line.strip() for line in file.readlines()]

    files = get_files(seqs)
    homographies = get_homographies(seqs)

    zip_name_pred = args.save_dir + args.split + '_pred.zip'

    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    cfg = yaml.safe_load(open(args.cfg, 'r'))
    cfg_l = yaml.safe_load(open(args.cfg_l, 'r'))

    loaded_state = torch.load(args.weights_kp, map_location=device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(device)
    model.eval()

    loaded_state_l = torch.load(args.weights_line, map_location=device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(device)
    model_l.eval()

    transform = T.Resize((540, 960))

    with zipfile.ZipFile(zip_name_pred, 'w') as zip_file:
        for count in tqdm(range(len(files)), desc="Processing Images"):
            image = Image.open(files[count])
            image = f.to_tensor(image).float().to(device).unsqueeze(0)
            image = image if image.size()[-1] == 960 else transform(image)
            b, c, h, w = image.size()


            with torch.no_grad():
                heatmaps = model(image)
                heatmaps_l = model_l(image)

                kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
                line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
                kp_dict = coords_to_dict(kp_coords, threshold=args.kp_th, ground_plane_only=True)
                lines_dict = coords_to_dict(line_coords, threshold=args.line_th, ground_plane_only=True)
                final_dict = complete_keypoints(kp_dict, lines_dict, w=w, h=h, normalize=True)

                json_data = json.dumps(final_dict)
                zip_file.writestr(f"{make_file_name(files[count])}.json", json_data)







