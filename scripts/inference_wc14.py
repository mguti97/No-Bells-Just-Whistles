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
    parser.add_argument('--use_gt', action='store_true', help='Use ground truth (default: False)')


    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    files = glob.glob(os.path.join(args.root_dir + args.split, "*.jpg"))

    if args.use_gt:
        zip_name_pred = args.save_dir + args.split + '_gt.zip'
    else:
        zip_name_pred = args.save_dir + args.split + '_pred.zip'


    if args.use_gt:
        device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

        with zipfile.ZipFile(zip_name_pred, 'w') as zip_file:
            for file in tqdm(files, desc="Processing Images"):
                image = Image.open(file)
                w, h = image.size

                homography_file = args.root_dir + args.split + '/' + \
                                  file.split('/')[-1].split('.')[0] + '.homographyMatrix'

                json_path = file.split('.')[0] + ".json"
                f = open(json_path)
                data = json.load(f)
                kp_db = KeypointsDB(data, image)
                line_db = LineKeypointsDB(data, image)
                heatmaps, _ = kp_db.get_tensor_w_mask()
                heatmaps = torch.tensor(heatmaps).unsqueeze(0)
                heatmaps_l = line_db.get_tensor()
                heatmaps_l = torch.tensor(heatmaps_l).unsqueeze(0)
                kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
                line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
                kp_dict = coords_to_dict(kp_coords, threshold=0.0378)
                lines_dict = coords_to_dict(line_coords, threshold=0.0634)
                final_dict = complete_keypoints(kp_dict, lines_dict, w=w, h=h, normalize=True)

                json_data = json.dumps(final_dict)
                zip_file.writestr(f"{file.split('/')[-1].split('.')[0]}.json", json_data)


    else:
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
            for file in tqdm(files, desc="Processing Images"):
                image = Image.open(file)
                image = f.to_tensor(image).float().to(device).unsqueeze(0)
                image = image if image.size()[-1] == 960 else transform(image)
                b, c, h, w = image.size()

                homography_file = args.root_dir + args.split + '/' + \
                                  file.split('/')[-1].split('.')[0] + '.homographyMatrix'

                with torch.no_grad():
                    heatmaps = model(image)
                    heatmaps_l = model_l(image)

                    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
                    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
                    kp_dict = coords_to_dict(kp_coords, threshold=0.0072, ground_plane_only=True)
                    lines_dict = coords_to_dict(line_coords, threshold=0.1962, ground_plane_only=True)
                    final_dict = complete_keypoints(kp_dict, lines_dict, w=w, h=h, normalize=True)

                    json_data = json.dumps(final_dict)
                    zip_file.writestr(f"{file.split('/')[-1].split('.')[0]}.json", json_data)







