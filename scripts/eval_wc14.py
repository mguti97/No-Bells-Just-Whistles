import os
import sys
import json
import glob
import torch
import zipfile
import argparse
import numpy as np

from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.utils_calib import FramebyFrameCalib
from model.metrics import calc_iou_part, calc_iou_whole_with_poly, calc_reproj_error, calc_proj_error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)

    args = parser.parse_args()
    return args


def pan_tilt_roll_to_orientation(pan, tilt, roll):
    """
    Conversion from euler angles to orientation matrix.
    :param pan:
    :param tilt:
    :param roll:
    :return: orientation matrix
    """
    Rpan = np.array([
        [np.cos(pan), -np.sin(pan), 0],
        [np.sin(pan), np.cos(pan), 0],
        [0, 0, 1]])
    Rroll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]])
    Rtilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt), np.cos(tilt)]])
    rotMat = np.dot(Rpan, np.dot(Rtilt, Rroll))
    return rotMat

def get_sn_homography(cam_params: dict, batch_size=1):
    # Extract relevant camera parameters from the dictionary
    pan_degrees = cam_params['cam_params']['pan_degrees']
    tilt_degrees = cam_params['cam_params']['tilt_degrees']
    roll_degrees = cam_params['cam_params']['roll_degrees']
    x_focal_length = cam_params['cam_params']['x_focal_length']
    y_focal_length = cam_params['cam_params']['y_focal_length']
    principal_point = np.array(cam_params['cam_params']['principal_point'])
    position_meters = np.array(cam_params['cam_params']['position_meters'])

    pan = pan_degrees * np.pi / 180.
    tilt = tilt_degrees * np.pi / 180.
    roll = roll_degrees * np.pi / 180.

    rotation = np.array([[-np.sin(pan) * np.sin(roll) * np.cos(tilt) + np.cos(pan) * np.cos(roll),
                          np.sin(pan) * np.cos(roll) + np.sin(roll) * np.cos(pan) * np.cos(tilt), np.sin(roll) * np.sin(tilt)],
                         [-np.sin(pan) * np.cos(roll) * np.cos(tilt) - np.sin(roll) * np.cos(pan),
                          -np.sin(pan) * np.sin(roll) + np.cos(pan) * np.cos(roll) * np.cos(tilt), np.sin(tilt) * np.cos(roll)],
                         [np.sin(pan) * np.sin(tilt), -np.sin(tilt) * np.cos(pan), np.cos(tilt)]], dtype='float')

    rotation = np.transpose(pan_tilt_roll_to_orientation(pan, tilt, roll))

def convert_homography_SN_to_WC14(H):
    T = np.eye(3)
    T[0, -1] = 105 / 2
    T[1, -1] = 68 / 2
    meter2yard = 1.09361
    S = np.eye(3)
    S[0, 0] = meter2yard
    S[1, 1] = meter2yard
    H_SN = S @ (T @ H)
    return H_SN

def get_homography_by_index(homography_file):
    with open(homography_file, 'r') as file:
        lines = file.readlines()
        matrix_elements = []
        for line in lines:
            matrix_elements.extend([float(element) for element in line.split()])
    homography = np.array(matrix_elements).reshape((3, 3))
    homography = homography / homography[2:3, 2:3]
    return homography

if __name__ == "__main__":
    args = parse_args()

    missed = 0
    iou_part_list, iou_whole_list = [], []
    rep_err_list, proj_err_list = [], []

    homographies = glob.glob(os.path.join(args.root_dir + args.split, "*.homographyMatrix"))
    prediction_archive = zipfile.ZipFile(args.pred_file, 'r')
    cam = FramebyFrameCalib(1280, 720, denormalize=True)

    for h_gt in tqdm(homographies):
        file_name = h_gt.split('/')[-1].split('.')[0]
        pred_name = file_name + '.json'

        if pred_name not in prediction_archive.namelist():
            missed += 1
            continue

        homography_gt = get_homography_by_index(h_gt)
        keypoints_dict = prediction_archive.read(pred_name)
        keypoints_dict = json.loads(keypoints_dict.decode('utf-8'))[0]
        keypoints_dict = {int(key): value for key, value in keypoints_dict.items()}

        cam.update(keypoints_dict)
        homography_pred = cam.get_homography_from_ground_plane(use_ransac=15, inverse=True)
        if homography_pred is None:
            missed += 1
            continue
        #homography_pred = cam.get_homography_from_3D_projection(use_ransac=5., inverse=True)
        homography_pred = convert_homography_SN_to_WC14(homography_pred)

        iou_p = calc_iou_part(homography_pred, homography_gt)
        iou_w, _, _ = calc_iou_whole_with_poly(homography_pred, homography_gt)
        rep_err = calc_reproj_error(homography_pred, homography_gt)
        proj_err = calc_proj_error(homography_pred, homography_gt)

        iou_part_list.append(iou_p)
        iou_whole_list.append(iou_w)
        rep_err_list.append(rep_err)
        proj_err_list.append(proj_err)


    print(f'Completeness: {1-missed/len(homographies)}')
    print('IOU Part')
    print(f'mean: {np.mean(iou_part_list)} \t median: {np.median(iou_part_list)}')
    print('\nIOU Whole')
    print(f'mean: {np.mean(iou_whole_list)} \t median: {np.median(iou_whole_list)}')
    print('\nReprojection Err.')
    print(f'mean: {np.mean(rep_err_list)} \t median: {np.median(rep_err_list)}')
    print('\nProjection Err. (meters)')
    print(f'mean: {np.mean(proj_err_list) * 0.9144} \t median: {np.median(proj_err_list) * 0.9144}')





