import cv2
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as f

from tqdm import tqdm
from PIL import Image
from matplotlib.patches import Polygon

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l

from utils.utils_calib import FramebyFrameCalib, pan_tilt_roll_to_orientation
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    complete_keypoints, coords_to_dict


lines_coords = [[[0., 54.16, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [0., 13.84, 0.]],
                [[88.5, 54.16, 0.], [105., 54.16, 0.]],
                [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
                [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                [[0., 37.66, -2.44], [0., 30.34, -2.44]],
                [[0., 37.66, 0.], [0., 37.66, -2.44]],
                [[0., 30.34, 0.], [0., 30.34, -2.44]],
                [[105., 37.66, -2.44], [105., 30.34, -2.44]],
                [[105., 30.34, 0.], [105., 30.34, -2.44]],
                [[105., 37.66, 0.], [105., 37.66, -2.44]],
                [[52.5, 0., 0.], [52.5, 68, 0.]],
                [[0., 68., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [0., 68., 0.]],
                [[105., 0., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [105., 0., 0.]],
                [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
                [[5.5, 24.84, 0.], [0., 24.84, 0.]],
                [[99.5, 43.16, 0.], [105., 43.16, 0.]],
                [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
                [[99.5, 24.84, 0.], [105., 24.84, 0.]]]


def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]])
    P = Q @ (rotation @ It)

    return P


def inference(cam, frame, model, model_l):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    frame = f.to_tensor(frame).float().unsqueeze(0)
    _, _, h_original, w_original = frame.size()
    frame = frame if frame.size()[-1] == 960 else transform2(frame)
    frame = frame.to(device)
    b, c, h, w = frame.size()

    with torch.no_grad():
        heatmaps = model(frame)
        heatmaps_l = model_l(frame)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
    kp_dict = coords_to_dict(kp_coords, threshold=0.1486)
    lines_dict = coords_to_dict(line_coords, threshold=0.388)
    final_dict = complete_keypoints(kp_dict, lines_dict, w=w, h=h, normalize=True)

    cam.update(final_dict[0])
    final_params_dict = cam.heuristic_voting()

    return final_params_dict


def project(frame, P):

    for line in lines_coords:
        w1 = line[0]
        w2 = line[1]
        i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
        i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
        i1 /= i1[-1]
        i2 /= i2[-1]
        frame = cv2.line(frame, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), (255, 0, 0), 3)

    r = 9.15
    pts1, pts2, pts3 = [], [], []
    base_pos = np.array([11-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(37, 143, 50):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts1.append([ipos[0], ipos[1]])

    base_pos = np.array([94-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(217, 323, 200):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts2.append([ipos[0], ipos[1]])

    base_pos = np.array([0, 0, 0., 0.])
    for ang in np.linspace(0, 360, 500):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts3.append([ipos[0], ipos[1]])

    XEllipse1 = np.array(pts1, np.int32)
    XEllipse2 = np.array(pts2, np.int32)
    XEllipse3 = np.array(pts2, np.int32)
    frame = cv2.polylines(frame, [XEllipse1], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [XEllipse2], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [XEllipse3], False, (255, 0, 0), 3)

    return frame


def process_video(video_path, model_kp, model_line, save_path=None, display=False):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)

    if save_path:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            final_params_dict = inference(cam, frame, model, model_l)
            if final_params_dict is not None:
                P = projection_from_cam_params(final_params_dict)
                projected_frame = project(frame, P)
            else:
                projected_frame = frame

            if display:
                cv2.imshow('Projected Frame', projected_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if save_path:
                out.write(projected_frame)

            pbar.update(1)

    cap.release()
    if save_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process video and plot lines on each frame.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the (kp model) configuration file")
    parser.add_argument("--cfg_l", type=str, required=True, help="Path to the (line model) configuration file")
    parser.add_argument("--weights_kp", type=str, help="Path to the model for keypoint inference.")
    parser.add_argument("--weights_line", type=str, help="Path to the model for line projection.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CPU or CUDA device index")
    parser.add_argument("--video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--save_path", type=str, default="", help="Path to save the processed video.")
    parser.add_argument("--display", action="store_true", help="Enable real-time display.")
    args = parser.parse_args()

    video_path = args.video_path
    model_kp = args.weights_kp
    model_line = args.weights_line
    save_path = args.save_path
    device = args.device
    display = args.display

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

    transform2 = T.Resize((540, 960))

    process_video(video_path, model_kp, model_line, save_path, display)