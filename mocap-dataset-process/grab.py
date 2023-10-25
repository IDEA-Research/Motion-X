
import pandas as pd
import os
import pickle
import numpy as np
from utils.rotation_conversions import *
from smplx import SMPLX
from utils.face_z_align_util import joint_idx, face_z_transform
import re
from tqdm import tqdm




def compute_canonical_transform(global_orient):
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=global_orient.dtype)
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path



def transform_motions(data):

    ex_fps = 20

    fps = 120

    down_sample = int(fps / ex_fps)


    frame_number = data['body']['params']['transl'].shape[0]
    


    fId = 0 # frame id of the mocap sequence
    pose_seq = []

    

    for fId in range(0, frame_number, down_sample):
        pose_root = data['body']['params']['global_orient'][fId:fId+1]
        pose_root = compute_canonical_transform(torch.from_numpy(pose_root)).detach().cpu().numpy()
        pose_body = data['body']['params']['body_pose'][fId:fId+1]

        ### grab hand pose is 24 * 2 dim (after PCA), which is not compatible with our representation, thus use zeros here. ####
        pose_left_hand = np.zeros((1, 45))
        pose_right_hand = np.zeros((1, 45))


        pose_jaw = data['body']['params']['jaw_pose'][fId:fId+1]

        ####grab expression is 10-dim, so we use zeros

        pose_expression = np.zeros((1, 50))
        pose_face_shape = np.zeros((1, 100))

        pose_trans = data['body']['params']['transl'][fId:fId+1]

        pose_body_shape = np.zeros((1, 10))
        pose = np.concatenate((pose_root, pose_body, pose_left_hand, pose_right_hand, pose_jaw, pose_expression, pose_face_shape, pose_trans, pose_body_shape), axis=1)
        pose_seq.append(pose)

    pose_seq = np.concatenate(pose_seq, axis=0)

    return pose_seq

def create_parent_dir(target_t2m_joints_case_path):
    target_t2m_joints_parent_directory = os.path.dirname(target_t2m_joints_case_path)

    if not os.path.exists(target_t2m_joints_parent_directory):
        os.makedirs(target_t2m_joints_parent_directory)


def process_text(text):
    result = re.sub(r'_\d+', '', text)
    result = result.replace('_', ' ')
    result = re.sub(r'\s+', ' ', result)
    return result

if __name__ == '__main__':

    for case_path in findAllFile('./GRAB'):
        data = np.load(case_path, allow_pickle=True)
        data = {k: data[k].item() for k in data.files}
        data = transform_motions(data)
        text = os.path.basename(case_path).replace('.npz', '')
        text = process_text(text)

        output_motion_path = case_path.replace('GRAB', 'GRAB_motion').replace('.npz', '.npy')
        output_text_path = case_path.replace('GRAB', 'GRAB_text').replace('.npz', '.txt')
        create_parent_dir(output_motion_path)
        create_parent_dir(output_text_path)
        np.save(output_motion_path, data)
        with open(output_text_path, 'w') as f:
            f.write(text)

        
