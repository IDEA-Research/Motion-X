import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

motion_folder = '../datasets/motion_data/smplx_322'

for mocap_dataset in ['humanml', 'EgoBody', 'GRAB']:

    mocap_motion_folder = os.path.join(motion_folder, mocap_dataset)
    mocap_motion_files = findAllFile(mocap_motion_folder)
    for mocap_motion_file in tqdm(mocap_motion_files):
        face_motion_file = mocap_motion_file.replace('/motion_data/', '/face_motion_data/')
        motion = np.load(mocap_motion_file)
        motion_length = motion.shape[0]
        face_motion = np.load(face_motion_file)

        # perform motion interpolation to avoid frame rate mismatch
        face_motion = torch.from_numpy(face_motion)
        face_motion = face_motion[None].permute(0, 2, 1)  # [1, num_feats, num_frames]
        face_motion = F.interpolate(face_motion, size=motion_length, mode='linear')  # motion interpolate
        face_motion = face_motion.permute(0, 2, 1)[0].numpy()  # [num_frames, num_feats]

        motion[:, 66 + 90:66 + 93], motion[:, 159:159 + 50], motion[:, 209:209 + 100] = \
            face_motion[:, :3], face_motion[:, 3:3+50], face_motion[:, 53:153]
        np.save(mocap_motion_file, motion)
