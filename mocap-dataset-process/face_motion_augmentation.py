import os
import numpy as np
from tqdm import tqdm

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
        face_motion = np.load(face_motion_file)
        motion[:, 66 + 90:66 + 93], motion[:, 159:159 + 50], motion[:, 209:209 + 100] = \
            face_motion[:, :3], face_motion[:, 3:3+50], face_motion[:, 53:153]
        np.save(mocap_motion_file, motion)
