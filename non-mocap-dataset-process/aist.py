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

motion_folder = '../datasets/motion_data/smplx_322/aist'
motion_files = findAllFile(motion_folder)

# rescale trans of the multi-view dataset
for motion_file in tqdm(motion_files):
    motion = np.load(motion_file)
    trans = motion[:, 309:309 + 3]
    trans = trans/94.0
    trans[:, 2] = trans[:, 2] * (-1)
    motion[:, 309:309 + 3] = trans
    np.save(motion_file, motion)
