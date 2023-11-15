import sys, os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from smplx2joints import get_smplx_layer, process_smplx_322_data
from dataset import MotionDatasetV2, mld_collate
from torch.utils.data import DataLoader

os.environ['PYOPENGL_PLATFORM'] = 'egl'

comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path



smplx_layer, smplx_model = get_smplx_layer(comp_device)

## change your path here with Motion-X SMPLX format with 322 dims
train_dataset = MotionDatasetV2(root_path='motion_data/smplx_322', debug=False)
train_loader = DataLoader(train_dataset, batch_size=8, drop_last=False, num_workers=4, shuffle=False, collate_fn=mld_collate)

def amass_to_pose(src_motion, src_path, length):

    # src_motion = src_motion.to(comp_device)
    # frame_number = src_motion.shape[0]
    # 
    fId = 0 # frame id of the mocap sequence
    pose_seq = []
    vert, joints, pose, faces = process_smplx_322_data(src_motion, smplx_layer, smplx_model, device=comp_device)

    joints += src_motion[..., 309:312].unsqueeze(2)
    for i in range(joints.shape[0]):
        joint = joints[i][:int(length[i])].detach().cpu().numpy()
        # change the save folder 
        save_path = src_path[i].replace('/smplx_322/', '/joint/')
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        np.save(save_path, joint)

    



for batch_data in tqdm(train_loader):
    motion = batch_data['motion'].to(comp_device)
    name = batch_data['name']
    length = batch_data['length']

    amass_to_pose(motion, name, length)




    