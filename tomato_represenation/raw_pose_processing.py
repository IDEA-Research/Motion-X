# coding=utf-8
# Copyright 2022 The IDEA Authors (Shunlin Lu and Ling-Hao Chen). All rights reserved.
#
# For all the datasets, be sure to read and follow their license agreements,
# and cite them accordingly.
# If the unifier is used in your research, please consider to cite as:
#
# @article{humantomato,
#   title={HumanTOMATO: Text-aligned Whole-body Motion Generation},
#   author={Lu, Shunlin and Chen, Ling-Hao and Zeng, Ailing and Lin, Jing and Zhang, Ruimao and Zhang, Lei and Shum, Heung-Yeung},
#   journal={arxiv:2310.12978},
#   year={2023}
# }
#
# @InProceedings{Guo_2022_CVPR,
#     author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
#     title     = {Generating Diverse and Natural 3D Human Motions From Text},
#     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#     month     = {June},
#     year      = {2022},
#     pages     = {5152-5161}
# }
#
# Licensed under the IDEA License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/IDEA-Research/HumanTOMATO/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. We provide a license to use the code, 
# please read the specific details carefully.
#
# ------------------------------------------------------------------------------------------------
# Copyright (c) Chuan Guo.
# ------------------------------------------------------------------------------------------------
# Portions of this code were adapted from the following open-source project:
# https://github.com/EricGuo5513/HumanML3D
# ------------------------------------------------------------------------------------------------

import sys
import os
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
    """
    Recursively find all files in the specified directory.

    Args:
        base (str): The base directory to start the search.

    Returns:
        list: A list of file paths found in the directory and its subdirectories.
    """
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


# Get SMPLX layer and model using a custom function get_smplx_layer
smplx_layer, smplx_model = get_smplx_layer(comp_device)

# change your path here with Motion-X SMPLX format with 322 dims
train_dataset = MotionDatasetV2(root_path='motion_data/smplx_322', debug=False)
train_loader = DataLoader(train_dataset, batch_size=8, drop_last=False,
                          num_workers=4, shuffle=False, collate_fn=mld_collate)


def amass_to_pose(src_motion, src_path, length):
    """
    Convert AMASS SMPL-X motion data to pose representation and save joint positions.

    Args:
        src_motion (torch.Tensor): Input SMPL-X motion data.
        src_path (list): List of paths to the source motion data.
        length (list): List of motion sequence lengths.

    Returns:
        None
    """
    # frame id of the mocap sequence
    fId = 0
    pose_seq = []

    # Process SMPLX 322-dimensional data
    vert, joints, pose, faces = process_smplx_322_data(
        src_motion, smplx_layer, smplx_model, device=comp_device)

    # Add global joint offsets to the processed joints
    joints += src_motion[..., 309:312].unsqueeze(2)

    # Iterate over frames to extract joint positions and save them to individual files
    for i in range(joints.shape[0]):
        joint = joints[i][:int(length[i])].detach().cpu().numpy()
        # change the save folder
        save_path = src_path[i].replace('/smplx_322/', '/joint/')
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        np.save(save_path, joint)


# Iterate over batches in the training loader using tqdm for progress tracking
for batch_data in tqdm(train_loader):
    # Move motion data to the computation device (e.g., GPU)
    motion = batch_data['motion'].to(comp_device)
    name = batch_data['name']
    length = batch_data['length']

    # Call the 'amass_to_pose' function to convert SMPL-X motion data to pose representation
    # and save joint positions for each batch
    amass_to_pose(motion, name, length)
