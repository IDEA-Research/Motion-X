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

from os.path import join as pjoin

from common.skeleton import Skeleton
import numpy as np
import os
from common.quaternion import *
from paramUtil import *

import torch
from tqdm import tqdm
import os


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


def uniform_skeleton(positions, target_offset):
    """
    Uniformly scales a skeleton to match a target offset.

    Args:
        positions (numpy.ndarray): Input skeleton joint positions.
        target_offset (torch.Tensor): Target offset for the skeleton.

    Returns:
        numpy.ndarray: New joint positions after scaling and inverse/forward kinematics.
    """
    # Creating a skeleton with a predefined kinematic chain
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')

    # Calculate the global offset of the source skeleton
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    # Calculate Scale Ratio as the ratio of legs
    src_leg_len = np.abs(src_offset[l_idx1]).max(
    ) + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max(
    ) + np.abs(tgt_offset[l_idx2]).max()

    # Scale ratio for uniform scaling
    scale_rt = tgt_leg_len / src_leg_len
    
    # Extract the root position of the source skeleton
    src_root_pos = positions[:, 0]
    # Scale the root position based on the calculated ratio
    tgt_root_pos = src_root_pos * scale_rt

    # Inverse Kinematics to get quaternion parameters
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)

    # Forward Kinematics with the new root position and target offset
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def process_file(positions, feet_thre):
    """
    Processes motion capture data, including downsampling, skeleton normalization,
    floor alignment, and feature extraction.

    Args:
        positions (numpy.ndarray): Motion capture data (seq_len, joints_num, 3).
        feet_thre (float): Threshold for foot detection.

    Returns:
        tuple: A tuple containing processed data, global positions, aligned positions, and linear velocity.
    """
    # Uniformly scale the skeleton to match a target offset
    positions = uniform_skeleton(positions, tgt_offsets)
    
    # Put the skeleton on the floor by subtracting the minimum height
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # Center the skeleton at the origin in the XZ plane
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # Ensure the initial facing direction is along Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # Ensure that all poses initially face Z+
    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / \
        np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    # Calculate quaternion for root orientation
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    # Rotate the motion capture data using the calculated quaternion
    positions_b = positions.copy()
    positions = qrot_np(root_quat_init, positions)
    
    # Store the global positions for further analysis
    global_positions = positions.copy()

    # You can try to visualize it!
    # plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)
    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array(
            [thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z)
                  < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z)
                  < velfactor)).astype(np.float32)
        return feet_l, feet_r
    
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        """
        Adjusts the motion capture data to a local pose representation and ensures
        that all poses face in the Z+ direction.

        Args:
            positions (numpy.ndarray): Input motion capture data with shape (seq_len, joints_num, 3).

        Returns:
            numpy.ndarray: Adjusted motion capture data in a local pose representation.
        """
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(
            np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        """
        Computes quaternion parameters, root linear velocity, and root angular velocity
        based on the input motion capture data.

        Args:
            positions (numpy.ndarray): Input motion capture data with shape (seq_len, joints_num, 3).

        Returns:
            tuple: A tuple containing quaternion parameters, root angular velocity, root linear velocity, and root rotation.
        """
        # Initialize a skeleton object with a specified kinematic chain
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        """
        Computes continuous 6D parameters, root linear velocity, and root angular velocity
        based on the input motion capture data.

        Args:
            positions (numpy.ndarray): Input motion capture data with shape (seq_len, joints_num, 3).

        Returns:
            tuple: A tuple containing continuous 6D parameters, root angular velocity, root linear velocity, and root rotation.
        """
        # Initialize a skeleton object with a specified kinematic chain
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(
            positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot
    
    # Extract additional features including root height and root data
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    # Root height
    root_y = positions[:, 0, 1:2]

    # Root rotation and linear velocity
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    # Get Joint Rotation Representation
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    # Get Joint Rotation Invariant Position Represention
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    # Get Joint Velocity Representation
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    # Concatenate all features into a single array
    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)

def recover_root_rot_pos(data):
    """
    Recover root rotation and position from the given motion capture data.

    Args:
        data (torch.Tensor): Input motion capture data with shape (..., features).

    Returns:
        tuple: A tuple containing the recovered root rotation quaternion and root position.
    """
    # Extract root rotation velocity from the input data
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    """
    Recover joint positions from the given motion capture data using root rotation information.

    Args:
        data (torch.Tensor): Input motion capture data with shape (..., features).
        joints_num (int): Number of joints in the skeleton.
        skeleton (Skeleton): Skeleton object used for forward kinematics.

    Returns:
        torch.Tensor: Recovered joint positions.
    """
    # Recover root rotation quaternion and position from the input data
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    # Convert root rotation quaternion to continuous 6D representation
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    # Define indices for relevant features in the input data
    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6

    # Extract continuous 6D parameters from the input data
    cont6d_params = data[..., start_indx:end_indx]
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    # Perform forward kinematics to obtain joint positions
    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)
    
    return positions


def recover_from_ric(data, joints_num):
    """
    Recover joint positions from the given motion capture data using root rotation information.

    Args:
        data (torch.Tensor): Input motion capture data with shape (..., features).
        joints_num (int): Number of joints in the skeleton.

    Returns:
        torch.Tensor: Recovered joint positions.
    """
    # Recover root rotation quaternion and position from the input data
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(
        positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


'''
For HumanML3D Dataset
'''

if __name__ == "__main__":
    """
    This script processes motion capture data, performs a recovery operation on the joint positions, 
    and saves the recovered joint positions along with the original joint vectors. The main steps include:

    Note: Exception handling is implemented to identify and print any issues encountered during processing.

    Output:
    - Recovered joint positions are saved in the 'new_joints' directory.
    - Original joint vectors are saved in the 'new_joint_vecs' directory.
    """

    example_id = "000021"
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # body,hand joint idx
    # 2*3*5=30, left first, then right
    hand_joints_id = [i for i in range(25, 55)]
    body_joints_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 22 joints
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 52
    # ds_num = 8

    # change your motion_data joint
    data_dir = 'motion_data/joint'
    # change your save folder
    save_dir1 = 'motion_data/new_joints/'
    # change your save folder
    save_dir2 = 'motion_data/new_joint_vecs/'

    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_body_hand_kinematic_chain

    # Get offsets of target skeleton
    # we random choose one
    example_data = np.load('motion_data/joint/humanml/000021.npy')
    example_data = example_data[:, body_joints_id + hand_joints_id, :]
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)

    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')

    # (joints_num, 3)
    # tgt_offsets is the 000021 skeleton bone lengths with the predefined offset directions. global postion offsets
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    source_list = findAllFile(data_dir)
    frame_num = 0
    for source_file in tqdm(source_list):

        source_data = np.load(source_file)[:, body_joints_id+hand_joints_id, :]
        try:
            data, ground_positions, positions, l_velocity = process_file(
                source_data, 0.002)
            rec_ric_data = recover_from_ric(torch.from_numpy(
                data).unsqueeze(0).float(), joints_num)

            os.makedirs(os.path.split(source_file.replace(
                'joint', 'new_joints'))[0], exist_ok=True)
            os.makedirs(os.path.split(source_file.replace(
                'joint', 'new_joint_vecs'))[0], exist_ok=True)
            np.save(source_file.replace('joint', 'new_joints'),
                    rec_ric_data.squeeze().numpy())
            np.save(source_file.replace('joint', 'new_joint_vecs'), data)
            frame_num += data.shape[0]
        except Exception as e:
            print(source_file)
            print(e)

    print('Total clips: %d, Frames: %d, Duration: %fm' %
          (len(source_list), frame_num, frame_num / 20 / 60))
