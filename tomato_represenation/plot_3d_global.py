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

import torch
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio
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


def plot_3d_motion(joints, out_name, title, kinematic_chain, figsize=(10, 10), fps=120, radius=4):
    """
    Plot 3D motion data.

    Parameters:
    - joints (numpy array): 3D motion data of joints, shape (frame_number, joint_number, 3).
    - out_name (str or None): If specified, save the plot to this file path. If None, return the plot as a tensor.
    - title (str or None): Title of the plot.
    - kinematic_chain (list): List defining the kinematic chain of joints.
    - figsize (tuple): Size of the figure in inches.
    - fps (int): Frames per second for the animation.
    - radius (int): Radius of the joint markers.

    Returns:
    - If out_name is None, returns the plot as a tensor.
    - If out_name is specified, saves the plot and returns None.
    """
    matplotlib.use('Agg')

    data = joints.copy().reshape(len(joints), -1, 3)

    nb_joints = joints.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [
        3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            # Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)
        fig = plt.figure(figsize=(
            480/96., 320/96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(10, 10), dpi=96)
        if title is not None:
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        ax = p3.Axes3D(fig)

        init()

        ax.lines = []
        ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()

        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number):
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)


def plot_3d_motion_smplh(joints, out_name, title, kinematic_chain, figsize=(10, 10), fps=120, radius=4):
    """
    Plot 3D motion data of SMPL-H model.

    Parameters:
    - joints (numpy array): 3D motion data of joints, shape (frame_number, joint_number, 3).
    - out_name (str or None): If specified, save the plot to this file path. If None, return the plot as a tensor.
    - title (str or None): Title of the plot.
    - kinematic_chain (list): List defining the kinematic chain of joints.
    - figsize (tuple): Size of the figure in inches.
    - fps (int): Frames per second for the animation.
    - radius (int): Radius of the joint markers.

    Returns:
    - If out_name is None, returns the plot as a tensor.
    - If out_name is specified, saves the plot and returns None.
    """
    matplotlib.use('Agg')
    data = joints.copy().reshape(len(joints), -1, 3)

    nb_joints = joints.shape[1]
    smpl_kinetic_chain = kinematic_chain

    limits = 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            # Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=(10, 10), dpi=96)
        if title is not None:
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        init()

        ax.lines = []
        ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5

        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()

        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number):
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)


def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None):

    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size):
        out.append(plot_3d_motion(
            [smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None]))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=20)
    out = torch.stack(out, axis=0)
    return out


def draw_to_batch_smplh(smpl_joints_batch, kinematic_chain, title_batch=None, outname=None):

    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size):
        out.append(plot_3d_motion_smplh(
            smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None, kinematic_chain))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=20)
    out = torch.stack(out, axis=0)
    return out


def draw_to_batch_smplh_folder(kinematic_chain, input_folder):
    for input_case in tqdm(findAllFile(input_folder)):
        try:
            joints = np.load(input_case)
            assert joints.shape[1] == 52
            xyz = joints.reshape(1, -1, 52, 3)

            output_path = input_case.replace('.npy', '.gif')
            # extract father path
            parent_directory = os.path.dirname(output_path)

            # check whether father exists
            if not os.path.exists(parent_directory):
                # if not, create it!
                os.makedirs(parent_directory)

            pos_viz = draw_to_batch_smplh(
                xyz, kinematic_chain, title_batch=None, outname=[output_path])
        except:
            pass


if __name__ == '__main__':
    # Visualize your final data, please define your example_path, like 'new_data_humanml_000067_joints_using_smplx_rotation.npy'
    example_path = None
    assert example_path != None
    joints = np.load(example)

    # 2*3*5=30, left first, then right
    hand_joints_id = [i for i in range(25, 55)]
    body_joints_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 22 joints

    t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [
        0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [
        20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
    t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [
        21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]
    t2m_body_hand_kinematic_chain = t2m_kinematic_chain + \
        t2m_left_hand_chain + t2m_right_hand_chain

    if joints.shape[1] != 52:
        joints = joints[:, body_joints_id+hand_joints_id, :]
    xyz = joints.reshape(1, -1, 52, 3)
    pose_vis = draw_to_batch_smplh(xyz, t2m_body_hand_kinematic_chain, title_batch=None, outname=[
                                   f'output.gif'])
