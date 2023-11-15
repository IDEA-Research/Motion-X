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

import numpy as np

# Define a kinematic tree for the skeletal struture
kit_kinematic_chain = [[0, 11, 12, 13, 14, 15], [
    0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]

kit_raw_offsets = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ]
)

# offsets is the global offset between children and parent joint.
# 22
t2m_raw_body_offsets = np.array([[0, 0, 0],  # pelvis
                                 [1, 0, 0],  # left_hip
                                 [-1, 0, 0],  # right_hip
                                 [0, 1, 0],  # spine1
                                 [0, -1, 0],  # left_knee
                                 [0, -1, 0],  # right_knee
                                 [0, 1, 0],  # spine2
                                 [0, -1, 0],  # left_ankle
                                 [0, -1, 0],  # right_ankle
                                 [0, 1, 0],  # spine3
                                 [0, 0, 1],  # left_foot
                                 [0, 0, 1],  # right_foot
                                 [0, 1, 0],  # neck
                                 [1, 0, 0],  # left_collar
                                 [-1, 0, 0],  # right_collar
                                 [0, 0, 1],  # head
                                 [0, -1, 0],  # left_shoulder
                                 [0, -1, 0],  # right_shoulder
                                 [0, -1, 0],  # left_elbow
                                 [0, -1, 0],  # right_elbow
                                 [0, -1, 0],  # left_wrist
                                 [0, -1, 0]])  # right_wrist

# 30
t2m_hand_raw_offsets = np.array([[1, 0, 0],  # left_index1
                                [1, 0, 0],  # left_index2
                                [1, 0, 0],  # left_index3
                                [1, 0, 0],  # left_middle1
                                [1, 0, 0],  # left_middle2
                                [1, 0, 0],  # left_middle3
                                [1, 0, 0],  # left_pinky1
                                [1, 0, 0],  # left_pinky2
                                [1, 0, 0],  # left_pinky3
                                [1, 0, 0],  # left_ring1
                                [1, 0, 0],  # left_ring2
                                [1, 0, 0],  # left_ring3
                                [1, 0, 0],  # left_thumb1
                                [1, 0, 0],  # left_thumb2
                                [1, 0, 0],  # left_thumb3
                                [-1, 0, 0],  # right_index1
                                [-1, 0, 0],  # right_index2
                                [-1, 0, 0],  # right_index3
                                [-1, 0, 0],  # right_middle1
                                [-1, 0, 0],  # right_middle2
                                [-1, 0, 0],  # right_middle3
                                [-1, 0, 0],  # right_pinky1
                                [-1, 0, 0],  # right_pinky2
                                [-1, 0, 0],  # right_pinky3
                                [-1, 0, 0],  # right_ring1
                                [-1, 0, 0],  # right_ring2
                                [-1, 0, 0],  # right_ring3
                                [-1, 0, 0],  # right_thumb1
                                [-1, 0, 0],  # right_thumb2
                                [-1, 0, 0],])  # right_thumb3


t2m_raw_offsets = np.concatenate(
    (t2m_raw_body_offsets, t2m_hand_raw_offsets), axis=0)

# pelvis --> right_hip ---> right_knee ---> right_ankle ---> right_foot 0, 2, 5, 8, 11
# pelvis ---> left_hip ---> left_knee ---> left_ankle ---> left_foot 0, 1, 4, 7, 10
# pelvis ---> spine1 ---> spine2 ---> spine3 ---> neck ---> head 0, 3, 6, 9, 12, 15
# spine3 ---> right_collar ---> right_shoulder --->  right_elbow ---> right_wrist  9, 14, 17, 19, 21
# spine3 ---> left_collar ---> left_shoulder --->  left_elbow ---> left_wrist 9, 13, 16, 18, 20

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [
    0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

# left_wrist ---> left_index1 ---> left_index2 --->left_index3
# left_wrist ---> left_thumb1 ---> left_thumb2 --->left_thumb3
# left_wrist ---> left_middle1 ---> left_middle2 --->left_middle3
# left_wrist ---> left_ring1 ---> left_ring2 --->left_ring3
# left_wrist ---> left_pinky1 ---> left_pinky2 --->left_pinky3


# t2m_left_hand_chain = [[20, 25, 26, 27], [20, 37, 38, 39], [20, 28, 29, 30], [20, 34, 35, 36], [20, 31, 32, 33]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [
    20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]

# right_wrist ---> right_pinky1 ---> right_pinky2 --->right_pinky3
# right_wrist ---> right_ring1 ---> right_ring2 --->right_ring3
# right_wrist ---> right_middle1 ---> right_middle2 --->right_middle3
# right_wrist ---> right_index1 ---> right_index2 --->right_index3
# right_wrist ---> right_thumb1 ---> right_thumb2 --->right_thumb3

# t2m_right_hand_chain = [[21, 46, 47, 48], [21, 34, 35, 36], [21, 43, 44, 45], [21, 40, 41, 42], [21, 52, 53, 54]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [
    21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]

t2m_body_hand_kinematic_chain = t2m_kinematic_chain + \
    t2m_left_hand_chain + t2m_right_hand_chain

kit_tgt_skel_id = '03950'

t2m_tgt_skel_id = '000021'


if __name__ == '__main__':
    import pdb
    pdb.set_trace()
