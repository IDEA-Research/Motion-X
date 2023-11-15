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

import os
from torch.utils import data
from tqdm import tqdm
import numpy as np
import torch


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


def collate_tensors(batch):
    # Function for collating a batch of PyTorch tensors
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def mld_collate(batch):
    # Adapter function for collating batches in the MotionDatasetV2 class
    notnone_batches = [b for b in batch if b is not None]
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "name": [b[1] for b in notnone_batches],
        "length":
        collate_tensors([torch.tensor(b[2]).float() for b in notnone_batches]),
    }

    return adapted_batch


class MotionDatasetV2(data.Dataset):
    # Custom dataset class for motion data
    def __init__(self, root_path, debug):

        # Lists to store motion data and corresponding lengths
        self.data = []
        self.lengths = []

        # Finding all files in the specified directory
        self.id_list = findAllFile(root_path)

        # Limiting the number of files for debugging purposes
        if debug:
            self.id_list = self.id_list[:100]

        # Loading motion data from files and populating data and lengths lists
        for name in tqdm(self.id_list):
            motion = np.load(name)
            self.lengths.append(motion.shape[0])
            self.data.append({'motion': motion, 'name': name})

    def __len__(self):
        # Returns the number of items in the dataset
        return len(self.id_list)

    def __getitem__(self, item):
        # Returns motion data, file name, and length for a given item

        motion = self.data[item]['motion']
        name = self.data[item]['name']
        length = self.lengths[item]

        return motion, name, length
