import os 
from torch.utils import data
from tqdm import tqdm
import numpy as np
import torch

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def collate_tensors(batch):
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


# an adapter to our collate func
def mld_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    # notnone_batches.sort(key=lambda x: x[3], reverse=True)
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "name": [b[1] for b in notnone_batches],
        "length":
        collate_tensors([torch.tensor(b[2]).float() for b in notnone_batches]),
    }

    return adapted_batch


class MotionDatasetV2(data.Dataset):
    def __init__(self, root_path, debug):


        self.data = []
        self.lengths = []

        self.id_list = findAllFile(root_path)

        if debug:
            self.id_list = self.id_list[:100]

        for name in tqdm(self.id_list):
            motion = np.load(name)
            self.lengths.append(motion.shape[0])
            self.data.append({'motion': motion, 'name': name})
        # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):

        motion = self.data[item]['motion']
        name = self.data[item]['name']
        length = self.lengths[item]


        return motion, name, length