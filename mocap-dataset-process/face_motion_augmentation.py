import argparse
import json
import os
import numpy as np
from tqdm import tqdm

import torch
from slerp import slerp_interpolate
from multiprocessing import Pool


def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


motion_folder = "../datasets/motion_data/smplx_322"


def face_motion_augmentation(mocap_motion_file):
    face_motion_file = mocap_motion_file.replace("/motion_data/", "/face_motion_data/")
    motion = np.load(mocap_motion_file)
    if os.path.exists(face_motion_file):
        face_motion = np.load(face_motion_file)
    else:
        return face_motion_file  # not found face motion file

    motion_length, face_motion_length = motion.shape[0], face_motion.shape[0]
    if motion_length != face_motion_length:
        face_motion = torch.from_numpy(face_motion)
        n_frames, n_dims = face_motion.shape
        n_joints = n_dims // 3
        face_motion = face_motion.reshape(n_frames, n_joints, 3)
        face_motion = slerp_interpolate(face_motion, motion_length)
        face_motion = face_motion.reshape(motion_length, -1).numpy()
    (
        motion[:, 66 + 90 : 66 + 93],
        motion[:, 159 : 159 + 50],
        motion[:, 209 : 209 + 100],
    ) = (face_motion[:, :3], face_motion[:, 3 : 3 + 50], face_motion[:, 53:153])
    np.save(mocap_motion_file, motion)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_proc", type=int, default=len(os.sched_getaffinity(0)) // 2
    )
    args = parser.parse_args()

    not_found_face_motion_files = []
    for mocap_dataset in ["humanml", "EgoBody", "GRAB"]:
        mocap_motion_folder = os.path.join(motion_folder, mocap_dataset)

        mocap_motion_files = findAllFile(mocap_motion_folder)
        with Pool(args.num_proc) as p:
            with tqdm(total=len(mocap_motion_files)) as pbar:
                for not_found_face_motion_file in p.imap_unordered(
                    face_motion_augmentation, mocap_motion_files
                ):
                    not_found_face_motion_files.append(not_found_face_motion_file)
                    pbar.update(1)

    not_found_face_motion_files = list(
        filter(lambda x: x is not None, not_found_face_motion_files)
    )

    print(
        "The count of not_found_face_motion_files: ", len(not_found_face_motion_files)
    )
    json.dump(
        not_found_face_motion_files,
        open(os.path.join("./", "not_found_face_motion_files.json"), "w"),
    )
