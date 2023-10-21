
import pandas as pd
import os
import pickle
import numpy as np
from utils.rotation_conversions import *
from smplx import SMPLX
from utils.face_z_align_util import joint_idx, face_z_transform
import re
from tqdm import tqdm

smplx_model_path = './body_models/smplx/SMPLX_NEUTRAL.npz'
smplx_model = SMPLX(smplx_model_path, num_betas=10, use_pca=False, use_face_contour=True, batch_size=1).cuda()

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

def compute_canonical_transform(global_orient):
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=global_orient.dtype)
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient

def transform_translation(trans):
    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
    trans[:, 2] = trans[:, 2] * (-1)
    return trans


def face_z_align(pose):
    pose = torch.from_numpy(pose).float().cuda()

    param = {
        'root_orient': pose[:, :3],  # controls the global root orientation
        'pose_body': pose[:, 3:3+63],  # controls the body
        'pose_hand': pose[:, 66:66+90],  # controls the finger articulation
        'pose_jaw': pose[:, 66+90:66+93],  # controls the yaw pose
        'face_expr': pose[:, 159:159+50],  # controls the face expression
        'face_shape': pose[:, 209:209+100],  # controls the face shape
        'trans': pose[:, 309:309+3],  # controls the global body position
        'betas': pose[:, 312:],  # controls the body shape. Body shape is static
    }

    batch_size = param['face_expr'].shape[0]
    zero_pose = torch.zeros((batch_size, 3)).float().cuda()

    smplx_output = smplx_model(betas=param['betas'], body_pose=param['pose_body'],
                               global_orient=param['root_orient'], pose2rot=True, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                               left_hand_pose=param['pose_hand'][:, :45], right_hand_pose=param['pose_hand'][:, 45:],
                               expression=param['face_expr'][:, :10], transl=param['trans'])
                        
    vertices = smplx_output.vertices
    joints = smplx_output.joints
    joints = joints[:, joint_idx, :]
    param['root_orient'], param['trans'] = face_z_transform(joints.cpu().numpy(), param['root_orient'], param['trans'])

    pose_list = []
    for k in ['root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'face_expr', 'face_shape', 'trans', 'betas']:
        pose_list.append(param[k])
    pose_list = torch.cat(pose_list, dim=-1).cpu().numpy()

    return pose_list

def get_egoody_data(folder_path, start_frame, end_frame):
    
    frames = findAllFile(folder_path)

    frames_path_folder = [os.path.dirname(i) for i in frames]
    frames_path_folder.sort()
    frames_folder = [os.path.basename(i) for i in frames_path_folder]
    frames_folder.sort()
    
    start_frame = 'frame_' + "{:05d}".format(start_frame)
    end_frame = 'frame_' + "{:05d}".format(end_frame)

    start_index = frames_folder.index(start_frame)
    end_index = frames_folder.index(end_frame)
    
    selected_frames = frames_path_folder[start_index:end_index + 1]
    # import pdb; pdb.set_trace()
    pose_seq = []

    for frame_path in selected_frames:
        assert len(findAllFile(frame_path)) == 1
        with open(findAllFile(frame_path)[0], 'rb') as f:
            frame_data = pickle.load(f)

        pose_root = frame_data['global_orient']
        pose_root = compute_canonical_transform(torch.from_numpy(pose_root)).detach().cpu().numpy()
        pose_body = frame_data['body_pose']
        #original egobody hand is 12-dim, not compatible with ours, so use zeros to initilize
        pose_hand = np.zeros((1, 90))
        pose_jaw = frame_data['jaw_pose']
        #original egobody expression is 10-dim, not compatible with ours, so use zeros to initilize
        pose_expression = np.zeros((1, 50))
        pose_face_shape = np.zeros((1, 100))
        pose_trans = frame_data['transl']
        pose_trans = transform_translation(pose_trans)
        pose_body_shape = frame_data['betas']
        pose = np.concatenate((pose_root, pose_body, pose_hand, pose_jaw, pose_expression, pose_face_shape, pose_trans, pose_body_shape), axis=1)
        pose_seq.append(pose)

    pose_seq = np.concatenate(pose_seq, axis=0)
    
    # import pdb; pdb.set_trace()
    return pose_seq, frame_path.split('results')[0]

    
def create_parent_dir(target_t2m_joints_case_path):
    target_t2m_joints_parent_directory = os.path.dirname(target_t2m_joints_case_path)

    if not os.path.exists(target_t2m_joints_parent_directory):
        os.makedirs(target_t2m_joints_parent_directory)


def process_text(text):
    result = re.sub(r'_clip\d+', '', text)
    result = result.replace('_', ' ')
    result = re.sub(r'\s+', ' ', result)
    return result



if __name__ == '__main__':

    index_path = './egobody_description_all.csv'
    index_file = pd.read_csv(index_path)

    wear_file_list_test_path = './EgoBody/smplx_camera_wearer_test'
    wear_file_list_test = [os.path.join(wear_file_list_test_path, i) for i in os.listdir(wear_file_list_test_path)]

    wear_file_list_train_path = './EgoBody/smplx_camera_wearer_train'
    wear_file_list_train = [os.path.join(wear_file_list_train_path, i) for i in os.listdir(wear_file_list_train_path)]

    wear_file_list_val_path = './EgoBody/smplx_camera_wearer_val'
    wear_file_list_val = [os.path.join(wear_file_list_val_path, i) for i in os.listdir(wear_file_list_val_path)]

    interactee_file_train_path = './EgoBody/smplx_interactee_train'
    interactee_file_list_train = [os.path.join(interactee_file_train_path, i) for i in os.listdir(interactee_file_train_path)]

    interactee_file_val_path = './EgoBody/smplx_interactee_val'
    interactee_file_list_val = [os.path.join(interactee_file_val_path, i) for i in os.listdir(interactee_file_val_path)]

    interactee_file_test_path = './EgoBody/smplx_interactee_test'
    interactee_file_list_test = [os.path.join(interactee_file_test_path, i) for i in os.listdir(interactee_file_test_path)]

    wear_dict = {}
    interactee_dcit = {}

    wear_list = wear_file_list_test + wear_file_list_train + wear_file_list_val
    interactee_list = interactee_file_list_train + interactee_file_list_val + interactee_file_list_test

    for i in wear_list:
        wear_dict[os.path.basename(i)] = i
    
    for i in interactee_list:
        interactee_dcit[os.path.basename(i)] = i


    for index, row in tqdm(index_file.iterrows()):
        
        recording_name = row['recording_name']
        frame_start = row['frame_interval_start']
        frame_end = row['frame_interval_end']
        body_idx = row['body_idx_0']

        if '1' in body_idx:
            path = wear_dict[recording_name]

        else:
            path = interactee_dcit[recording_name]

        text_description = row['body_0_des']
        text_description = process_text(text_description)
        pose_data, output_folder_path = get_egoody_data(path, frame_start, frame_end)
        pose_data = face_z_align(pose_data)

        output_folder_path = re.sub(r'/EgoBody/([^/]+)/', '/EgoBody_motion/', output_folder_path)


        clip = 0
        while os.path.exists(os.path.join(output_folder_path, "{:03d}".format(clip) + '.npy')):
            clip += 1

        save_path = os.path.join(output_folder_path, "{:03d}".format(clip) + '.npy')
        create_parent_dir(save_path)
        np.save(save_path, pose_data)
        save_txt_path = save_path.replace('EgoBody_motion', 'EgoBody_txt').replace('.npy', '.txt')
        create_parent_dir(save_txt_path)

        with open(save_txt_path, 'w') as f:
            f.write(text_description)