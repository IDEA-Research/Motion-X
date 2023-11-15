import numpy as np
import copy
import smplx
import os.path as osp
import pickle
import torch

# Change your own model path
HUMAN_MODEL_PATH = './body_models'

class SMPLX(object):
    def __init__(self):
        self.layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
        self.layer = {'neutral': smplx.create(HUMAN_MODEL_PATH, 'smplx', gender='NEUTRAL', use_pca=False, use_face_contour=True, **self.layer_arg),
                        'male': smplx.create(HUMAN_MODEL_PATH, 'smplx', gender='MALE', use_pca=False, use_face_contour=True, **self.layer_arg),
                        'female': smplx.create(HUMAN_MODEL_PATH, 'smplx', gender='FEMALE', use_pca=False, use_face_contour=True, **self.layer_arg)
                        }
        self.vertex_num = 10475
        self.face = self.layer['neutral'].faces
        self.shape_param_dim = 10
        self.expr_code_dim = 10
        with open(osp.join(HUMAN_MODEL_PATH, 'smplx', 'SMPLX_to_J14.pkl'), 'rb') as f:
            self.j14_regressor = pickle.load(f, encoding='latin1')
        with open(osp.join(HUMAN_MODEL_PATH, 'smplx', 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
            self.hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.face_vertex_idx = np.load(osp.join(HUMAN_MODEL_PATH, 'smplx', 'SMPL-X__FLAME_vertex_ids.npy'))
        self.J_regressor = self.layer['neutral'].J_regressor.numpy()
        self.J_regressor_idx = {'pelvis': 0, 'lwrist': 20, 'rwrist': 21, 'neck': 12}
        self.orig_hand_regressor = self.make_hand_regressor()
        #self.orig_hand_regressor = {'left': self.layer.J_regressor.numpy()[[20,37,38,39,25,26,27,28,29,30,34,35,36,31,32,33],:], 'right': self.layer.J_regressor.numpy()[[21,52,53,54,40,41,42,43,44,45,49,50,51,46,47,48],:]}

        # original SMPLX joint set
        self.orig_joint_num = 53 # 22 (body joints) + 30 (hand joints) + 1 (face jaw joint)
        self.orig_joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', # body joints
        'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', # left hand joints
        'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', # right hand joints
        'Jaw' # face jaw joint
        )
        self.orig_flip_pairs = \
        ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body joints
        (22,37), (23,38), (24,39), (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51) # hand joints
        )
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_part = \
        {'body': range(self.orig_joints_name.index('Pelvis'), self.orig_joints_name.index('R_Wrist')+1),
        'lhand': range(self.orig_joints_name.index('L_Index_1'), self.orig_joints_name.index('L_Thumb_3')+1),
        'rhand': range(self.orig_joints_name.index('R_Index_1'), self.orig_joints_name.index('R_Thumb_3')+1),
        'face': range(self.orig_joints_name.index('Jaw'), self.orig_joints_name.index('Jaw')+1)}

        # changed SMPLX joint set for the supervision
        self.joint_num = 137 # 25 (body joints) + 40 (hand joints) + 72 (face keypoints)
        self.joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose',# body joints
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
         *['Face_' + str(i) for i in range(1,73)] # face keypoints (too many keypoints... omit real names. have same name of keypoints defined in FLAME class)
         )
        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.lwrist_idx = self.joints_name.index('L_Wrist')
        self.rwrist_idx = self.joints_name.index('R_Wrist')
        self.neck_idx = self.joints_name.index('Neck')
        self.flip_pairs = \
        ( (1,2), (3,4), (5,6), (8,9), (10,11), (12,13), (14,17), (15,18), (16,19), (20,21), (22,23), # body joints
        (25,45), (26,46), (27,47), (28,48), (29,49), (30,50), (31,51), (32,52), (33,53), (34,54), (35,55), (36,56), (37,57), (38,58), (39,59), (40,60), (41,61), (42,62), (43,63), (44,64), # hand joints
        (67,68), # face eyeballs
        (69,78), (70,77), (71,76), (72,75), (73,74), # face eyebrow
        (83,87), (84,86), # face below nose
        (88,97), (89,96), (90,95), (91,94), (92,99), (93,98), # face eyes
        (100,106), (101,105), (102,104), (107,111), (108,110), # face mouth
        (112,116), (113,115), (117,119), # face lip
        (120,136), (121,135), (122,134), (123,133), (124,132), (125,131), (126,130), (127,129) # face contours
        )
        # self.joint_idx = \
        # (0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55, # body joints
        # 37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, # left hand joints
        # 52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75, # right hand joints
        # 22,15, # jaw, head #2
        # 57,56, # eyeballs #2
        # 76,77,78,79,80,81,82,83,84,85, # eyebrow #10
        # 86,87,88,89, # nose #4 
        # 90,91,92,93,94, # below nose # 5
        # 95,96,97,98,99,100,101,102,103,104,105,106, # eyes # 12
        # 107, # right mouth # 1
        # 108,109,110,111,112, # upper mouth # 5
        # 113, # left mouth # 1
        # 114,115,116,117,118, # lower mouth # 5
        # 119, # right lip # 1 
        # 120,121,122, # upper lip # 3
        # 123, # left lip # 1 
        # 124,125,126, # lower lip # 3
        # 127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143 # face contour # 17
        # )
        self.joint_idx = range(144)
        self.joint_part = \
        {'body': range(self.joints_name.index('Pelvis'), self.joints_name.index('Nose')+1),
        'lhand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.joints_name.index('R_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'hand': range(self.joints_name.index('L_Thumb_1'), self.joints_name.index('R_Pinky_4')+1),
        'face': range(self.joints_name.index('Face_1'), self.joints_name.index('Face_72')+1)}
        
        # changed SMPLX joint set for PositionNet prediction
        self.pos_joint_num = 65 # 25 (body joints) + 40 (hand joints)
        self.pos_joints_name = \
        ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear', 'L_Eye', 'R_Eye', 'Nose', # body joints
         'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb_4', 'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4', # left hand joints
         'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4', # right hand joints
         )
        self.pos_joint_part = \
        {'body': range(self.pos_joints_name.index('Pelvis'), self.pos_joints_name.index('Nose')+1),
        'lhand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('L_Pinky_4')+1),
        'rhand': range(self.pos_joints_name.index('R_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1),
        'hand': range(self.pos_joints_name.index('L_Thumb_1'), self.pos_joints_name.index('R_Pinky_4')+1)}
        self.pos_joint_part['L_MCP'] = [self.pos_joints_name.index('L_Index_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Middle_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Ring_1') - len(self.pos_joint_part['body']),
                                        self.pos_joints_name.index('L_Pinky_1') - len(self.pos_joint_part['body'])]
        self.pos_joint_part['R_MCP'] = [self.pos_joints_name.index('R_Index_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Middle_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Ring_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand']),
                                        self.pos_joints_name.index('R_Pinky_1') - len(self.pos_joint_part['body']) - len(self.pos_joint_part['lhand'])]
    
    def make_hand_regressor(self):
        regressor = self.layer['neutral'].J_regressor.numpy()
        lhand_regressor = np.concatenate((regressor[[20,37,38,39],:],
                                            np.eye(self.vertex_num)[5361,None],
                                                regressor[[25,26,27],:],
                                                np.eye(self.vertex_num)[4933,None],
                                                regressor[[28,29,30],:],
                                                np.eye(self.vertex_num)[5058,None],
                                                regressor[[34,35,36],:],
                                                np.eye(self.vertex_num)[5169,None],
                                                regressor[[31,32,33],:],
                                                np.eye(self.vertex_num)[5286,None]))
        rhand_regressor = np.concatenate((regressor[[21,52,53,54],:],
                                            np.eye(self.vertex_num)[8079,None],
                                                regressor[[40,41,42],:],
                                                np.eye(self.vertex_num)[7669,None],
                                                regressor[[43,44,45],:],
                                                np.eye(self.vertex_num)[7794,None],
                                                regressor[[49,50,51],:],
                                                np.eye(self.vertex_num)[7905,None],
                                                regressor[[46,47,48],:],
                                                np.eye(self.vertex_num)[8022,None]))
        hand_regressor = {'left': lhand_regressor, 'right': rhand_regressor}
        return hand_regressor

        
    def reduce_joint_set(self, joint):
        new_joint = []
        for name in self.pos_joints_name:
            idx = self.joints_name.index(name)
            new_joint.append(joint[:,idx,:])
        new_joint = torch.stack(new_joint,1)
        return new_joint


def get_smplx_layer(device):
    smplx_model = SMPLX()
    smplx_layer = copy.deepcopy(smplx_model.layer['neutral']).to(device)
    return smplx_layer, smplx_model

def process_smplx_322_data(smplx_data, smplx_layer, smplx_model, device, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    '''
    INPUT:
    smplx_data: (bs, frames, 322)
    
    RETURN:
    vertices: 
    joints: 
    pose: 
    faces:  

    '''
    # generate mesh
    # motion_representation = 'rot3d'
    # if pose is None:
    #     if motion_representation=='rot6d':
    #         pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
    #     else:
    #         pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    
    pose = torch.tensor(smplx_data, dtype=torch.float32)
    assert pose.shape[-1] == 322
    assert len(pose.shape) == 3

    
    use_flame = (pose.shape[-1] == 322)
    # cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)

    batch_size = pose.shape[0]
    num_frames = pose.shape[1]

    zero_pose = torch.zeros((batch_size, num_frames, 3)).float().to(device)  # eye poses
    zero_expr = torch.zeros((batch_size, num_frames, 10)).float().to(device)  # smplx face expression


    # concat poses
    pose = pose.reshape(batch_size*num_frames, 322)
    zero_pose = zero_pose.reshape(batch_size*num_frames, -1)
    zero_expr = zero_expr.reshape(batch_size*num_frames, -1)


    if face_carnical:
        neck_idx = 12
        pose[..., neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[..., :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[..., :3],  # controls the global root orientation
            'pose_body': pose[..., 3:3+63],  # controls the body
            'pose_hand': pose[..., 66:66+90],  # controls the finger articulation
            'pose_jaw': pose[..., 66+90:66+93],  # controls the yaw pose
            'face_expr': pose[..., 159:159+50],  # controls the face expression
            'face_shape': pose[..., 209:209+100],  # controls the face shape
            'trans': pose[..., 309:309+3],  # controls the global body position
            'betas': pose[..., 312:],  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[..., :3].to(device),  # controls the global root orientation 3
            'pose_body': pose[..., 3:3 + 63].to(device),  # controls the body 63
            'pose_hand': pose[..., 66:66 + 90].to(device),  # controls the finger articulation 90
            'pose_jaw': pose[..., 66 + 90:66 + 93].to(device),  # controls the yaw pose 3
            'trans': pose[..., 159:159 + 3].to(device),  # controls the global body position 3
            'betas': pose[..., 162:].to(device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][:, i:i+1, :], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])
    # import pdb; pdb.set_trace()

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][..., :45], right_hand_pose=body_parms['pose_hand'][..., 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zero_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
        # 
        
        # output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
        #                      left_hand_pose=body_parms['pose_hand'][..., :45], right_hand_pose=body_parms['pose_hand'][..., 45:],
        #                      jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr,
        #                      transl=body_parms['trans'])
        # import pdb; pdb.set_trace()

    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][..., :45], right_hand_pose=body_parms['pose_hand'][..., 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])

                    
    del zero_pose
    del zero_expr

    vertices = output.vertices.reshape(batch_size, num_frames, 10475, 3)
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smplx_model.joint_idx, :].reshape(batch_size, num_frames, len(smplx_model.joint_idx), 3)
    faces = smplx_model.face

    # import pdb; pdb.set_trace()


    

    # import trimesh
    # mesh = trimesh.load('/home/linjing/code/OSX/demo/output.obj')
    # mesh.vertices = vertices[0].cpu().numpy()
    # mesh.export('/home/linjing/exp/align_global_orient/wo_transform.obj')

    return vertices, joints, pose, faces 



if __name__ == '__main__':

    data = torch.tensor(np.load('/comp_robot/lushunlin/datasets/Motion-X/motion_data/smplx_322/kungfu/subset_0/000006.npy')).unsqueeze(0)
    import pdb; pdb.set_trace() 
    device = 'cuda:1'
    smplx_layer, smplx_model = get_smplx_layer(device)
    vert, joints, pose, faces = process_smplx_322_data(data, smplx_layer, smplx_model, device=device)
    import pdb; pdb.set_trace()
    import trimesh
    mesh = trimesh.Trimesh(vertices=vert.squeeze().detach().cpu().numpy()[50], faces=faces)
    mesh.export('test.obj')
    import pdb; pdb.set_trace()

    # JOINT_NAMES = [
    #     "pelvis",
    #     "left_hip",
    #     "right_hip",
    #     "spine1",
    #     "left_knee",
    #     "right_knee",
    #     "spine2",
    #     "left_ankle",
    #     "right_ankle",
    #     "spine3",
    #     "left_foot",
    #     "right_foot",
    #     "neck",
    #     "left_collar",
    #     "right_collar",
    #     "head",
    #     "left_shoulder",
    #     "right_shoulder",
    #     "left_elbow",
    #     "right_elbow",
    #     "left_wrist",
    #     "right_wrist",
    #     "jaw",
    #     "left_eye_smplhf",
    #     "right_eye_smplhf",
    #     "left_index1",
    #     "left_index2",
    #     "left_index3",
    #     "left_middle1",
    #     "left_middle2",
    #     "left_middle3",
    #     "left_pinky1",
    #     "left_pinky2",
    #     "left_pinky3",
    #     "left_ring1",
    #     "left_ring2",
    #     "left_ring3",
    #     "left_thumb1",
    #     "left_thumb2",
    #     "left_thumb3",
    #     "right_index1",
    #     "right_index2",
    #     "right_index3",
    #     "right_middle1",
    #     "right_middle2",
    #     "right_middle3",
    #     "right_pinky1",
    #     "right_pinky2",
    #     "right_pinky3",
    #     "right_ring1",
    #     "right_ring2",
    #     "right_ring3",
    #     "right_thumb1",
    #     "right_thumb2",
    #     "right_thumb3",
    #     "nose",
    #     "right_eye",
    #     "left_eye",
    #     "right_ear",
    #     "left_ear",
    #     "left_big_toe",
    #     "left_small_toe",
    #     "left_heel",
    #     "right_big_toe",
    #     "right_small_toe",
    #     "right_heel",
    #     "left_thumb",
    #     "left_index",
    #     "left_middle",
    #     "left_ring",
    #     "left_pinky",
    #     "right_thumb",
    #     "right_index",
    #     "right_middle",
    #     "right_ring",
    #     "right_pinky",
    #     "right_eye_brow1",
    #     "right_eye_brow2",
    #     "right_eye_brow3",
    #     "right_eye_brow4",
    #     "right_eye_brow5",
    #     "left_eye_brow5",
    #     "left_eye_brow4",
    #     "left_eye_brow3",
    #     "left_eye_brow2",
    #     "left_eye_brow1",
    #     "nose1",
    #     "nose2",
    #     "nose3",
    #     "nose4",
    #     "right_nose_2",
    #     "right_nose_1",
    #     "nose_middle",
    #     "left_nose_1",
    #     "left_nose_2",
    #     "right_eye1",
    #     "right_eye2",
    #     "right_eye3",
    #     "right_eye4",
    #     "right_eye5",
    #     "right_eye6",
    #     "left_eye4",
    #     "left_eye3",
    #     "left_eye2",
    #     "left_eye1",
    #     "left_eye6",
    #     "left_eye5",
    #     "right_mouth_1",
    #     "right_mouth_2",
    #     "right_mouth_3",
    #     "mouth_top",
    #     "left_mouth_3",
    #     "left_mouth_2",
    #     "left_mouth_1",
    #     "left_mouth_5",  # 59 in OpenPose output
    #     "left_mouth_4",  # 58 in OpenPose output
    #     "mouth_bottom",
    #     "right_mouth_4",
    #     "right_mouth_5",
    #     "right_lip_1",
    #     "right_lip_2",
    #     "lip_top",
    #     "left_lip_2",
    #     "left_lip_1",
    #     "left_lip_3",
    #     "lip_bottom",
    #     "right_lip_3",
    #     # Face contour
    #     "right_contour_1",
    #     "right_contour_2",
    #     "right_contour_3",
    #     "right_contour_4",
    #     "right_contour_5",
    #     "right_contour_6",
    #     "right_contour_7",
    #     "right_contour_8",
    #     "contour_middle",
    #     "left_contour_8",
    #     "left_contour_7",
    #     "left_contour_6",
    #     "left_contour_5",
    #     "left_contour_4",
    #     "left_contour_3",
    #     "left_contour_2",
    #     "left_contour_1",
    # ]

    # face_joint_id = [22,15, # jaw, head #2
    #     57,56, # eyeballs #2
    #     76,77,78,79,80,81,82,83,84,85, # eyebrow #10
    #     86,87,88,89, # nose #4 
    #     90,91,92,93,94, # below nose # 5
    #     95,96,97,98,99,100,101,102,103,104,105,106, # eyes # 12
    #     107, # right mouth # 1
    #     108,109,110,111,112, # upper mouth # 5
    #     113, # left mouth # 1
    #     114,115,116,117,118, # lower mouth # 5
    #     119, # right lip # 1 
    #     120,121,122, # upper lip # 3
    #     123, # left lip # 1 
    #     124,125,126, # lower lip # 3
    #     127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143]# face contour # 17
    # hand_joints_id = [37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, 52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75]
    # body_joints_id = [0,1,2,4,5,7,8,12,16,17,18,19,20,21,60,61,62,63,64,65,59,58,57,56,55]
    # face_name_joint = []
    # for i in face_joint_id:
    #     face_name_joint.append(JOINT_NAMES[i])

    # body_name_joint = []
    # for i in body_joints_id:
    #     body_name_joint.append(JOINT_NAMES[i])

    # hand_name_joint = []
    # for i in hand_joints_id:
    #     hand_name_joint.append(JOINT_NAMES[i])
    




    # import pdb; pdb.set_trace()


    
