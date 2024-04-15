## Load data for quick start

### Load a motion sequence for generation

```python
motion = np.load('/your/data/root/smplx_322/Act_cute_and_sitting_at_the_same_time_clip1.npy')
motion = torch.tensor(motion).float()
motion_params = {
            'root_orient': motion[:, :3],  # controls the global root orientation
            'pose_body': motion[:, 3:3+63],  # controls the body
            'pose_hand': motion[:, 66:66+90],  # controls the finger articulation
            'pose_jaw': motion[:, 66+90:66+93],  # controls the yaw pose
            'face_expr': motion[:, 159:159+50],  # controls the face expression
            'face_shape': motion[:, 209:209+100],  # controls the face shape
            'trans': motion[:, 309:309+3],  # controls the global body position
            'betas': motion[:, 312:],  # controls the body shape. Body shape is static
        }

# read text labels
semantic_text = np.loadtxt('semantic_labels/000001.npy')     # semantic labels 
```



### Load a local motion sequence for mesh recovery

```python
annot_path = "/your/data/root/mesh_recovery/local_motion/Act_cute_and_sitting_at_the_same_time_clip1.json"
db = COCO(annot_path)
for aid in db.anns.keys():
    ann = db.anns[aid]
    img = db.loadImgs(ann['image_id'])[0]
    motion_params = {
                'root_orient': ann["smplx_params"]["root_pose"],  # controls the global root orientation
                'pose_body': ann["smplx_params"]["body_pose"],  # controls the body
                'pose_lhand': ann["smplx_params"]["lhand_pose"],  # controls the left hand finger articulation
                'pose_rhand': ann["smplx_params"]["rhand_pose"],  # controls the right hand finger articulation
                'pose_jaw': ann["smplx_params"]["jaw_pose"],  # controls the yaw pose
                'face_expr': ann["smplx_params"]["expr"],  # controls the face expression
                'trans': ann["smplx_params"]["trans"],  # controls the global body position
                'betas': ann["smplx_params"]["shape"],  # controls the body shape. Body shape is static
            }
	camera_params = ann["camera_params"] # load camera parameters for mesh recovery
```



### Load a global motion sequence for mesh recovery

```python
smooth_fit_path = "/your/data/root/mesh_recovery/local_motion/Act_cute_and_sitting_at_the_same_time_clip1.json"
db = COCO(smooth_fit_path)
for aid in db.anns.keys():
    ann = db.anns[aid]
    img = db.loadImgs(ann['id'])[0]
    motion_params = {
                'root_orient': ann["smplx_params"]["root_orient"],  # controls the global root orientation
                'pose_body': ann["smplx_params"]["pose_body"],  # controls the body
                'pose_lhand': ann["smplx_params"]["lhand_pose"],  # controls the left hand finger articulation
                'pose_rhand': ann["smplx_params"]["rhand_pose"],  # controls the right hand finger articulation
                'pose_jaw': ann["smplx_params"]["pose_jaw"],  # controls the yaw pose
                'face_expr': ann["smplx_params"]["face_expr"],  # controls the face expression
                'face_shape': ann["smplx_params"]["face_shape"], # controls the face shape
                'trans': ann["smplx_params"]["trans"],  # controls the global body position
                'betas': ann["smplx_params"]["betas"],  # controls the body shape. Body shape is static
            }
    cam_params = ann["cam_params"] # load camera parameters for mesh recovery
```



### Load a wholebody keypoints sequence

```python
kpt_annot_path = "/your/data/root/keypoints/Act_cute_and_sitting_at_the_same_time_clip1.json"
db = COCO(kpt_annot_path)
for aid in db.anns.keys():
    ann = db.anns[aid]
    img = db.loadImgs(ann['id'])[0] 
    body_kpts = ann["body_kpts"] # load body keypoints
    foot_kpts = ann["foot_kpts"] # load foot keypoints
    lefthand_kpts = ann["lefthand_kpts"] # load left hand keypoints
    righthand_kpts = ann["righthand_kpts"] # load right hand keypoints
    face_kpts = ann["face_kpts"] # load face keypoints
```

