import json
import numpy as np
import os
import os.path as osp
from pycocotools.coco import COCO
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader
)
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer import TexturesVertex
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.renderer import (PerspectiveCameras, RasterizationSettings, MeshRasterizer, MeshRenderer, HardPhongShader, TexturesVertex)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
import torch
import cv2
from smplx import SMPL, SMPLX, SMPLH
from tqdm import tqdm
import numpy

SMPLX_path = "/comp_robot/zhangyuhong1/code/Hand4Whole/common/human_model_files/smplx/SMPLX_NEUTRAL.npz"

def render_mesh_to_image(smpl_vertices, smpl_faces, cameras, image, camera_params):

    raster_settings = RasterizationSettings(image_size=[image.shape[0], image.shape[1]],
                                            blur_radius=0,
                                            faces_per_pixel=1,
                                            bin_size=0,
                                            max_faces_per_bin=1000,
                                            perspective_correct=True)
    smpl_vertices = smpl_vertices.detach().cpu().numpy()

    verts = torch.tensor([smpl_vertices], dtype=torch.float32, device=device)
    faces = torch.from_numpy(smpl_faces.astype(np.int32)).to(device)
    faces = faces[None, :, :]
    verts_rgb = torch.ones_like(verts)
    verts_rgb[:, :, 2] = verts_rgb[:, :, 2] * 0
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras,
                                                      raster_settings=raster_settings),
                            shader=HardPhongShader(device=device, cameras=cameras))
    images = renderer(
        meshes_world=Meshes(verts=verts.to(device), faces=faces.to(device), textures=textures))

    images = images.cpu().numpy().squeeze() * 255
    images_rgb = images[:, :, :3].astype(numpy.uint8)
    mask = images_rgb.max(axis=2) < 255  
    images_rgb_resized = cv2.resize(images_rgb, (image.shape[1], image.shape[0]))
    mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0])) > 0
    image[mask_resized] = images_rgb_resized[mask_resized]
    
    # render with color vertex 
    # camera_matrix = np.array(camera_params['matrix'])
    # camera_matrix = torch.tensor([camera_matrix], dtype=torch.float32, device=device)
    # rvec = camera_params['R']
    # tvec = camera_params['T']
    # cameraMatrix = np.array(camera_params['matrix'])
    # distCoeffs = numpy.zeros([4, 1])
    # points = verts.cpu().numpy()
    # points2d = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)
    # points2d = points2d[0]
    # points2d = points2d.astype(int)
    # for i in range(points2d.shape[0]):
    #     cv2.circle(image, tuple(points2d[i][0]), 1, (0, 255, 255))
    return image



if __name__ == "__main__":
    
    smooth_fit_coco_file = "mesh_Gmotion/*.json"
    # The original video corresponding to the json file
    video_path = "video/*.mp4"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    db = COCO(smooth_fit_coco_file)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    smplx = SMPLX(SMPLX_path, use_pca=False, flat_hand_mean=True).eval().to(device)
    faces = smplx.faces
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "test.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for aid in tqdm(db.anns.keys()):
        ret, frame = cap.read()
        ann = db.anns[aid]
        
        R = np.array(ann["cam_params"]["cam_R"])
        T = np.array(ann["cam_params"]["cam_T"])
        R_vec = cv2.Rodrigues(R)[0]
        T_vec = np.array(ann["cam_params"]["cam_T"])
        
        intrins = ann["cam_params"]["intrins"] # [1500.0, 1500.0, 960.0, 540.0]
        camera_matrix = np.array([[intrins[0], 0, intrins[2]],[0, intrins[1], intrins[3]],[0,0,1]])
        
        camera_params = {
            "matrix":camera_matrix,
            "R":R_vec,
            "T":T_vec
        }
        R = torch.from_numpy(R).to(device=device, dtype=torch.float32)[None, :, :]
        T = torch.from_numpy(T).to(device=device, dtype=torch.float32)[None, :]
        camera_matrix = torch.from_numpy(camera_matrix).to(device=device, dtype=torch.float32)[None, :, :]
        image_size = torch.tensor([frame.shape[0], frame.shape[1]], device=device).unsqueeze(0)
        cameras = cameras_from_opencv_projection(R, T, camera_matrix, image_size)
 
        # smplx pose to vertex
        pose_body = np.array(ann['smplx_params']['pose_body'])
        global_orient = np.array(ann['smplx_params']['root_orient'])
        transl = np.array(ann['smplx_params']['trans'])
        left_hand_pose = np.array(ann['smplx_params']['pose_hand'][:45])
        right_hand_pose = np.array(ann['smplx_params']['pose_hand'][45:])
        expr = np.array(ann['smplx_params']['face_expr'][:10])
        
        pose_body = torch.from_numpy(pose_body).to(device=device, dtype=torch.float32).unsqueeze(0)
        global_orient = torch.from_numpy(global_orient).to(device=device, dtype=torch.float32).unsqueeze(0)
        transl = torch.from_numpy(transl).to(device=device, dtype=torch.float32).unsqueeze(0)
        left_hand_pose = torch.from_numpy(left_hand_pose).to(device=device, dtype=torch.float32).unsqueeze(0)
        right_hand_pose = torch.from_numpy(right_hand_pose).to(device=device, dtype=torch.float32).unsqueeze(0)
        expr = torch.from_numpy(expr).to(device=device, dtype=torch.float32).unsqueeze(0)
        
        output = smplx.forward(
            betas = torch.zeros([1, 10]).to(device),
            transl = transl,
            global_orient = global_orient,
            body_pose = pose_body,
            jaw_pose = torch.zeros([1, 3]).to(device),
            leye_pose = torch.zeros([1, 3]).to(device),
            reye_pose = torch.zeros([1, 3]).to(device),
            left_hand_pose = left_hand_pose,
            right_hand_pose = right_hand_pose,
            expression= expr,
        )

        vertices = output.vertices

        processed_frame = render_mesh_to_image(vertices[0], faces, cameras, frame, camera_params)
        out.write(processed_frame)
        
    cap.release()
    out.release()
