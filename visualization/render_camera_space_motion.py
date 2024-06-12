import copy
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
from pytorch3d.renderer import (PerspectiveCameras, RasterizationSettings, MeshRasterizer, MeshRenderer,
HardPhongShader, TexturesVertex)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
import torch
import cv2
from smplx import SMPL, SMPLX, SMPLH
from tqdm import tqdm
import numpy
import trimesh
import pyrender
SMPLX_path = "common/human_model_files/smplx/SMPLX_NEUTRAL.npz"
import os
#os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "egl"
from utils.human_models import smpl_x
smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()

input_body_shape = (256, 192)
output_hm_shape = (16, 16, 12)
focal = (5000, 5000)
princpt = (input_body_shape[1]/2, input_body_shape[0]/2)
focal = list(focal)
princpt = list(princpt)


def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

def render_mesh(img, mesh, face, cam_param, img_white_bg=None, deg=0):
    # mesh
    cur_mesh = mesh.copy()
    mesh = trimesh.Trimesh(mesh, face)

    if (deg != 0):
        rot = trimesh.transformations.rotation_matrix(np.radians(deg), [0, 1, 0], point=np.mean(cur_mesh, axis=0))
        mesh.apply_transform(rot)

    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    # baseColorFactor = (1.0, 1.0, 0.9, 1.0)   # graw color
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  baseColorFactor=(1.0, 1.0, 0.9, 1.0))

    # Set other material properties for appearance
    #material.baseColorFactor = [0.25, 0.4, 0.65, 1.0]      # gray
    #material.baseColorFactor = [0.3, 0.3, 0.3, 1.0]      # silver
    material.baseColorFactor = [1.0, 1.0, 0.9, 1.0]     # white
    material.metallicFactor = 0.2
    material.roughnessFactor = 0.7

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    flags = (pyrender.RenderFlags.RGBA |
             pyrender.RenderFlags.SKIP_CULL_FACES)
    rgb, depth = renderer.render(scene, flags=flags)
    renderer.delete()
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    return img

def process_bbox(bbox, img_width, img_height, scale=1):
    bbox = sanitize_bbox(bbox, img_width, img_height)
    if bbox is None:
        return bbox

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = input_img_shape[1]/input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*scale
    bbox[3] = h*scale
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox

def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        bbox = None

    return bbox

def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on the given frame.

    Args:
    frame (numpy.ndarray): The input frame on which to draw the bounding box.
    bbox (tuple): The bounding box coordinates (x, y, w, h).
    color (tuple): The color of the bounding box (default is green).
    thickness (int): The thickness of the bounding box lines (default is 2).

    Returns:
    numpy.ndarray: The frame with the bounding box drawn.
    """
    x, y, w, h = bbox
    top_left = (int(x), int(y))
    bottom_right = (int(x + w), int(y + h))
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)
    return frame

if __name__ == "__main__":

    local_coco_file = "mesh_Lmotion/*.json"
    # The original video corresponding to the json file
    video_path = "video/*.mp4"
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = ""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    smplx = SMPLX(SMPLX_path, use_pca=False, flat_hand_mean=True).eval().to(device)
    faces = smplx.faces
    
    db = COCO(local_coco_file)
    
    for aid in tqdm(db.anns.keys()):
        ret, frame = cap.read()
        ann = db.anns[aid]

        root_pose = torch.FloatTensor(np.array(ann["smplx_params"]['root_pose'])).unsqueeze(0).to("cuda:0")
        body_pose = torch.FloatTensor(np.array(ann["smplx_params"]['body_pose'])).unsqueeze(0).to("cuda:0")
        lhand_pose = torch.FloatTensor(np.array(ann["smplx_params"]['lhand_pose'])).unsqueeze(0).to("cuda:0")
        rhand_pose = torch.FloatTensor(np.array(ann["smplx_params"]['rhand_pose'])).unsqueeze(0).to("cuda:0")
        jaw_pose = torch.FloatTensor(np.array(ann["smplx_params"]['jaw_pose'])).unsqueeze(0).to("cuda:0")
        shape = torch.FloatTensor(np.array(ann["smplx_params"]['shape'])).unsqueeze(0).to("cuda:0")
        expr = torch.FloatTensor(np.array(ann["smplx_params"]['expr'])).unsqueeze(0).to("cuda:0")
        cam_trans = torch.FloatTensor(np.array(ann["smplx_params"]['trans'])).unsqueeze(0).to("cuda:0")
        
        output = smplx_layer(betas=shape, 
                            body_pose=body_pose, 
                            global_orient=root_pose, 
                            right_hand_pose=rhand_pose,
                            left_hand_pose=lhand_pose, 
                            jaw_pose=torch.zeros([1, 3]).to(device),
                            leye_pose=torch.zeros([1, 3]).to(device),
                            reye_pose=torch.zeros([1, 3]).to(device), 
                            expression=expr)

        vertices = output.vertices

        mesh_cam = vertices + cam_trans[:, None, :] #(bs 10475 3)
        bbox = ann["bbox"]
        
        input_body_shape = (256, 192) 
        output_hm_shape = (16, 16, 12)
        focal = (5000, 5000)
        princpt = (input_body_shape[1]/2, input_body_shape[0]/2)
        focal = list(focal)
        princpt = list(princpt)

        focal[0] = focal[0] / input_body_shape[1] * bbox[2]
        focal[1] = focal[1] / input_body_shape[0] * bbox[3]
        princpt[0] = princpt[0] / input_body_shape[1] * bbox[2] + bbox[0]
        princpt[1] = princpt[1] / input_body_shape[0] * bbox[3] + bbox[1]
                
        img = draw_bbox(frame, bbox)
        mesh_cam = mesh_cam.cpu().numpy()[0]
        
        img= render_mesh(img, mesh_cam, faces, {'focal': focal, 'princpt': princpt})
        img = img.astype(numpy.uint8)
        out.write(img)
        
    cap.release()
    out.release()
        

