import os
import os.path as osp
import json
import random
import logging
import hydra
import colorlog
from omegaconf import DictConfig, OmegaConf
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import time
import trimesh
from PIL import Image
import torch
from tqdm import trange, tqdm
import numpy as np
from mesh_to_sdf import mesh_to_voxels
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, quaternion_to_matrix
from src.models.hoi_optim_module import HOI_Sync
from src.utils.logging import log_init
from src.utils.vis_utils import visualize_mesh
from src.utils.cam_utils import (
    load_cam,
    get_projection,
    correct_image_orientation,
    resize_frame,
    center_looking_at_camera_pose,
    calc_orig_cam_params,
    adjust_principal_point
)
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def image_process(input_image, hand_mask, obj_mask, inpaint_mask):
    input_image = correct_image_orientation(input_image)
    hand_mask = correct_image_orientation(hand_mask).convert('L') # grayscale
    obj_mask = correct_image_orientation(obj_mask).convert('L')
    inpaint_mask = correct_image_orientation(inpaint_mask).convert('L')
    
    input_image = (input_image)
    hand_mask = (hand_mask).convert('L') # grayscale
    obj_mask = (obj_mask).convert('L')
    inpaint_mask = (inpaint_mask).convert('L')
    
    """
    input_image, new_size = resize_frame(input_image)
    
    hand_mask = hand_mask.resize(new_size)
    obj_mask = obj_mask.resize(new_size)
    inpaint_mask = inpaint_mask.resize(new_size)
    """
    
    hand_mask, obj_mask, inpaint_mask = np.array(hand_mask), np.array(obj_mask), np.array(inpaint_mask)
    
    return input_image, hand_mask, obj_mask, inpaint_mask

# NOTE:加载相机参数
def get_obj_cam(device, dir):
    cam_intr_path = osp.join(dir, "camera", "gpt.json")
    with open(cam_intr_path, 'r') as f:
        cam_intr = json.load(f)
        
    obj_pose_path = osp.join(dir, "megapose", "gpt", "outputs", "object_data.json")
    with open(obj_pose_path, 'r') as f:
        obj_pose_data = json.load(f)
    
    # megapose6d输入是xyzw
    quat_xyzw = torch.tensor(obj_pose_data[0]['TWO'][0], device=device).float()
    transl = torch.tensor(obj_pose_data[0]['TWO'][1], device=device).float()
    
    # 将 xyzw 转换为 wxyz
    quat_wxyz = torch.cat([quat_xyzw[-1:], quat_xyzw[:-1]])
    rot_mat = quaternion_to_matrix(quat_wxyz.unsqueeze(0)).squeeze(0)

    object_pose = torch.eye(4, device=device)
    object_pose[:3, :3] = rot_mat
    object_pose[:3, 3] = transl
    T = torch.linalg.inv(object_pose)
    
    # openGL坐标系, x右y上z后
    # megapose坐标系, x右y下z前
    extrinsics = torch.tensor([
        [1.0,  0.0,  0.0, 0.0],
        [0.0,  -1.0,  0.0, 0.0],
        [0.0,  0.0,  -1.0, 0.0],
        [0.0,  0.0,  0.0, 1.0]
    ], device=device)

    extrinsics = T @ extrinsics

    K = np.array(cam_intr['K'])
    h, w = np.array(cam_intr['resolution'][0]), np.array(cam_intr['resolution'][1])
    obj_cam = {
        'fx': K[0, 0] / w,
        'fy': K[1, 1] / h,
        'cx': K[0, 2] / w,
        'cy': K[1, 2] / h
    }
    
    obj_cam["extrinsics"] = extrinsics
    obj_cam["projection"] = torch.FloatTensor(get_projection(obj_cam, width=w, height=h)).to(device)
    
    return obj_cam, object_pose

def try_until_success(func, max_attempts=5, exception_to_check=Exception, verbose=True, **kwargs):
    """
    Tries to execute a function until it succeeds or reaches the maximum attempts.

    Args:
        func (callable): The function to execute.
        max_attempts (int, optional): Maximum number of attempts. Defaults to 5.
        exception_to_check (Exception, optional): The type of exception to catch. 
                                                  Defaults to Exception (catches all exceptions).
        verbose (bool, optional): Whether to print messages about attempts. Defaults to True.

    Returns:
        The result of the function if successful, otherwise None.
    """

    for attempt in range(1, max_attempts + 1):
        if verbose:
            logging.info(f"Attempt {attempt}/{max_attempts}...")

        try:
            random.seed(attempt)
            np.random.seed(attempt)
            result = func(**kwargs)  # Try executing the function
            return result     # Return the result if successful
        except exception_to_check as e:
            if verbose:
                  logging.error(f"Attempt {attempt} failed, Exception Type: {repr(e)}")
            if attempt < max_attempts:
                if verbose:
                    logging.info("Retrying ...")

    if verbose:
        logging.error(f"Function failed after {max_attempts} attempts.")
    return None  # Return None if all attempts fail

# 数据加载
def load_data_single(dir):
    # resize the input image, because the resolution for nvdiffrastmust must be [<=2048, <=2048]
    img_path = osp.join(dir, "gpt.png")
    input_image = Image.open(img_path)
    
    hand_mask = Image.open(osp.join(dir, "LISA", "hand", "mask.png"))
    obj_mask = Image.open(osp.join(dir, "LISA", "obj", "mask.png"))
    inpaint_mask = Image.open(osp.join(dir, "megapose", "gpt", "visualizations", "object_mask.png"))
    origin_w, origin_h = input_image.width, input_image.height

    input_image, hand_mask, obj_mask, inpaint_mask = image_process(input_image, hand_mask, obj_mask, inpaint_mask)

    hand_cam_file = osp.join(dir, "hamer", "gpt_cam.json")
    obj_mesh_path = osp.join(dir, "pc", "fixed.obj")
    
    obj_mesh = trimesh.load(obj_mesh_path)
    try:
        object_colors = obj_mesh.visual.vertex_colors
    except Exception as e:
        object_colors = None
        
    """ load hand info """
    logging.info(f"Loading hand MANO parameters")
    info = torch.load(osp.join(dir, "hamer", "gpt.pt"))
    
    hand_info = {'mano_params': {}}
    # 默认只有一只手,因此index=0
    for key in info['mano_params']:
        hand_info['mano_params'][key] = info['mano_params'][key][0]
            
    hand_info.update({key: info[key][0] 
                        for key in info 
                        if key not in ['batch_size','mano_params']})
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mano_params = hand_info["mano_params"]
    for key in mano_params:
        mano_params[key] = torch.tensor(mano_params[key]).to(device).unsqueeze(0)
        
    """Adjust the cam params"""
    logging.info("Initializing hand camera parameters")
    hand_cam = load_cam(hand_cam_file, device=device)
    logging.warning(f"hand_cam: {hand_cam}")
    hand_cam["projection"] = torch.tensor(get_projection(hand_cam, origin_w, origin_h)).float().to(device)
    
    logging.info("Initializing object camera parameters")
    obj_cam, object_pose = get_obj_cam(device, dir)
        
    obj_verts = torch.tensor(obj_mesh.vertices).float().cuda()
    obj_faces = obj_mesh.faces
    
    """ process object mesh into sdf """
    logging.info("Computing object mesh SDF distance function")
    obj_mesh = trimesh.Trimesh(obj_verts.clone().cpu().numpy(), obj_faces)
        
    obj_sdf_origin = obj_mesh.bounding_box.centroid.copy()
    obj_sdf_scale = 2.0 / np.max(obj_mesh.bounding_box.extents)
    os.makedirs(osp.join(dir, "easyhoi", "SDF"), exist_ok=True)
    obj_sdf_path = osp.join(dir, "easyhoi", "SDF","sdf.npy")
    
    sdf_valid = False

    if os.path.exists(obj_sdf_path):
        obj_sdf_voxel = np.load(obj_sdf_path, allow_pickle=True)
        logging.info(f"Loading sdf from {obj_sdf_path}")
        if not (obj_sdf_voxel.size == 1 and obj_sdf_voxel.item() is None):
            logging.info(f"SDF loaded successfully")
            sdf_valid = True
        else:
            logging.warning(f"SDF loading failed, recalculating")
    

    if not sdf_valid:
        
        obj_sdf_voxel = try_until_success(
                                    mesh_to_voxels,
                                    verbose=True,
                                    mesh = obj_mesh,
                                    voxel_resolution=64, 
                                    check_result=True, 
                                    surface_point_method="scan",
                                    sign_method="depth",
                                    sample_point_count=500000,
                                )
        np.save(obj_sdf_path, obj_sdf_voxel)
        
    if obj_sdf_voxel is None:
        logging.error(f"obj sdf 为空, 跳过当前item: {dir.split('/')[-1]}")
        return None
        
    obj_sdf = {"origin": torch.FloatTensor(obj_sdf_origin).cuda(),
               "scale": torch.FloatTensor([obj_sdf_scale]).cuda(),
               "voxel": torch.FloatTensor(obj_sdf_voxel).cuda()}
    
    
    ret = {
            "name": dir.split('/')[-1],
            "img_path": img_path,
            "resolution":[origin_h,origin_w],
            "image": np.array(input_image),
            "hand_mask": torch.tensor(hand_mask == 0).cuda(), # the hand mask for inpaint has zero for hand region
            "obj_mask": torch.tensor(obj_mask > 0).cuda(),
            "inpaint_mask": torch.tensor(inpaint_mask > 0).cuda(),
            "hand_cam": hand_cam,   # [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
            "obj_cam": obj_cam, # inverse of obj_pose
            "mano_params": mano_params,# load from .pt
            "object_verts": obj_verts.unsqueeze(0),
            "object_pose": object_pose,
            "object_faces": torch.LongTensor(obj_mesh.faces).cuda(),
            "object_colors": object_colors,
            "object_sdf": obj_sdf,
            "cam_transl": torch.tensor(hand_info["cam_transl"]).unsqueeze(0).float().cuda(),
            "is_right": hand_info["is_right"],
            "mesh_path": osp.join(dir, "pc", "fixed.obj")
        }
    return ret

def main(dir: str) -> None:
    lock_file = osp.join(dir, "easyhoi", f"easyhoi.lock")
    
    easyhoi_dir = osp.join(dir, "easyhoi")
    if osp.exists(easyhoi_dir):
        import shutil
        shutil.rmtree(easyhoi_dir)

    hoi_sync = HOI_Sync(dir)
    
    data_item = load_data_single(dir)
    logging.info(f"处理: {dir.split('/')[-1]}")
    
    vis_output_dir = osp.join(dir, "easyhoi", "debug")
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # -----------优化--------------
    
    with open(lock_file, 'w') as f:
        pass  # This creates an empty file
        
        
        hoi_sync.get_data(data_item)
        
        hoi_sync.get_hamer_hand_mask()
        
        # stage 1: camera setup
        logging.info("step1:优化相机参数")
        start_time = time.time()
        hoi_sync.optim_obj_cam()
        end_time = time.time()
        logging.info(f"step1运行时间: {end_time - start_time}")
        hoi_sync.export(prefix="camera_setup")
        hoi_sync.export_mano(prefix="camera_setup")
        
        # Stage 2: contact alignment
        logging.info("step2:手部接触对齐")
        start_time = time.time()
        hoi_sync.run_handpose_global()
        end_time = time.time()
        logging.info(f"step2运行时间: {end_time - start_time}")
        hoi_sync.export(prefix="contact_alignment")
        hoi_sync.export_mano(prefix="contact_alignment")
        
        # Stage 3: hand refine
        logging.info("step3:手部细化")
        start_time = time.time()
        hoi_sync.run_handpose_refine()
        end_time = time.time()
        logging.info(f"step3运行时间: {end_time - start_time}")
        
        hoi_sync.export(prefix="final")
        hoi_sync.export_mano(prefix="final")
        
        os.remove(lock_file)
        
    
if __name__ == "__main__":
    import sys
    dir = sys.argv[1]
    main(dir)