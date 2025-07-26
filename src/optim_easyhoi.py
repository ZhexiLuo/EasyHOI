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
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from src.models.hoi_optim_module import HOI_Sync
from src.utils.logging import log_init

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
    
    input_image, new_size = resize_frame(input_image)
    
    hand_mask = hand_mask.resize(new_size)
    obj_mask = obj_mask.resize(new_size)
    inpaint_mask = inpaint_mask.resize(new_size)
    
    hand_mask, obj_mask, inpaint_mask = np.array(hand_mask), np.array(obj_mask), np.array(inpaint_mask)
    
    return input_image, hand_mask, obj_mask, inpaint_mask
        
def get_obj_cam(device, w, h, crop_bbox):
    # param from instantmesh
    DEFAULT_DIST = 4.5
    DEFAULT_FOV = 30.0
    
    cam_dist, fov = calc_orig_cam_params(DEFAULT_DIST, DEFAULT_FOV, W_orig=w, H_orig=h, crop_bbox=crop_bbox)
    cx, cy = adjust_principal_point(crop_bbox, w, h)
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    cam_pose = torch.FloatTensor([cam_dist, 0, 0]).cuda()
    
    ratio = w/h
    
    obj_cam = {'fx':focal_length, 'fy': focal_length*ratio, 'cx': cx, 'cy': cy}
    obj_cam["extrinsics"] = center_looking_at_camera_pose(cam_pose).to(device)
    obj_cam["projection"] = torch.FloatTensor(get_projection(obj_cam, width=w, height=h)).to(device)
    
    return obj_cam

def get_obj_cam_tripo(device, w, h):
    DEFAULT_DIST = 3.5  # 初始距离
    DEFAULT_FOV = 30.0  # 视野广度, 单位是度
    focal_length = 0.5 / np.tan(np.deg2rad(DEFAULT_FOV) * 0.5)  # 焦距
    cam_pose = torch.FloatTensor([DEFAULT_DIST, 0, 0]).cuda()  # 相机位置
    
    ratio = w/h  # 宽高比
    logging.info(f"ratio: {ratio}")
    
    obj_cam = {'fx':focal_length, 'fy': focal_length*ratio, 'cx': 0.5, 'cy': 0.5}
    obj_cam["extrinsics"] = center_looking_at_camera_pose(cam_pose).to(device)  # 以物体为中心, 相机朝向原点, 计算相机外参
    obj_cam["projection"] = torch.FloatTensor(get_projection(obj_cam, width=w, height=h)).to(device)    # 投影矩阵, 将3D坐标投影到2D坐标
    
    return obj_cam

def load_hamer_info(file_path):
    if not osp.exists(file_path):
        return None
    hamer_info = torch.load(file_path)
    boxes = hamer_info["boxes"]
    mano_params = hamer_info["mano_params"]
    for key, item in mano_params.items():
        mano_params[key] = torch.tensor(item)
    fullpose = torch.cat([mano_params["global_orient"], mano_params["hand_pose"]], dim=1)
    mano_params['fullpose'] = matrix_to_axis_angle(fullpose).reshape(-1, 16*3) #[B, 16* 3]
    
    info_list = []
    for i in range(len(boxes)):
        info = {"id": i}
        info["mano_params"] = {key:mano_params[key][i:i+1] for key in mano_params }
        
        for key in hamer_info:
            if key in ["batch_size", "mano_params"]:
                continue
            info[key] = hamer_info[key][i]
        info_list.append(info)
        
    return info_list

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
                  logging.error(f"可能是网络质量太差, 导致SDF相邻像素差距过大")
            if attempt < max_attempts:
                if verbose:
                    logging.info("Retrying ...")

    if verbose:
        logging.error(f"Function failed after {max_attempts} attempts.")
    return None  # Return None if all attempts fail

# 数据加载
def load_data_single(cfg: DictConfig, file, hand_id, is_tripo = False):
    # resize the input image, because the resolution for nvdiffrastmust must be [<=2048, <=2048]
    try:
        img_path = osp.join(cfg.input_dir, file)
        input_image = Image.open(img_path)
    except:
        img_path = osp.join(cfg.input_dir, file.replace(".png", ".jpg"))
        input_image = Image.open(img_path)
        
        
    img_fn = file.split(".")[0]
    
    if not os.path.exists(osp.join(cfg.inpaint_dir, f"{img_fn}.png")):
        logging.error(f"inpaint image not exist: {osp.join(cfg.inpaint_dir, f'{img_fn}.png')}")
        return None
    
    
    hand_mask = Image.open(osp.join(cfg.hand_mask_dir, f"{img_fn}.png"))
    obj_mask = Image.open(osp.join(cfg.obj_mask_dir, f"{img_fn}.png"))
    inpaint_mask = Image.open(osp.join(cfg.inpaint_dir, f"{img_fn}.png"))
    origin_w, origin_h = input_image.width, input_image.height
    
    input_image, hand_mask, obj_mask, inpaint_mask = image_process(input_image, hand_mask, obj_mask, inpaint_mask)
    
    w,h = input_image.width, input_image.height
    
    hand_cam_file = osp.join(cfg.hand_dir, f"{img_fn}_cam.json")
    obj_mesh_path = osp.join(cfg.obj_dir,img_fn, "fixed.obj")
    
    if not os.path.exists(obj_mesh_path):
        logging.error(f"{obj_mesh_path} 文件不存在, 跳过当前item")
        return None
    obj_mesh = trimesh.load(obj_mesh_path)
    object_colors = obj_mesh.visual.vertex_colors
    
    """ load hand info """
    logging.info(f"加载手部MANO参数")
    info = torch.load(osp.join(cfg.hand_dir, f"{img_fn}.pt"))
    
    hand_info = {'mano_params': {}}
    for key in info['mano_params']:
        if hand_id >= info['mano_params'][key].shape[0]:
            logging.error(f"手部MANO参数索引超出范围, 跳过当前item: {img_fn}")
            return None
        hand_info['mano_params'][key] = info['mano_params'][key][hand_id]
            
    hand_info.update({key: info[key][hand_id] 
                        for key in info 
                        if key not in ['batch_size','mano_params']})
    
    logging.info("HAMER:当前item为右手" if int(hand_info['is_right'].item()) else "HAMER:当前item为左手")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mano_params = hand_info["mano_params"]
    for key in mano_params:
        mano_params[key] = torch.tensor(mano_params[key]).to(device).unsqueeze(0)
        
    """Adjust the cam params"""
    logging.info("初始化相机参数")
    hand_cam = load_cam(hand_cam_file, device=device)
    hand_cam["projection"] = torch.tensor(get_projection(hand_cam, origin_w, origin_h)).float().to(device)
    
    bbox_file = osp.join(cfg.base_dir, "obj_recon/inpaint/hoi_box/", f"{img_fn}.json")
    with open(bbox_file, "r") as file:
        bbox = json.load(file)  # Directly loads the bbox as a list
    if is_tripo:
        obj_cam = get_obj_cam_tripo(device, origin_w, origin_h) # 虚拟相机初始化
    else:
        obj_cam = get_obj_cam(device, origin_w, origin_h, bbox)
        
    obj_verts = torch.tensor(obj_mesh.vertices).float().cuda()
    obj_faces = obj_mesh.faces
    if not is_tripo:
        # mirror reflection w.r.t. xy plane
        rot1 = torch.tensor([[1,0,0],
                            [0,1,0],
                            [0,0,-1]], dtype=torch.float, requires_grad=False).to(obj_verts.device)
        
        obj_verts = obj_verts @ rot1.T
        obj_faces = np.array(obj_mesh.faces[:, ::-1])
    else:
        rot1 = torch.tensor([[1,0,0],
                            [0,0,-1],
                            [0,1,0]], dtype=torch.float, requires_grad=False).to(obj_verts.device)
        
        obj_verts = obj_verts @ rot1.T
        logging.info("Done tripo trans")
    
    # process object mesh into sdf
    logging.info("处理物体网格为SDF")
    obj_mesh = trimesh.Trimesh(obj_verts.clone().cpu().numpy(), obj_faces)
        
    obj_sdf_origin = obj_mesh.bounding_box.centroid.copy()
    obj_sdf_scale = 2.0 / np.max(obj_mesh.bounding_box.extents)
    
    obj_sdf_path = osp.join(cfg.obj_dir,img_fn, "sdf.npy")
    
    sdf_valid = False

    if os.path.exists(obj_sdf_path):
        obj_sdf_voxel = np.load(obj_sdf_path, allow_pickle=True)
        logging.info(f"从{obj_sdf_path}加载sdf")
        if not (obj_sdf_voxel.size == 1 and obj_sdf_voxel.item() is None):
            logging.info(f"sdf加载成功")
            sdf_valid = True
        else:
            logging.warning(f"sdf加载失败, 重新计算")
    

    if not sdf_valid:
        
        obj_sdf_voxel = try_until_success(
                                    mesh_to_voxels,
                                    verbose=True,
                                    mesh = obj_mesh,
                                    voxel_resolution=64, 
                                    check_result=True, 
                                    surface_point_method="sample",
                                    sample_point_count=500000,
                                )
        np.save(obj_sdf_path, obj_sdf_voxel)
        
    if obj_sdf_voxel is None:
        logging.error(f"obj sdf 为空, 跳过当前item: {img_fn}")
        return None
        
    obj_sdf = {"origin": torch.FloatTensor(obj_sdf_origin).cuda(),
               "scale": torch.FloatTensor([obj_sdf_scale]).cuda(),
               "voxel": torch.FloatTensor(obj_sdf_voxel).cuda()}
    
    
    ret = {
            "name": img_fn,
            "img_path": img_path,
            "resolution":[h,w],
            "image": np.array(input_image),
            "hand_mask": torch.tensor(hand_mask == 0).cuda(), # the hand mask for inpaint has zero for hand region
            "obj_mask": torch.tensor(obj_mask > 0).cuda(),
            "inpaint_mask": torch.tensor(inpaint_mask > 0).cuda(),
            "hand_cam": hand_cam,
            "obj_cam": obj_cam,
            "mano_params": mano_params,
            "object_verts": obj_verts.unsqueeze(0),
            "object_faces": torch.LongTensor(obj_mesh.faces).cuda(),
            "object_colors": object_colors,
            "object_sdf": obj_sdf,
            "cam_transl": torch.tensor(hand_info["cam_transl"]).unsqueeze(0).float().cuda(),
            "is_right": hand_info["is_right"],
            "mesh_path": osp.join(cfg.obj_dir,img_fn, "fixed.obj")
        }
    return ret
    

@hydra.main(version_base=None, config_path="./configs", config_name="optim_teaser")
def main(cfg : DictConfig) -> None:
    # 初始化彩色日志
    log_init()
    
    exp_cfg = OmegaConf.create(cfg['experiments'])
    data_cfg = OmegaConf.create(cfg['data']) 
    if "is_tripo" in cfg:
        is_tripo = True
    else:
        is_tripo = False
        
    os.makedirs(cfg['out_dir'], exist_ok=True)
    
    exp_cfg['out_dir'] = cfg['out_dir']
    exp_cfg['log_dir'] = cfg['log_dir']
    if "hand_scale" in cfg:
        exp_cfg['hand_scale'] = cfg['hand_scale']
        
    
    filtered_file = osp.join(data_cfg.base_dir,f"{data_cfg.split}_filtered.npy")
    if os.path.exists(filtered_file):
        img_id_list = np.load(filtered_file, allow_pickle=True)
        hamer_info_list = [d['hamer_info'] for d in img_id_list]
        img_id_list = [d['img_id'] for d in img_id_list]
    else:
        img_id_list = []
        hamer_info_list = []
        for file in os.listdir(data_cfg.input_dir):
            logging.info(file)
            if not file.endswith(("png", "jpg")):
                continue
            img_id = file.split('.')[0]
            info = load_hamer_info(os.path.join(data_cfg.hand_dir, f"{img_id}.pt"))
            if info == None or len(info) == 0:
                logging.warning("No Hamer Info!")
                continue
            img_id_list.append(img_id)
            hamer_info_list.append(info)
    
    for i in trange(len(img_id_list)):
        img_fn = img_id_list[i]
        file = img_fn + ".png"
        hand_infos = hamer_info_list[i]
        hand_id = None
        
        path = osp.join(exp_cfg.out_dir, f"after_{img_fn}.ply")
        lock_file = osp.join(exp_cfg.out_dir, f"after_{img_fn}.lock")
        if osp.exists(path) or osp.exists(lock_file):
            continue
        
        progress_bar = tqdm(total=exp_cfg['iteration'], desc="Processing")
        hoi_sync = HOI_Sync(cfg=exp_cfg, progress_bar=progress_bar)
        
        """ Find the best match hand """
        # HAMER 可能从一张图片中检查到多个手, 找到最匹配的手
        min_iou = float('inf')
        for item in hand_infos:
            data_item = load_data_single(data_cfg, file, item["id"], is_tripo)
            if data_item is None:
                logging.warning("data_item is None")
                break
            hoi_sync.get_data(data_item)
            hand_iou, o2h_dist = hoi_sync.get_hamer_hand_mask() # 计算 手和 hand_mask 一致性; 和 object_mask 的距离
            logging.info(f"{item['id']} hand iou: {hand_iou}, o2h_dist: {o2h_dist}")
            if hand_iou is None or o2h_dist is None:
                iou = None
            else:
                iou = hand_iou + o2h_dist
                
            if iou is not None and iou < min_iou:
                min_iou = iou
                hand_id = item["id"]
            
        if hand_id is None:
            continue
        # 加载优化阶段的输入数据
        data_item = load_data_single(data_cfg, file, hand_id, is_tripo) # 加载最匹配的id
        if data_item is None:
            with open(lock_file, 'w') as f:
                f.write("Failed to construct a SDF!")
            continue
        
        logging.info(f"处理文件路径: {data_item['name']}")
        
        with open(lock_file, 'w') as f:
            pass  # This creates an empty file
        
        
        hoi_sync.get_data(data_item)
        
        hoi_sync.get_hamer_hand_mask()
        # hoi_sync.export_for_eval(prefix="before_camsetup")
        
        # stage 1: camera setup
        logging.info("step1:优化相机参数")
        start_time = time.time()
        hoi_sync.optim_obj_cam()
        end_time = time.time()
        logging.info(f"step1运行时间: {end_time - start_time}")
        hoi_sync.export(prefix="camera_setup")
        hoi_sync.export_for_eval(prefix="camera_setup")
        
        # Stage 2: contact alignment
        logging.info("step2:手部接触对齐")
        start_time = time.time()
        hoi_sync.run_handpose_global()
        end_time = time.time()
        logging.info(f"step2运行时间: {end_time - start_time}")
        hoi_sync.export(prefix="contact_alignment")
        hoi_sync.export_for_eval(prefix="contact_alignment")
        
        # Stage 3: hand refine
        logging.info("step3:手部细化")
        start_time = time.time()
        hoi_sync.run_handpose_refine()
        end_time = time.time()
        logging.info(f"step3运行时间: {end_time - start_time}")
        
        hoi_sync.export(prefix="final")
        hoi_sync.export_for_eval(prefix="final")
        
        os.remove(lock_file)
        
    
if __name__ == "__main__":
    main()