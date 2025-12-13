from typing import Any, Dict, Tuple
from yacs.config import CfgNode
import pickle
import json
import os
import os.path as osp
from pathlib import Path
import sys
import time
from PIL import Image
import trimesh
import torch
from torch import optim, nn, utils, Tensor
from torchvision import transforms

from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops.knn import knn_gather, knn_points

from chamfer_distance import ChamferDistance
from geomloss import SamplesLoss

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.logging import log_init
logging = log_init()


from src.utils.losses import (
    compute_obj_contact_loss, 
    soft_iou_loss, 
    compute_nonzero_distance_2d,
    chamfer_dist_loss, 
    anatomy_loss, 
    compute_penetr_loss, 
    compute_h2o_sdf_loss,
    # DROTLossFunction,
    compute_sinkhorn_loss,
    compute_sinkhorn_loss_rgb,
    compute_depth_loss,
    compute_obj_contact,
    compute_hand_contact,
    icp_with_scale,
    statistical_outlier_removal,
    moment_based_comparison
)

from src.utils.cam_utils import verts_transfer_cam, center_looking_at_camera_pose, get_projection
from src.utils.mesh_utils import render_mesh, pc_to_sphere_mesh
from src.utils import geom_utils, hand_utils, image_utils
from src.utils.utils import should_run_icp

from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.axislayer import AxisAdaptiveLayer, AxisLayerFK
from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.anchorlayer import AnchorLayer

import nvdiffrast.torch as dr

ToPILImage = transforms.ToPILImage()

# 核心类, 包含三阶段的手物对齐和优化
class HOI_Sync:
    def __init__(self, dir, project_root=None):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            logging.error("CUDA not avilable, use CPU")
            
        if project_root is None:
             raise ValueError("project_root must be provided")
             
        # Build paths internally based on project_root
        project_root_path = Path(project_root)
        
        easyhoi_assets_root = str(project_root_path / "thirdparty" / "EasyHOI" / "assets")
        
        # MANO assets: .../Project_refactoring/checkpoints/hamer/_DATA/data/mano
        mano_assets_root = str(project_root_path / "checkpoints" / "hamer" / "_DATA" / "data" / "mano")
        self.mano_assets_root = mano_assets_root
        
        """ Hyperparameters """
        self.obj_iteration = 0  # stage1, 不需要optim
        self.hand_refine_iteration = 100  # stage3
        self.icp_fix_R = True  # 固定旋转
        self.icp_fix_scale = True  # 固定尺度
        
        """ for optim """
        self.L1Loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.sinkhorn_loss = SamplesLoss('sinkhorn')
        self.axisFK = AxisLayerFK(mano_assets_root=mano_assets_root).to(self.device)
        self.anchor_layer = AnchorLayer(anchor_root=osp.join(easyhoi_assets_root, "anchor")).to(self.device)
        self.anatomyLoss = AnatomyConstraintLossEE().to(self.device)
        self.anatomyLoss.setup()
        # self.chd_loss = ChamferDistance()
        self.loss_weights = {
            # step3优化损失
            "contact": 5.0,
            "penetr": 20.0,
            "loss_2d": 1.0,
            "regularize": 2.0,
            # icp接触对齐损失
            "error_3d": 10.0,  # 3D误差权重
            "error_2d": 1.0,   # 2D误差权重
        }
        self.param_dim = {'hand':[('scale',1), ('transl',3)],
                          'obj': [('scale',1), ('transl',3), ('orient',3)]}
        
        """ global params"""
        self.global_params = {
            'hand': torch.FloatTensor([1, 0, 0, 0]).to(self.device),# NOTE: hand scale and transl
            'obj': torch.FloatTensor([1, 0,0,0, 0,0,0]).to(self.device),
        }
        self.T = None
        """
        # init, include scale and transl
        self.global_params['hand']
        # from json
        self.mano_params 
        # fullpose
        fullpose = torch.cat([self.mano_params["global_orient"], self.mano_params["hand_pose"]], dim=1)
        self.mano_params['fullpose'] = matrix_to_axis_angle(fullpose).reshape(-1, 16*3) #[B, 16* 3]
        # betas
        self.mano_params['betas']
        """
        
        for key in self.global_params:
            self.global_params[key].requires_grad_(True)
                
        """ for log """
        self.global_step = 0
        """ for render """
        self.glctx = dr.RasterizeCudaContext()
        
        """ for obj cam optim """
        self.phi_center = None
        self.phi_range = None
        
        
        """ for mano template """
        self.mano_layer = ManoLayer(side="right", mano_assets_root=mano_assets_root).to(self.device)
        with open(osp.join(easyhoi_assets_root, "mano_backface_ids.pkl"), "rb") as f:
            self.hand_backface_ids = pickle.load(f)
            
        contact_zone = np.load(osp.join(easyhoi_assets_root, "contact_zones.pkl"), allow_pickle=True)['contact_zones']# 接触区域(掌心)
        self.hand_contact_zone = []
        for key in contact_zone:
            self.hand_contact_zone += contact_zone[key]
        
        
        """ for export """
        self.vis_mid_results = True
        self.out_dir = osp.join(dir, "easyhoi")
        if self.vis_mid_results:
            os.makedirs(osp.join(self.out_dir, "midresult", "test_hand_obj_cam"), exist_ok=True)
            os.makedirs(osp.join(self.out_dir, "midresult", "test_obj_cam"), exist_ok=True)
            os.makedirs(osp.join(self.out_dir, "midresult", "test_hand_cam"), exist_ok=True)
            
            
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(osp.join(self.out_dir, "render"), exist_ok=True)
        os.makedirs(osp.join(self.out_dir, "eval"), exist_ok=True)
        os.makedirs(osp.join(self.out_dir, "vis"), exist_ok=True)
        os.makedirs(osp.join(self.out_dir, "retarget"), exist_ok=True)
        os.makedirs(osp.join(self.out_dir, "contact"), exist_ok=True)

    
    def get_params_for(self, option):
        key = option
        res = {}
        offset = 0
        for pair in self.param_dim[option]:
            name, dim = pair
            res[name] = self.global_params[key][offset:offset+dim]
            offset += dim
        return res
        
    # 获取pipeline的输入数据, 通过data_item传入
    def get_data(self, data_item, **kwarg):
        self.data = data_item   # 优化阶段的输入数据
        """
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
        """
        self.mano_params = data_item["mano_params"]
        
        self.hand_faces = self.mano_layer.get_mano_closed_faces().to(self.device)
        if not self.data["is_right"]:
            self.hand_faces = self.hand_faces[:,[0,2,1]] # faces for left hand
        
        fullpose = torch.cat([self.mano_params["global_orient"], self.mano_params["hand_pose"]], dim=1)
        self.mano_params['fullpose'] = matrix_to_axis_angle(fullpose).reshape(-1, 16*3) #[B, 16* 3]
        
        
    def log(self, value_dict, step, log_dir="./logs/optim", tag="optim"):
        output = ""
        for key in value_dict:
            if isinstance(value_dict[key], torch.Tensor):
                output += f"{key}:{value_dict[key].item():.4e};"
            else:
                output += f"{key}:0;"
    
    def get_mano_output(self):
        mano_params = self.mano_params
        fullpose = mano_params['fullpose']
        betas = mano_params['betas']
        mano_output: MANOOutput = self.mano_layer(fullpose, betas)
        return mano_output
    
    def diffrender_proj(self, pos, cam):
        ones = torch.ones(1, pos.shape[1], 1).to(pos.device)
        pos = torch.cat((pos, ones), dim=2).float() # augumented pos
        
        view_matrix = torch.cat([cam["extrinsics"], torch.tensor([[0,0,0,1]], device=pos.device)], dim=0)
        view_matrix = torch.inverse(view_matrix)
        proj_matrix = cam["projection"]
        
        mat = (proj_matrix @ view_matrix).unsqueeze(0)
        # mat = proj_matrix.unsqueeze(0)
        pos_clip = pos @ mat.mT
        
        return pos_clip
    
    def get_hamer_hand_mask(self, enlargement=20):
        mano_output = self.get_mano_output()
        hand_verts = self.get_hand_for_handcam(mano_output.verts, scale=1., transl=torch.zeros(3, device=self.device))
        
        pos_clip = self.diffrender_proj(hand_verts, self.data["hand_cam"])
        tri = self.hand_faces.squeeze().int()
        
        color = torch.tensor([1, 0, 0]).repeat(hand_verts.shape[1], 1)
        
        color = color.unsqueeze(0).float().to(hand_verts.device)
        
        rast, _ = dr.rasterize(self.glctx, pos_clip, tri, resolution=self.data["resolution"])
        out, _ = dr.interpolate(color, rast, tri)
        out = dr.antialias(out, rast, pos_clip, tri)
        img = torch.flip(out[0], dims=[0]) # Flip vertically.
        
        self.data["hamer_hand_mask"] = img[...,0].detach()
        
        tgt_mask = self.data["hamer_hand_mask"]
        # Get the indices of non-zero elements
        indices = torch.nonzero(tgt_mask, as_tuple=False)
        if len(indices) == 0:
            return None, None
        min_row, min_col = indices.min(dim=0)[0]
        max_row, max_col = indices.max(dim=0)[0]

        # Enlarge the bounding box
        min_row = max(0, min_row - enlargement)
        min_col = max(0, min_col - enlargement)
        max_row = min(tgt_mask.shape[0] - 1, max_row + enlargement)
        max_col = min(tgt_mask.shape[1] - 1, max_col + enlargement)

        # Zero out pixels outside the enlarged bounding box in the hand mask
        modified_hand_mask = torch.zeros_like(self.data["hand_mask"])
        modified_hand_mask[min_row:max_row+1, min_col:max_col+1] = self.data["hand_mask"][min_row:max_row+1, min_col:max_col+1]
        self.data["hand_mask"] = modified_hand_mask
        
        hand_iou = soft_iou_loss(self.data["hamer_hand_mask"], self.data["hand_mask"])
        o2h_dist = compute_nonzero_distance_2d(self.data["hamer_hand_mask"], 
                                              self.data["inpaint_mask"])
        
        if self.vis_mid_results:
            name = self.data['name']
            
            mask = self.data["hamer_hand_mask"].cpu().numpy()
            mask = np.clip(np.rint(mask * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
            Image.fromarray(mask).save(osp.join(self.out_dir, f"midresult/test_hand_cam/{name}_hamer.png"))
            
            mask = self.data["hand_mask"].cpu().numpy()
            mask = np.clip(np.rint(mask * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
            Image.fromarray(mask).save(osp.join(self.out_dir, f"midresult/test_hand_cam/{name}_seg.png"))
            
        return hand_iou.item(), o2h_dist.item()
            
    
    def render_hoi_image(self, 
                    hand_verts, hand_faces, 
                    obj_verts, obj_faces,
                    resolution,
                    hand_color=None,
                    ):
        vtx_offset = hand_verts.shape[1]
        verts = torch.cat([hand_verts, obj_verts], dim=1)
        tri = torch.cat([hand_faces, obj_faces + vtx_offset], dim=0).int()
        # tri = hand_faces
        if hand_color is None:
            col_hand = torch.tensor([1, 0, 0]).repeat(hand_verts.shape[1], 1)
        else:
            col_hand = hand_color
        col_obj = torch.tensor([0, 1, 0]).repeat(obj_verts.shape[1], 1)
        color = torch.cat((col_hand, col_obj), dim=0).to(self.device) # color for each vertex
        color = color.unsqueeze(0).float()
        
        projections = self.data["obj_cam"]["projection"]
        c2ws = self.data["obj_cam"]["extrinsics"]
        if c2ws.shape[0] == 3:
            c2ws = torch.cat([c2ws, torch.tensor([[0,0,0,1]], device=c2ws.device)], dim=0)
        
        img = self.renderer(verts, tri, color, projections, c2ws, resolution)
        
        return img
    
    def render_hand_image(self, hand_verts):
        color_hand = torch.FloatTensor([1, 0, 0]).repeat(hand_verts.shape[1], 1)
        color_hand = color_hand.unsqueeze(0).to(hand_verts.device)
        
        projections = self.data["obj_cam"]["projection"]
        c2ws = self.data["obj_cam"]["extrinsics"]
        
        if c2ws.shape[0] == 3:
            c2ws = torch.cat([c2ws, torch.tensor([[0,0,0,1]], device=c2ws.device)], dim=0)
        
        img = self.renderer(hand_verts, self.hand_faces.squeeze().int(), color_hand, projections, c2ws, self.data["resolution"])
        return img
        
    
    # stage1: 优化相机位置
    def optim_obj_cam(self):
        # TODO:使用物体图像mask替代inpaint_mask
        gt_obj_mask = self.data["inpaint_mask"].float()
        # gt_obj_mask = self.data["obj_mask"].float()
        print(gt_obj_mask.shape)
        resolution = gt_obj_mask.shape
        params = {
            "boost": 3,
            "alpha": 0.98,
            "loss": "IoU", #"sinkhorn", 
            "step_size": 1e-2,
            "optimizer": torch.optim.Adam, 
            "remesh": [50,100,150], 
        }
        # 物体mesh
        verts = self.data["object_verts"]
        tri = self.data["object_faces"].int()
        color_obj = torch.FloatTensor([0, 1, 0]).repeat(verts.shape[1], 1)
        color_obj = color_obj.unsqueeze(0).to(self.device)
        
        step_size = params.get("step_size") # Step size
        optimizer = params.get("optimizer", torch.optim.Adam) # Which optimizer to use
        
        projections = self.data["obj_cam"]["projection"]
        
        c2ws = self.data["obj_cam"]["extrinsics"]
        
        
        device = projections.device
        projections_origin = projections.clone()
        projections_residual = torch.nn.Parameter(torch.zeros((4, 4), device=device, dtype=projections_origin.dtype))
        # projections_residual = torch.nn.Parameter(torch.zeros((1), device=device, dtype=projections_origin.dtype))
        projections_mask = torch.tensor([   # 相机内参mask, 只优化fx fy, 固定cx cy
                [1., 0., 0., 0.], 
                [0., 1., 0., 0.], 
                [0., 0., 0., 0.], 
                [0., 0., 0., 0.]
            ], device=device, dtype=projections.dtype)
        
        # c2ws_origin = c2ws.clone()
        c2ws_residual = torch.nn.Parameter(torch.zeros(6, device=device, dtype=c2ws.dtype))
        logging.info(f"优化相机外参(旋转+平移), 相机内参(fx+fy)")
        opt = optimizer([projections_residual, c2ws_residual], lr=step_size)
        
        if params["loss"] == "l1":
            loss_func = torch.nn.L1Loss()
        elif params["loss"] == "l2":
            loss_func = torch.nn.MSELoss()
        elif params["loss"] == "IoU":
            loss_func = soft_iou_loss
        else:
            loss_func = compute_sinkhorn_loss
            
            
        c2ws_r_orig, c2ws_t_orig, c2ws_s_orig = geom_utils.matrix_to_axis_angle_t(c2ws)
        
        
        # c2ws = self.find_c2ws_init(verts, tri, color_obj, projections, resolution, gt_obj_mask)
        # c2ws_r_orig, c2ws_t_orig, c2ws_s_orig = geom_utils.matrix_to_axis_angle_t(c2ws)
        
        logging.info(f"obj_iteration: {self.obj_iteration}")
        with torch.no_grad():
            img = self.renderer(verts, tri, color_obj, projections, c2ws, resolution=resolution)
            mask_opt = img[..., 1]
            mask_init = mask_opt.clone()

        for i in range(self.obj_iteration):
            projections = projections_origin + projections_residual * projections_mask
            c2ws_r = c2ws_r_orig + c2ws_residual[:3] * 0.1 # NOTE:control the step of rot
            c2ws_t = c2ws_t_orig + c2ws_residual[3:]
            c2ws = geom_utils.axis_angle_t_to_matrix(c2ws_r, c2ws_t, c2ws_s_orig)

            # 渲染物体 并获得mask
            img = self.renderer(verts, tri, color_obj, projections, c2ws, resolution=resolution)
            mask_opt = img[..., 1] # green channel
            
            # if not torch.any(mask_opt>0):
            #     return False
            
            # 损失函数
            iou_loss = loss_func(mask_opt, gt_obj_mask)
            
            if iou_loss > 0.9:
                sinkhorn_loss = compute_sinkhorn_loss(mask_opt.contiguous(), gt_obj_mask.contiguous())
                loss = sinkhorn_loss + iou_loss
            else:
                sinkhorn_loss = 0
                loss = iou_loss
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            self.log({
                "sinkhorn loss": sinkhorn_loss,
                "total loss": loss
                }, step=i
            )
            if loss.item() < 0.01:
                break
        
        self.data["obj_cam"]["projection"] = projections.detach()
        self.data["obj_cam"]["extrinsics"] = c2ws.detach()[:3, :] # (3, 4)
        logging.info(f"优化相机外参(旋转+平移), 相机内参(fx+fy)完成")
        
        # 深度剥离  一个像素点对应n个物体表面
        logging.info(f"计算接触点, 作为step2的输入")
        object_depth, object_rast = self.depth_peel(verts, tri, projections, c2ws, resolution,
                                            znear=0.1, zfar=100)  
        self.object_depth = object_depth.squeeze().detach() # [num_layers, H, W]
        self.object_rast = object_rast.squeeze().detach() # [num_layers, H, W, 4]
        
        obj_mesh = trimesh.Trimesh(vertices=verts.squeeze().cpu().numpy(), 
                                   faces=tri.squeeze().cpu().numpy(),
                                   process=False)
        normals = torch.tensor(obj_mesh.vertex_normals).float().to(self.device)
        
        # TODO: 检查接触点是否合理
        
        # 物体megapose重建mask - 物体图像mask(被遮挡) = front mask
        hoi_mask = self.data["hamer_hand_mask"].bool() & self.data["inpaint_mask"].bool()
        front_mask = (hoi_mask & self.data["inpaint_mask"] & (~self.data["obj_mask"])).int()
        # 计算接触点
        obj_pts_front, obj_contact_normals_front, contact_mask_front = compute_obj_contact( 
            side='obj_front',
            mask=front_mask,
            verts=verts.squeeze(),
            faces=tri.squeeze(),
            normals=normals,
            rast=self.object_rast
        )
        
        # 物体遮挡手
        # 手部hamer重建mask - 手部图像mask(被遮挡) = bask mask
        back_mask = (hoi_mask & self.data["inpaint_mask"] & self.data["hamer_hand_mask"].bool() & (~self.data["hand_mask"])).int()
        obj_pts_back, obj_contact_normals_back, contact_mask_back = compute_obj_contact(
            side='obj_back',
            mask=back_mask,
            verts=verts.squeeze(),
            faces=tri.squeeze(),
            normals=normals,
            rast=self.object_rast
        )
        
        if obj_pts_front is not None and obj_pts_back is not None:
            obj_pts = torch.concat([obj_pts_front, obj_pts_back], dim=0)
            obj_contact_normals = torch.concat([obj_contact_normals_front, obj_contact_normals_back], dim=0)
        else:
            obj_pts = None
            obj_contact_normals = None
        # 接触点坐标
        self.obj_contact = {'front': obj_pts_front, 
                            'back': obj_pts_back,
                            'both': obj_pts}
        # 接触点法向量
        self.obj_contact_normals = {'front': obj_contact_normals_front, 
                                    'back': obj_contact_normals_back,
                                    'both': obj_contact_normals}
        
        if self.vis_mid_results:
            # vis for check, can be commented
            img_id = self.data["name"]
            depth = self.object_depth.unsqueeze(1) # [N, 1, H, W]
            image_utils.save_depth(depth, 
                                fname=os.path.join(self.out_dir, f"midresult/test_obj_cam/{img_id}_depth"),
                                text_list=["layer_0", "layer_1", "layer_2", "layer_3"],
                                )
            
            mask = transforms.ToPILImage()(mask_opt)
            mask.save(os.path.join(self.out_dir, f"midresult/test_obj_cam/{img_id}_optimized.png"))
            mask = transforms.ToPILImage()(mask_init)
            mask.save(os.path.join(self.out_dir, f"midresult/test_obj_cam/{img_id}_init.png"))
            gt_obj_mask = transforms.ToPILImage()(gt_obj_mask)
            gt_obj_mask.save(os.path.join(self.out_dir, f"midresult/test_obj_cam/{img_id}_gt.png"))
            
            contact_mask_img = ToPILImage(front_mask.detach().cpu().float())
            contact_mask_img.save(os.path.join(self.out_dir, "contact", f"{img_id}_obj_mask_front.png"))
            
            contact_mask_img = ToPILImage(back_mask.detach().cpu().float())
            contact_mask_img.save(os.path.join(self.out_dir, "contact", f"{img_id}_obj_mask_back.png"))
            
            # output the object mesh with contact point
            verts = verts.squeeze().cpu().numpy()
            
            if obj_pts_front is not None:
                obj_pts_front = pc_to_sphere_mesh(obj_pts_front.cpu().numpy())
                num_front = obj_pts_front.vertices.shape[0]
                mesh = obj_mesh + obj_pts_front
            else:
                obj_pts_front = trimesh.Trimesh(vertices=[], faces=[])
                num_front = 0
                mesh = obj_mesh
            if obj_pts_back is not None:
                obj_pts_back = pc_to_sphere_mesh(obj_pts_back.cpu().numpy())
                mesh = mesh + obj_pts_back
            else:
                obj_pts_back = trimesh.Trimesh(vertices=[], faces=[])

            vertex_colors = np.ones((len(mesh.vertices), 4))  # [R, G, B, A]
            vertex_colors[len(verts):len(verts)+num_front, :] = [1.0, 0.0, 0.0, 1.0]  # Red
            vertex_colors[len(verts)+num_front:, :] = [0.0, 1.0, 0.0, 1.0]  # Green
            mesh.visual.vertex_colors = vertex_colors
            mesh.export(os.path.join(self.out_dir, "contact", f"{img_id}_obj.ply"))
            
        
    def depth_peel(self, verts, tri, projection, c2ws, resolution, num_layers=4, znear=0.1, zfar=100):
        device = projection.device
        
        ones = torch.ones(1, verts.shape[1], 1).to(device)
        pos = torch.cat((verts, ones), dim=2).float() # augumented pos
        
        view_matrix = torch.inverse(c2ws)
        mat = (projection @ view_matrix).unsqueeze(0)
        pos_clip = pos @ mat.mT
        
        depth_list = []
        rast_list = []
        with dr.DepthPeeler(self.glctx, pos_clip, tri, resolution) as peeler:
            for i in range(num_layers):
                rast, _ = peeler.rasterize_next_layer()
                rast = torch.flip(rast, dims=[1]) # Flip vertically.
                
                # rast has shape [minibatch_size, height, width, 4] 
                # and contains the main rasterizer output in order (u, v, z/w, triangle_id)
                depth = rast[..., 2] # [minibatch_size, H, W]
                mask = (depth == 0)
                depth = (2 * znear * zfar) / (zfar + znear - (zfar - znear) * depth)
                depth[mask] = 0
                
                depth_list.append(depth) 
                rast_list.append(rast)
                
        multi_depth = torch.stack(depth_list, dim=0) # [num_layers, minibatch_size, H, W]
        multi_rast = torch.stack(rast_list, dim=0) # [num_layers, minibatch_size, H, W, 4]
        return multi_depth, multi_rast
        
    
    def renderer(self, verts, tri, color, projection, c2ws, resolution):
        device = projection.device
        
        ones = torch.ones(1, verts.shape[1], 1).to(device)
        pos = torch.cat((verts, ones), dim=2).float() # augumented pos
        
        try:
            view_matrix = torch.inverse(c2ws)
        except:
            view_matrix = torch.linalg.pinv(c2ws)
        mat = (projection @ view_matrix).unsqueeze(0)
        pos_clip = pos @ mat.mT
        
        rast, _ = dr.rasterize(self.glctx, pos_clip, tri, resolution)
        out, _ = dr.interpolate(color, rast, tri)
        out = dr.antialias(out, rast, pos_clip, tri)
        img = torch.flip(out[0], dims=[0]) # Flip vertically.
        
        return img
    
    def optimize_pca(self, fullpose, betas, mano_layer, num_iterations=500, learning_rate=0.01):
        mano_output: MANOOutput = self.mano_layer(fullpose, betas)
        gt_verts = mano_output.verts
        gt_joints = mano_output.joints
        
        pca_params = torch.zeros([1,10], dtype=fullpose.dtype, device=fullpose.device, requires_grad=True)
        optimizer = optim.Adam([pca_params], lr=learning_rate)

        for i in range(num_iterations):
            optimizer.zero_grad()
            
            new_pose = torch.concat([fullpose[:,:3], pca_params], dim=-1)
            mano_output: MANOOutput = mano_layer(new_pose, betas)
            pred_verts = mano_output.verts
            pred_joints = mano_output.joints
            
            verts_loss = torch.mean((pred_verts - gt_verts) ** 2)
            joints_loss =  torch.mean((pred_joints - gt_joints) ** 2)
            
            loss = verts_loss + joints_loss
            
            loss.backward()
            optimizer.step()
            
            self.log({
                "verts loss": verts_loss,
                "joints loss": joints_loss
            }, step=i)

        # Return the optimized PCA parameters
        new_pose = torch.concat([fullpose[:,:3], pca_params], dim=-1)
        
        if self.vis_mid_results:
            os.makedirs(os.path.join(self.out_dir, "pca"), exist_ok=True)
            name = self.data['name']
            mesh = trimesh.Trimesh(pred_verts.detach().squeeze().cpu(), self.hand_faces.cpu())
            mesh.export(os.path.join(self.out_dir, "pca" , f"{name}_pca_hand.ply"))
            
            mesh = trimesh.Trimesh(gt_verts.detach().squeeze().cpu(), self.hand_faces.cpu())
            mesh.export(os.path.join(self.out_dir, "pca" , f"{name}_gt_hand.ply"))
            
        return new_pose.detach()
    
    def optim_handpose(self, pca_params, pca_params_orig, betas, mano_layer=None, iteration=None, total_iterations=None):
        if mano_layer is None:
            mano_output: MANOOutput = self.mano_layer(pca_params, betas)
        else:
            mano_output: MANOOutput = mano_layer(pca_params, betas)
            
        hand_verts = self.get_hand_verts(mano_output.verts, **self.get_params_for('hand'))
        obj_verts = self.transform_obj(**self.get_params_for('obj'))
        anchors = self.anchor_layer(hand_verts)
        gt_hand_mask = self.data["hand_mask"].float()
        gt_obj_mask = self.data["obj_mask"].float()
        
        """
        penetration and contact loss
        """
                
        penetr_loss, contact_loss = compute_h2o_sdf_loss(self.data["object_sdf"], 
                                            hand_verts,
                                            self.hand_contact_zone)
        
        # contact_loss = compute_obj_contact_loss(self.obj_pts.unsqueeze(0),
        #                                         hand_verts)
        
        contact_loss = self.loss_weights["contact"] * contact_loss
        penetr_loss = self.loss_weights["penetr"] * penetr_loss 
        loss_3d = (contact_loss + penetr_loss)
        
        """
        regularize loss
        """
        # reg_loss = anatomy_loss(mano_output, self.axisFK, self.anatomyLoss)
        # reg_loss = self.loss_weights["regularize"] * anatomy_loss(mano_output, self.axisFK, self.anatomyLoss)
        reg_loss = self.loss_weights["regularize"] * self.L1Loss(pca_params, pca_params_orig)
            
        """
        loss for hand mask under differentialable rendering
        """
        img = self.render_hoi_image(hand_verts=hand_verts,
                                hand_faces=self.hand_faces.squeeze(),
                                obj_verts=obj_verts,
                                obj_faces=self.data["object_faces"].squeeze(),
                                resolution=self.data["resolution"])
        
        pred_hand_mask = img[..., 0]
        
        pred_obj_mask = img[..., 1]
        
        if not torch.any(pred_hand_mask>0):
            loss_2d = 0
        else:
            # sinkhorn_loss = compute_sinkhorn_loss(pred_hand_mask.contiguous(), gt_hand_mask.contiguous()) + compute_sinkhorn_loss(pred_obj_mask.contiguous(), gt_obj_mask.contiguous())
            loss_2d = soft_iou_loss(pred_hand_mask, gt_hand_mask) #+ soft_iou_loss(pred_obj_mask, gt_obj_mask)
            # loss_2d = self.L1Loss(pred_hand_mask, gt_hand_mask) + self.L1Loss(pred_obj_mask, gt_obj_mask)
        
        loss_2d = self.loss_weights["loss_2d"] * loss_2d
        
        loss = (loss_3d +loss_2d +reg_loss)
        
        if iteration is not None and total_iterations is not None:
            if iteration == 0 or iteration == total_iterations - 1:
                self.log({
                        "contact": contact_loss, 
                        "penetr": penetr_loss,
                        "reg loss": reg_loss,
                        "2d loss": loss_2d,
                        "total loss": loss}, step=self.global_step)
        
        fullpose = mano_output.full_poses
        return loss, fullpose.detach(), pred_hand_mask.detach(), pred_obj_mask.detach()
    
    def optim_handpose_global(self, fullpose, betas, scale=None, use_3d_loss = False):
        mano_output: MANOOutput = self.mano_layer(fullpose, betas)
        if scale is None:
            hand_verts = self.get_hand_verts(mano_output.verts, **self.get_params_for('hand'))
        else:
            params = {k: v.clone() for k, v in self.get_params_for('hand').items()}
            params['scale'] = scale
            hand_verts = self.get_hand_verts(mano_output.verts, **params)
        
        info = {
            "mano_verts": mano_output.verts, 
            **self.get_params_for('hand'),
            "hand_verts": hand_verts, 
        }
        
        info = {k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in info.items()}
        
        """
        loss for hand mask under differentialable rendering
        """
        
        gt_hand_mask = self.data["hamer_hand_mask"].float() # no obj rendered
        
        img = self.render_hand_image(hand_verts)
        pred_hand_mask = img[...,0]
        
        if not torch.any(pred_hand_mask>0):
            iou = torch.Tensor([1.0])
            use_3d_loss = True
        else:
            iou = soft_iou_loss(pred_hand_mask, gt_hand_mask)
            
        if iou.item() >= 0.9:
            if torch.sum(pred_hand_mask) == 0:
                logging.error("Empty vector detected: pred_hand_mask is empty, rendered from current hand_verts")
            if torch.sum(gt_hand_mask) == 0:
                logging.error("Empty vector detected: gt_hand_mask is empty, rendered from initial MANO parameters of hamer model")
            sinkhorn_loss = compute_sinkhorn_loss(pred_hand_mask.contiguous(), gt_hand_mask.contiguous())
            loss_2d = sinkhorn_loss
        else:
            loss_2d = 10 * iou 
        
        if use_3d_loss:
            # 3d损失包括: 接触损失, 穿透损失
            penetr_loss, contact_loss = compute_h2o_sdf_loss(self.data["object_sdf"], 
                                                hand_verts,
                                                self.hand_contact_zone)
            loss_3d = (contact_loss * 10 + penetr_loss)
            loss = loss_2d + loss_3d
            self.log({"iou": iou,
                    "2d mask loss": loss_2d,
                    "3d loss": loss_3d
                    }, step=self.global_step)
        else:
            loss = loss_2d 
            self.log({"iou": iou,
                    "2d mask loss": loss_2d,
                    # "3d loss": loss_3d
                    }, step=self.global_step)

        return loss, pred_hand_mask, info, iou.item()
    
    # stage3: 抓取姿势优化
    def run_handpose_refine(self, outer_iteration = 10):  
        """ Fix object pose, optimize hand pose"""
        # init param
        fullpose:torch.Tensor = self.mano_params['fullpose'].detach().clone()
        betas:torch.Tensor = self.mano_params['betas'].clone()
        hand_layer = ManoLayer(use_pca=True, ncomps=10, mano_assets_root=self.mano_assets_root).to(self.device)
        
        pca_pose = self.optimize_pca(fullpose, betas, hand_layer)
        fullpose_residual = torch.nn.Parameter(torch.zeros_like(pca_pose))
        fullpose_mask = torch.ones_like(pca_pose)
        fullpose_mask[:, :3] = 0
        fullpose_residual.requires_grad_()
        betas.requires_grad_()
        
        params_group = [    # 优化的参数: 全局平移, 手部残差, 形状
            {'params': self.global_params['hand'], 'lr': 1e-2},
            {'params': fullpose_residual, 'lr': 1e-2},
            {'params': betas, 'lr': 1e-4},
        ]
        
        self.optimizer = optim.Adam(params_group)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5) 
        
        num_iterations = self.hand_refine_iteration
        _, _, init_hand_mask, _ = self.optim_handpose(pca_pose, pca_pose, betas, hand_layer)
            
        best_loss = float('inf')
        best_fullpose = None
        best_global_param = None
        total_iterations = outer_iteration * num_iterations
        for iteration in range(total_iterations):
            self.optimizer.zero_grad()
            pcapose_new = pca_pose + fullpose_residual * fullpose_mask
            loss, fullpose_new, pred_hand_mask, pred_obj_mask = self.optim_handpose(pcapose_new, pca_pose, betas, hand_layer, iteration=iteration, total_iterations=total_iterations)
                            
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            self.global_step = iteration
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_fullpose = fullpose_new.detach().clone()
                best_global_param = self.global_params['hand'].detach().clone()

        self.mano_params['fullpose'] = best_fullpose.detach()
        self.mano_params['betas'] = betas.detach()
        self.global_params['hand'] = best_global_param
        
        name = self.data['name']
        
        # vis for check, can be commented
        if self.vis_mid_results:
            pred_hand_mask = pred_hand_mask.cpu()
            pred_hand_mask = ToPILImage(pred_hand_mask)
            pred_hand_mask.save(osp.join(self.out_dir, 
                                            f"midresult/test_hand_obj_cam/{name}_optim_non_global.png"))
    
    # stage2优化: 手部全局平移
    def run_handpose_global(self):  
        """ Fix object pose, optimize hand pose"""
        # init param
        fullpose:torch.Tensor = self.mano_params['fullpose'].clone()
        betas:torch.Tensor = self.mano_params['betas'].clone()
        
        orient_res = torch.nn.Parameter(torch.zeros([1,3], device=self.device, dtype=fullpose.dtype))
        orient_res.requires_grad_()
        betas.requires_grad_()
        
        params_group = [
            {'params': self.global_params['hand'], 'lr': 5e-2}, # 由scale和transl组成
            # {'params': betas, 'lr': 1e-4},
            {'params': orient_res, 'lr': 1e-5},
        ]
        
        
        self.optimizer = optim.Adam(params_group)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5) 
        
        num_iterations = self.hand_refine_iteration
        use_3d_loss = False
        
        _, init_hand_mask, init_info, iou = self.optim_handpose_global(fullpose, betas)
        
        # 单次ICP配准, 直接使用初始fullpose
        logging.warning("run single icp")
        fullpose_new = fullpose.clone()
        best_global_params = self.global_params['hand'].detach().clone()
        best_fullpose = fullpose_new.detach().clone()
        
        hand_mesh, hand_verts, hand_contact, hand_c_normals = self.get_hand_contact(fullpose_new, betas.detach())
        # NOTE:优化translation, R,scale固定
        succ = self.optim_contact(hand_mesh, hand_verts, hand_contact, hand_c_normals) 
        if succ == True:
            best_global_params = self.global_params['hand'].detach().clone() # 更新平移
        else:
            logging.error("icp配准失败")
        
        loss, pred_hand_mask, _, _ = self.optim_handpose_global(fullpose_new, betas, 
                                                                            use_3d_loss=True)
        
        self.mano_params['fullpose'] = best_fullpose
        self.global_params['hand'] = best_global_params
        self.mano_params['betas'] = betas.detach()
        name = self.data['name']
        
        
        # vis for check, can be commented
        if self.vis_mid_results:
            pred_hand_mask = pred_hand_mask.detach().cpu()
            # mask = np.clip(np.rint(mask * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
            pred_hand_mask = ToPILImage(pred_hand_mask)
            
            pred_hand_mask.save(osp.join(self.out_dir, 
                                            f"midresult/test_hand_obj_cam/{name}_optim.png"))
            
            init_hand_mask = init_hand_mask.detach().cpu()
            init_hand_mask = ToPILImage(init_hand_mask)
            init_hand_mask.save(osp.join(self.out_dir, 
                                            f"midresult/test_hand_obj_cam/{name}_init.png"))
            
            gt_hand_mask = self.data["hamer_hand_mask"].cpu()
            gt_hand_mask = ToPILImage(gt_hand_mask.float())
            gt_hand_mask.save(osp.join(self.out_dir, 
                                        f"midresult/test_hand_obj_cam/{name}_gt.png"))
                
            
    def get_hand_contact(self, fullpose, betas):
        mano_output: MANOOutput = self.mano_layer(fullpose, betas)
        hand_verts_handcam = self.get_hand_for_handcam(mano_output.verts, scale=1., transl=torch.zeros(3, device=self.device))
        hand_verts_objcam = self.get_hand_verts(mano_output.verts, **self.get_params_for('hand'))
        
        hand_mesh_objcam = trimesh.Trimesh(vertices=hand_verts_objcam.detach().squeeze().cpu().numpy(), 
                                    faces=self.hand_faces.squeeze().cpu().numpy())
        
        # projections = self.data["hand_cam"]["projection"]
        # c2ws = self.data["hand_cam"]["extrinsics"]
        # c2ws = torch.cat([c2ws, torch.tensor([[0,0,0,1]], device=c2ws.device)], dim=0)
        # hand_depth, hand_rast = self.depth_peel(hand_verts_handcam, self.hand_faces.squeeze().int(), projections, c2ws, self.data["resolution"])
        
        projections = self.data["obj_cam"]["projection"]
        c2ws = self.data["obj_cam"]["extrinsics"]
        
        if c2ws.shape[0] == 3:
            c2ws = torch.cat([c2ws, torch.tensor([[0,0,0,1]], device=c2ws.device)], dim=0)
        hand_depth, hand_rast = self.depth_peel(hand_verts_objcam, self.hand_faces.squeeze().int(), projections, c2ws, self.data["resolution"])
        
        hoi_mask = self.data["hamer_hand_mask"].bool() & self.data["inpaint_mask"]
        front_mask = ( hoi_mask & self.data["inpaint_mask"] & (~self.data["obj_mask"])).int()
        
        normals = hand_mesh_objcam.vertex_normals.copy()
        normals = torch.Tensor(normals).float().to(self.device)
        hand_pts_front, hand_normals_front, contact_mask_front = compute_hand_contact(
                                                    side='hand_front',
                                                    mask = front_mask,
                                                    verts=hand_verts_objcam.squeeze(),
                                                    faces=self.hand_faces.squeeze(),
                                                    normals=normals,
                                                    rast=hand_rast.detach().squeeze(),
                                                    skipped_face_ids=self.hand_backface_ids
                                                )
        
        back_mask = ( hoi_mask & self.data["hamer_hand_mask"].bool() & (~self.data["hand_mask"])).int()
        hand_pts_back, hand_normals_back, contact_mask_back = compute_hand_contact(
                                                    side='hand_back',
                                                    mask = back_mask,
                                                    verts=hand_verts_objcam.squeeze(),
                                                    faces=self.hand_faces.squeeze(),
                                                    normals=normals,
                                                    rast=hand_rast.detach().squeeze(),
                                                    skipped_face_ids=self.hand_backface_ids
                                                )
        
        hand_pts = torch.concat([hand_pts_front, hand_pts_back], dim=0)
        hand_normals = torch.concat([hand_normals_front, hand_normals_back], dim=0)
        contact_mask = (contact_mask_front | contact_mask_back)
        hand_contact = {'front': hand_pts_front, 
                        'back': hand_pts_back,
                        'both': hand_pts}
        hand_contact_normal = {'front': hand_normals_front, 
                            'back': hand_normals_back,
                            'both': hand_normals}
        
        # <---------- For visualization ------------>
        if self.vis_mid_results:
            name = self.data['name']
            contact_mask_img = ToPILImage(front_mask.detach().cpu().float())
            contact_mask_img.save(os.path.join(self.out_dir, "contact", f"{name}_hand_mask_front.png"))
            
            contact_mask_img = ToPILImage(back_mask.detach().cpu().float())
            contact_mask_img.save(os.path.join(self.out_dir, "contact", f"{name}_hand_mask_back.png"))
            
            image_utils.save_depth(hand_depth.detach(), 
                                    os.path.join(self.out_dir, "contact", f"{name}_hand_depth"),
                                    text_list=["layer_0", "layer_1", "layer_2", "layer_3"])
            
            ids = torch.nonzero(contact_mask == 0)
            hand_depth[:, :, ids[:,0], ids[:,1]] = 0
            image_utils.save_depth(hand_depth.detach(), 
                                    os.path.join(self.out_dir, "contact", f"{name}_filtered_hand_depth"),
                                    text_list=["layer_0", "layer_1", "layer_2", "layer_3"])
        
        return hand_mesh_objcam, hand_verts_objcam, hand_contact, hand_contact_normal
    
    # 接触优化, 只优化translation, 并且加入mask损失避免局部最优
    def optim_contact(self, hand_mesh, hand_verts, hand_contact, hand_contact_normal):
        obj_verts = self.transform_obj(**self.get_params_for('obj'))
        gt_hand_mask = self.data["hand_mask"].float()
        
        min_error = float('inf')
        best_t = None
        best_side = None
        trans_hand_pts = {}

        for side in ['front', 'back', 'both']:
        # for side in ['both']:
            if self.obj_contact[side] is None:
                continue
            if len(hand_contact[side]) == 0:
                hand_contact[side] = hand_verts.squeeze()
                hand_contact_normal[side] = hand_mesh.vertex_normals
                hand_contact[side] = hand_contact[side][self.hand_contact_zone]
                hand_contact_normal[side] = hand_contact_normal[side][self.hand_contact_zone]
                hand_contact_normal[side] = torch.Tensor(hand_contact_normal[side]).float().to(self.device)
            R, t, scale, trans_hand_pts[side], error_3d = icp_with_scale(  # icp
                src_points=hand_contact[side],
                src_norm=-hand_contact_normal[side], # we hope the hand normal be opposite to object's
                tgt_points=self.obj_contact[side],
                tgt_norm=self.obj_contact_normals[side],
                fix_R=True,
                fix_scale=self.icp_fix_scale,
                device=self.device
            )
            center = hand_verts.mean(dim=1, keepdim=True)
            after_hand_verts = (scale * (hand_verts - center) + center) @ R.T + t

            img = self.render_hoi_image(after_hand_verts, self.hand_faces.squeeze(), 
                                obj_verts, self.data["object_faces"].squeeze(),
                                resolution=self.data["resolution"])
            pred_hand_mask = img[...,0]
            error_2d = soft_iou_loss(pred_hand_mask, gt_hand_mask) # 交叉验证, 避免icp陷入局部最优
            error = self.loss_weights["error_3d"] * error_3d + self.loss_weights["error_2d"] * error_2d
            print(side, 'error: ', float(error), 'error_2d: ', float(error_2d), 'error_3d: ', float(error_3d))
            
            if error < min_error and not torch.isnan(t).any():
                logging.info(f"stage2接触点icp配准:使用side: {side} 进行优化")
                best_t = t
                best_s = scale
                best_R = R
                best_side = side
                min_error = error
        
        if best_t is None:
            return False
        
        with torch.no_grad():
            self.global_params['hand'][1:] += best_t
            if not self.icp_fix_scale:
                self.global_params['hand'][0] *= best_s
                
        # after_hand_verts = (R @ (scale * hand_verts).T).T + t
        
        if not self.icp_fix_scale:
            center = hand_verts.mean(dim=1, keepdim=True)
            after_hand_verts = (best_s * (hand_verts - center) + center) @ R.T + best_t
        else:
            after_hand_verts = hand_verts @ R.T + best_t
        
        name = self.data['name']
        # output the hand mesh with contact point
        verts = hand_verts.detach().squeeze().cpu().numpy()
        hand_pts_front = pc_to_sphere_mesh(hand_contact['front'].detach().cpu().numpy())
        hand_pts_back = pc_to_sphere_mesh(hand_contact['back'].detach().cpu().numpy())
        
        num_front = hand_pts_front.vertices.shape[0]
        mesh = hand_mesh + hand_pts_front + hand_pts_back
        vertex_colors = np.tile([0.5,0.5,0.5,1], (len(mesh.vertices), 1))  # [R, G, B, A]
        vertex_colors[len(verts):len(verts)+num_front, :] = [1.0, 0.0, 0.0, 1.0]  # Red
        vertex_colors[len(verts)+num_front:, :] = [0.0, 1.0, 0.0, 1.0]  # Green
        
        mesh.visual.vertex_colors = vertex_colors
        mesh.export(os.path.join(self.out_dir, "contact", f"{name}_hand.ply"))
        
        # output the transformed hand mesh with contact point
        trans_hand_pts = trans_hand_pts[best_side]
        hand_mesh = trimesh.Trimesh(vertices=after_hand_verts.detach().squeeze().cpu().numpy(), 
                                    faces=self.hand_faces.squeeze().cpu().numpy())
        
        trans_hand_pts = pc_to_sphere_mesh(trans_hand_pts.detach().cpu().numpy())
        
        mesh = hand_mesh + trans_hand_pts
        vertex_colors = np.tile([0.5,0.5,0.5,1], (len(mesh.vertices), 1))  # [R, G, B, A]
        if best_side == 'front':
            vertex_colors[len(verts):, :] = [1.0, 0.0, 0.0, 1.0]  # Red
        elif best_side == 'back':
            vertex_colors[len(verts):, :] = [0.0, 1.0, 0.0, 1.0]  # Green
        else:
            vertex_colors[len(verts):, :] = [0.0, 0.0, 1.0, 1.0]  # Blue
        if vertex_colors.size > 0:
            mesh.visual.vertex_colors = vertex_colors
            mesh.export(os.path.join(self.out_dir, "contact", f"{name}_hand_after.ply"))
            
        return True
        
    
    def hamer_process(self, vertices):
        """
        check third_party/hamer/hamer/utils/renderer.py vertices_to_trimesh method
        """
        vertices = vertices + self.data["cam_transl"]
        rot = torch.tensor([[[1,0,0],
                            [0,-1,0],
                            [0,0,-1]]], dtype=torch.float, requires_grad=False).to(vertices.device)
        vertices = vertices @ rot.mT
        return vertices
    
    def get_hand_for_handcam(self, hand_verts, scale, transl, need_hamer_process=True):
        hand_verts = hand_verts * scale
        hand_verts = hand_verts + transl
        
        hand_verts[:,:,0] = (2*self.data["is_right"]-1)*hand_verts[:,:,0]
        
        if need_hamer_process is True:
            hand_verts = self.hamer_process(hand_verts)
        
        return hand_verts
    
    def get_hand_verts(self, hand_verts, scale, transl):
        # hand_verts = self.hamer_process(hand_verts)
        # hand_verts = verts_transfer_cam(hand_verts, self.data["hand_cam"], self.data["obj_cam"])
        hand_verts[:,:,0] = (2*self.data["is_right"]-1)*hand_verts[:,:,0]
        hand_verts = self.hamer_process(hand_verts) # 绕x旋转180
        
        _, self.T = verts_transfer_cam(hand_verts, self.data["hand_cam"], self.data["obj_cam"])#hand_cam -> obj_cam
        hand_verts = hand_verts @ (scale * self.T[:3, :3].mT) + (self.T[:3, 3] * scale + transl)
        return hand_verts
        
    
    def transform_obj(self, scale, transl, orient, need_hamer_process=True):
        # rot_mat = axis_angle_to_matrix(orient)
        # obj_verts = scale * self.data["object_verts"]@rot_mat.T + transl
        
        obj_verts = self.data["object_verts"]
        
        return obj_verts
    

    def export_mano(self, prefix=None):
        
        output_dir = osp.join(self.out_dir, "export_" + prefix)
        os.makedirs(output_dir, exist_ok=True)
            
        # TODO:export mano params
        mano_params = self.mano_params
        fullpose = mano_params['fullpose']
        betas = mano_params['betas']
        cam_transl = self.data["cam_transl"]
        is_right = self.data["is_right"]
        T = self.T
        hand_params = self.get_params_for('hand')
        cam_extrinsics = self.data["obj_cam"]["extrinsics"]
        cam_projection = self.data["obj_cam"]["projection"]
        # export json file
        def tensor_to_list(t):
            if hasattr(t, 'detach'):
                return t.detach().cpu().numpy().tolist()
            elif isinstance(t, np.ndarray):
                return t.tolist()
            elif hasattr(t, 'item'):  # numpy scalar
                return t.item()
            else:
                return t
        
        ret = {}
        ret['fullpose'] = tensor_to_list(fullpose)
        ret['betas'] = tensor_to_list(betas)
        ret['cam_transl'] = tensor_to_list(cam_transl)
        ret['is_right'] = tensor_to_list(is_right)
        ret['T'] = tensor_to_list(T)
        ret['cam_extrinsics'] = tensor_to_list(cam_extrinsics)
        ret['cam_projection'] = tensor_to_list(cam_projection)
        
        serializable_hand_params = {}
        for k, v in hand_params.items():
            serializable_hand_params[k] = tensor_to_list(v)
        ret['hand_params'] = serializable_hand_params
        
        with open(osp.join(output_dir, f"res.json"), 'w') as f:
            json.dump(ret, f, indent=4)
    
    def export(self, prefix=None):
        mano_output = self.get_mano_output()
        hand_verts = self.get_hand_verts(mano_output.verts, **self.get_params_for('hand'))
        obj_verts = self.transform_obj(**self.get_params_for('obj'))
        if prefix is not None:
            filename = f"{prefix}_{self.data['name']}"
        else:
            filename = self.data['name']
    
        # hand_faces = torch.tensor(self.hand_faces.astype(np.int32)).to(hand_verts.device)
        
        # export rendered image
        img = self.render_hoi_image(hand_verts, self.hand_faces.squeeze(), 
                                obj_verts, self.data["object_faces"].squeeze(),
                                resolution=self.data["resolution"])
        img = img.detach().cpu().numpy() 
        img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
        image = Image.fromarray(img)
        image.save(osp.join(self.out_dir, "render", f"{filename}.png"))
        
        
        # export meshes
        hand_verts = hand_verts.squeeze().detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=hand_verts, faces=self.hand_faces.squeeze().cpu())
        obj_mesh = trimesh.Trimesh(vertices=obj_verts.detach().squeeze().cpu(),
                                   faces=self.data["object_faces"].squeeze().cpu(),
                                   vertex_colors=self.data["object_colors"])
        # obj_mesh = self.transform_obj_origin(self.data["mesh_path"], **self.get_params_for('obj'))
        mesh = mesh+obj_mesh
        path = osp.join(self.out_dir, f"{filename}.ply")
        mesh.export(path)
        
        
            