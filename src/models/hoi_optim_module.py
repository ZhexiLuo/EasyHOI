import os
import os.path as osp
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch import nn, optim
from torchvision import transforms
import numpy as np
from PIL import Image
import trimesh
from pytorch3d.transforms import matrix_to_axis_angle

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.logging import log_init
logging = log_init()

from src.utils.losses import (soft_iou_loss, compute_nonzero_distance_2d, compute_h2o_sdf_loss, 
                              compute_sinkhorn_loss, compute_obj_contact, compute_hand_contact, icp_with_scale)
from src.utils.cam_utils import verts_transfer_cam
from src.utils.mesh_utils import pc_to_sphere_mesh
from src.utils import image_utils

from manotorch.manolayer import ManoLayer, MANOOutput
import nvdiffrast.torch as dr
ToPILImage = transforms.ToPILImage()


class HOI_Sync:
    """Hand-object interaction optimization with 3-stage alignment."""
    
    # Loss weights
    DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
        "contact": 5.0,
        "penetr": 20.0,
        "loss_2d": 1.0,
        "regularize": 2.0,
        "error_3d": 10.0,
        "error_2d": 1.0,
    }
    
    # Parameter dimensions
    PARAM_DIM: Dict[str, List[Tuple[str, int]]] = {
        "hand": [("scale", 1), ("transl", 3)],
        "obj": [("scale", 1), ("transl", 3), ("orient", 3)],
    }
    
    DEFAULT_HAND_REFINE_ITERATION: int = 100
    
    def __init__(self, output_dir: str, project_root: str) -> None:
        self._setup_device()
        self._setup_paths(output_dir, project_root)
        self._setup_hyperparameters()
        self._setup_loss_config()
        self._setup_global_params()
        self._setup_render_context()
        self._setup_mano_model()
        self._setup_output_directories()
    
    def _setup_device(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            logging.error("CUDA not available, using CPU")
    
    def _setup_paths(self, output_dir: str, project_root: str) -> None:
        project_root_path = Path(project_root)
        
        self._easyhoi_assets_root = project_root_path / "thirdparty" / "EasyHOI" / "assets"
        self._mano_assets_root = project_root_path / "checkpoints" / "hamer" / "_DATA" / "data" / "mano"
        self.mano_assets_root = str(self._mano_assets_root)
        
        self.out_dir = Path(output_dir) / "easyhoi"
    
    def _setup_hyperparameters(self) -> None:
        self.hand_refine_iteration = self.DEFAULT_HAND_REFINE_ITERATION
        self.icp_fix_R = True
        self.icp_fix_scale = True
    
    def _setup_loss_config(self) -> None:
        self.L1Loss = nn.L1Loss()
        self.loss_weights = self.DEFAULT_LOSS_WEIGHTS.copy()
        self.param_dim = self.PARAM_DIM.copy()
    
    def _setup_global_params(self) -> None:
        self.global_params: Dict[str, torch.Tensor] = {
            "hand": torch.FloatTensor([1, 0, 0, 0]).to(self.device),  # scale(1) + transl(3)
            "obj": torch.FloatTensor([1, 0, 0, 0, 0, 0, 0]).to(self.device),  # scale(1) + transl(3) + orient(3)
        }
        self.T = None
        
        for param in self.global_params.values():
            param.requires_grad_(True)
        
        self.global_step = 0
    
    def _setup_render_context(self) -> None:
        self.glctx = dr.RasterizeCudaContext()
    
    def _setup_mano_model(self) -> None:
        self.mano_layer = ManoLayer(
            side="right", 
            mano_assets_root=self.mano_assets_root
        ).to(self.device)
        
        backface_path = self._easyhoi_assets_root / "mano_backface_ids.pkl"
        with open(backface_path, "rb") as f:
            self.hand_backface_ids: List[int] = pickle.load(f)
        
        contact_zone_path = self._easyhoi_assets_root / "contact_zones.pkl"
        contact_zone_data = np.load(contact_zone_path, allow_pickle=True)["contact_zones"]
        self.hand_contact_zone: List[int] = []
        for zone_indices in contact_zone_data.values():
            self.hand_contact_zone.extend(zone_indices)
    
    def _setup_output_directories(self) -> None:
        subdirs = ["render", "contact"]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for subdir in subdirs:
            (self.out_dir / subdir).mkdir(parents=True, exist_ok=True)

    
    def get_params_for(self, option):
        key = option
        res = {}
        offset = 0
        for pair in self.param_dim[option]:
            name, dim = pair
            res[name] = self.global_params[key][offset:offset+dim]
            offset += dim
        return res
        
    def get_data(self, data_item, **kwarg):
        self.data = data_item
        self.mano_params = data_item["mano_params"]
        
        self.hand_faces = self.mano_layer.get_mano_closed_faces().to(self.device)
        if not self.data["is_right"]:
            self.hand_faces = self.hand_faces[:,[0,2,1]]
        
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
        pos = torch.cat((pos, ones), dim=2).float()         
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

        modified_hand_mask = torch.zeros_like(self.data["hand_mask"])
        modified_hand_mask[min_row:max_row+1, min_col:max_col+1] = self.data["hand_mask"][min_row:max_row+1, min_col:max_col+1]
        self.data["hand_mask"] = modified_hand_mask
        
        hand_iou = soft_iou_loss(self.data["hamer_hand_mask"], self.data["hand_mask"])
        o2h_dist = compute_nonzero_distance_2d(self.data["hamer_hand_mask"], 
                                              self.data["inpaint_mask"])
        
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
            
    
    def prepare_contact_data(self):
        """Stage1: compute object contact points for ICP."""
        verts = self.data["object_verts"]
        tri = self.data["object_faces"].int()
        projections = self.data["obj_cam"]["projection"]
        c2ws = self.data["obj_cam"]["extrinsics"]
        resolution = self.data["inpaint_mask"].shape
        
        if c2ws.shape[0] == 4:
            c2ws = c2ws[:3, :]
            self.data["obj_cam"]["extrinsics"] = c2ws.detach()
        c2ws_4x4 = torch.cat([c2ws, torch.tensor([[0,0,0,1]], device=c2ws.device)], dim=0)
        
        logging.info("Computing contact points for stage2")
        object_depth, object_rast = self.depth_peel(verts, tri, projections, c2ws_4x4, resolution,
                                            znear=0.1, zfar=100)  
        self.object_depth = object_depth.squeeze().detach() # [num_layers, H, W]
        self.object_rast = object_rast.squeeze().detach() # [num_layers, H, W, 4]
        
        obj_mesh = trimesh.Trimesh(vertices=verts.squeeze().cpu().numpy(), 
                                   faces=tri.squeeze().cpu().numpy(),
                                   process=False)
        normals = torch.tensor(obj_mesh.vertex_normals).float().to(self.device)
        
        hoi_mask = self.data["hamer_hand_mask"].bool() & self.data["inpaint_mask"].bool()
        front_mask = (hoi_mask & self.data["inpaint_mask"] & (~self.data["obj_mask"])).int()
        obj_pts_front, obj_contact_normals_front, contact_mask_front = compute_obj_contact( 
            side='obj_front',
            mask=front_mask,
            verts=verts.squeeze(),
            faces=tri.squeeze(),
            normals=normals,
            rast=self.object_rast
        )
        
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
        self.obj_contact = {'front': obj_pts_front, 
                            'back': obj_pts_back,
                            'both': obj_pts}
        self.obj_contact_normals = {'front': obj_contact_normals_front, 
                                    'back': obj_contact_normals_back,
                                    'both': obj_contact_normals}
        
        # save contact outputs
        img_id = self.data["name"]
        contact_mask_img = ToPILImage(front_mask.detach().cpu().float())
        contact_mask_img.save(os.path.join(self.out_dir, "contact", f"{img_id}_obj_mask_front.png"))
        contact_mask_img = ToPILImage(back_mask.detach().cpu().float())
        contact_mask_img.save(os.path.join(self.out_dir, "contact", f"{img_id}_obj_mask_back.png"))
        
        verts_np = verts.squeeze().cpu().numpy()
        if obj_pts_front is not None:
            front_spheres = pc_to_sphere_mesh(obj_pts_front.cpu().numpy())
            num_front = front_spheres.vertices.shape[0]
            mesh = obj_mesh + front_spheres
        else:
            num_front = 0
            mesh = obj_mesh
        if obj_pts_back is not None:
            back_spheres = pc_to_sphere_mesh(obj_pts_back.cpu().numpy())
            mesh = mesh + back_spheres

        vertex_colors = np.ones((len(mesh.vertices), 4))
        vertex_colors[len(verts_np):len(verts_np)+num_front, :] = [1.0, 0.0, 0.0, 1.0]  # Red: front
        vertex_colors[len(verts_np)+num_front:, :] = [0.0, 1.0, 0.0, 1.0]  # Green: back
        mesh.visual.vertex_colors = vertex_colors
        mesh.export(os.path.join(self.out_dir, "contact", f"{img_id}_obj.ply"))
            
        
    def depth_peel(self, verts, tri, projection, c2ws, resolution, num_layers=4, znear=0.1, zfar=100):
        device = projection.device
        
        ones = torch.ones(1, verts.shape[1], 1).to(device)
        pos = torch.cat((verts, ones), dim=2).float()         
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
                
        multi_depth = torch.stack(depth_list, dim=0)
        multi_rast = torch.stack(rast_list, dim=0)
        return multi_depth, multi_rast
        
    
    def renderer(self, verts, tri, color, projection, c2ws, resolution):
        device = projection.device
        
        ones = torch.ones(1, verts.shape[1], 1).to(device)
        pos = torch.cat((verts, ones), dim=2).float()         
        try:
            view_matrix = torch.inverse(c2ws)
        except:
            view_matrix = torch.linalg.pinv(c2ws)
        mat = (projection @ view_matrix).unsqueeze(0)
        pos_clip = pos @ mat.mT
        
        rast, _ = dr.rasterize(self.glctx, pos_clip, tri, resolution)
        out, _ = dr.interpolate(color, rast, tri)
        out = dr.antialias(out, rast, pos_clip, tri)
        img = torch.flip(out[0], dims=[0])
        
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
        
        return new_pose.detach()
    
    def optim_handpose(self, pca_params, pca_params_orig, betas, mano_layer=None, iteration=None, total_iterations=None):
        if mano_layer is None:
            mano_output: MANOOutput = self.mano_layer(pca_params, betas)
        else:
            mano_output: MANOOutput = mano_layer(pca_params, betas)
            
        hand_verts = self.get_hand_verts(mano_output.verts, **self.get_params_for('hand'))
        obj_verts = self.transform_obj(**self.get_params_for('obj'))
        gt_hand_mask = self.data["hand_mask"].float()
        gt_obj_mask = self.data["obj_mask"].float()
        
        penetr_loss, contact_loss = compute_h2o_sdf_loss(self.data["object_sdf"], 
                                            hand_verts,
                                            self.hand_contact_zone)
        
        contact_loss = self.loss_weights["contact"] * contact_loss
        penetr_loss = self.loss_weights["penetr"] * penetr_loss 
        loss_3d = (contact_loss + penetr_loss)
        
        reg_loss = self.loss_weights["regularize"] * self.L1Loss(pca_params, pca_params_orig)
        
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
        
        gt_hand_mask = self.data["hamer_hand_mask"].float()
        
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
    
    def run_handpose_refine(self, outer_iteration = 10):
        """Stage3: optimize hand pose with fixed object."""
        fullpose:torch.Tensor = self.mano_params['fullpose'].detach().clone()
        betas:torch.Tensor = self.mano_params['betas'].clone()
        hand_layer = ManoLayer(use_pca=True, ncomps=10, mano_assets_root=self.mano_assets_root).to(self.device)
        
        pca_pose = self.optimize_pca(fullpose, betas, hand_layer)
        fullpose_residual = torch.nn.Parameter(torch.zeros_like(pca_pose))
        fullpose_mask = torch.ones_like(pca_pose)
        fullpose_mask[:, :3] = 0
        fullpose_residual.requires_grad_()
        betas.requires_grad_()
        
        params_group = [
            {'params': self.global_params['hand'], 'lr': 1e-2},
            {'params': fullpose_residual, 'lr': 1e-2},
            {'params': betas, 'lr': 1e-4},
        ]
        
        self.optimizer = optim.Adam(params_group)
        
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
            self.global_step = iteration
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_fullpose = fullpose_new.detach().clone()
                best_global_param = self.global_params['hand'].detach().clone()

        self.mano_params['fullpose'] = best_fullpose.detach()
        self.mano_params['betas'] = betas.detach()
        self.global_params['hand'] = best_global_param
    
    def run_handpose_global(self):
        """Stage2: optimize hand global translation."""
        fullpose:torch.Tensor = self.mano_params['fullpose'].clone()
        betas:torch.Tensor = self.mano_params['betas'].clone()
        
        orient_res = torch.nn.Parameter(torch.zeros([1,3], device=self.device, dtype=fullpose.dtype))
        orient_res.requires_grad_()
        betas.requires_grad_()
        
        params_group = [
            {'params': self.global_params['hand'], 'lr': 5e-2},
            {'params': orient_res, 'lr': 1e-5},
        ]
        
        
        self.optimizer = optim.Adam(params_group)
        
        num_iterations = self.hand_refine_iteration
        use_3d_loss = False
        
        _, init_hand_mask, init_info, iou = self.optim_handpose_global(fullpose, betas)
        
        logging.warning("run single icp")
        fullpose_new = fullpose.clone()
        best_global_params = self.global_params['hand'].detach().clone()
        best_fullpose = fullpose_new.detach().clone()
        
        hand_mesh, hand_verts, hand_contact, hand_c_normals = self.get_hand_contact(fullpose_new, betas.detach())
        succ = self.optim_contact(hand_mesh, hand_verts, hand_contact, hand_c_normals) 
        if succ == True:
            best_global_params = self.global_params['hand'].detach().clone()
        else:
            logging.error("ICP alignment failed")
        
        loss, pred_hand_mask, _, _ = self.optim_handpose_global(fullpose_new, betas, 
                                                                            use_3d_loss=True)
        
        self.mano_params['fullpose'] = best_fullpose
        self.global_params['hand'] = best_global_params
        self.mano_params['betas'] = betas.detach()
                
            
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
        
        # save contact outputs
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
    
    def optim_contact(self, hand_mesh, hand_verts, hand_contact, hand_contact_normal):
        """Optimize translation via ICP with mask loss."""
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
            R, t, scale, trans_hand_pts[side], error_3d = icp_with_scale(
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
            error_2d = soft_iou_loss(pred_hand_mask, gt_hand_mask)
            error = self.loss_weights["error_3d"] * error_3d + self.loss_weights["error_2d"] * error_2d
            print(side, 'error: ', float(error), 'error_2d: ', float(error_2d), 'error_3d: ', float(error_3d))
            
            if error < min_error and not torch.isnan(t).any():
                logging.info(f"Stage2 ICP: using side={side}")
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
                
        if not self.icp_fix_scale:
            center = hand_verts.mean(dim=1, keepdim=True)
            after_hand_verts = (best_s * (hand_verts - center) + center) @ R.T + best_t
        else:
            after_hand_verts = hand_verts @ R.T + best_t
        
        name = self.data['name']
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
        """Apply HaMeR camera transform."""
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
        hand_verts = self.hamer_process(hand_verts)
        
        _, self.T = verts_transfer_cam(hand_verts, self.data["hand_cam"], self.data["obj_cam"])
        hand_verts = hand_verts @ (scale * self.T[:3, :3].mT) + (self.T[:3, 3] * scale + transl)
        return hand_verts
        
    
    def transform_obj(self, scale, transl, orient, need_hamer_process=True):
        obj_verts = self.data["object_verts"]
        return obj_verts
    

    def export_mano(self, prefix=None):
        output_dir = osp.join(self.out_dir, "export_" + prefix)
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        img = self.render_hoi_image(hand_verts, self.hand_faces.squeeze(), 
                                obj_verts, self.data["object_faces"].squeeze(),
                                resolution=self.data["resolution"])
        img = img.detach().cpu().numpy() 
        img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8
        image = Image.fromarray(img)
        image.save(osp.join(self.out_dir, "render", f"{filename}.png"))
        
        hand_verts = hand_verts.squeeze().detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=hand_verts, faces=self.hand_faces.squeeze().cpu())
        obj_mesh = trimesh.Trimesh(vertices=obj_verts.detach().squeeze().cpu(),
                                   faces=self.data["object_faces"].squeeze().cpu(),
                                   vertex_colors=self.data["object_colors"])
        mesh = mesh+obj_mesh
        path = osp.join(self.out_dir, f"{filename}.ply")
        mesh.export(path)
        
        
            