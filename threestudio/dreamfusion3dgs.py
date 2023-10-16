from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussiansplatting.scene.cameras import Camera
from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud
import numpy as np
# from arguments import ModelParams, PipelineParams, OptimizationParams
from point_e.diffusion.configs import diffusion_from_config as diffusion_from_config_pointe
from point_e.diffusion.configs import DIFFUSION_CONFIGS

from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh
import io  
from PIL import Image  
from threestudio.systems.utils import parse_optimizer, parse_scheduler
import pytorch_lightning as pl
import os  
import signal  
import torch.nn.functional as F
import open3d as o3d

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


@threestudio.register("dreamfusion3dgs-system")
class DreamFusion3dgs(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        optimizer: dict = field(default_factory=dict)


    cfg: Config
    def configure(self) -> None:
       
        self.parser = ArgumentParser(description="Training script parameters")
        opt = OptimizationParams(self.parser)
        self.pipe = PipelineParams(self.parser)
        
        self.gaussian = GaussianModel(sh_degree = 2)
        bg_color = [1, 1, 1] if False else [0, 0, 0]

        self.background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.renderer = render
        point_cloud = self.pcb()
        self.cameras_extent = 4.0
        self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)

        self.gaussian.training_setup(opt)
        self.automatic_optimization = False
        self.all_train_step = 1200
        # self.trainstep = 0
    def pointe(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('creating base model...')
        base_name = 'base40M-textvec'
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config_pointe(DIFFUSION_CONFIGS[base_name])

        print('creating upsample model...')
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config_pointe(DIFFUSION_CONFIGS['upsample'])

        print('downloading base checkpoint...')
        base_model.load_state_dict(load_checkpoint(base_name, device))

        print('downloading upsampler checkpoint...')
        upsampler_model.load_state_dict(load_checkpoint('upsample', device))

        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, self.num_pts - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
        )

        # Set a prompt to condition on.
        prompt = str(self.cfg.prompt_processor.prompt)
        print('prompt',prompt)

        # Produce a sample from the model.
        samples = None
        for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])):
            samples = x

        pc = sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
        # import pdb; pdb.set_trace()
        # fig.savefig(self.get_save_path("pointe.png"))
        self.pointefig=fig
        coords = pc.coords # (0.5,-0.5)
        rgb = np.concatenate([pc.channels['R'][:,None],pc.channels['G'][:,None],pc.channels['B'][:,None]],axis=1) # (0,1)
        # rgb = np.array([pc.channels['R'],pc.channels['G'],pc.channels['B']]) # (0,1)
        # import pdb; pdb.set_trace()
        return coords,rgb,0.6
    
    def save_gif_to_file(self,images, output_file):  
        with io.BytesIO() as writer:  
            images[0].save(  
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0  
            )  
            writer.seek(0)  
            with open(output_file, 'wb') as file:  
                file.write(writer.read())
    
    def shape(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device)
        diffusion = diffusion_from_config_shape(load_config('diffusion'))

        batch_size = 1
        guidance_scale = 15.0
        prompt = str(self.cfg.prompt_processor.prompt)
        print('prompt',prompt)

        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        render_mode = 'nerf' # you can change this to 'stf'
        size = 256 # this is the size of the renders; higher values take longer to render.

        cameras = create_pan_cameras(size, device)

        self.shapeimages = decode_latent_images(xm, latents[0], cameras, rendering_mode=render_mode)

        pc = decode_latent_mesh(xm, latents[0]).tri_mesh()

        # import pdb; pdb.set_trace()
        # fig.savefig(self.get_save_path("pointe.png"))
        # self.pointefig=fig
        skip = 4
        coords = pc.verts
        rgb = np.concatenate([pc.vertex_channels['R'][:,None],pc.vertex_channels['G'][:,None],pc.vertex_channels['B'][:,None]],axis=1) 

        coords = coords[::skip]
        rgb = rgb[::skip]

        

        # rgb = np.array([pc.channels['R'],pc.channels['G'],pc.channels['B']]) # (0,1)
        # import pdb; pdb.set_trace()
        return coords,rgb,0.4
    

    def pcb(self):
        # Since this data set has no colmap data, we start with random points

        coords,rgb,scale = self.shape()
        # TODO 2000?
        # xyz shs.freezen()
        # high res magic3d
        # fit time high res
        # single image to sfm points controlnet depth

        
        # We create random points inside the bounds of the synthetic Blender scenes
        # import pdb; pdb.set_trace()

        bound= self.cfg.geometry.radius * scale
        # import pdb; pdb.set_trace()

        # xyz = np.random.random((num_pts, 3)) * bound - np.array([[bound/2.0,bound/2.0, bound/2.0]])

        # xyz = np.random.random((num_pts, 3)) * np.array([[7.0,4.0, 1.0]])

        # xyz = xyz.dot(np.array([[-1,1,1]]))
        # xyz = xyz*np.array([[-1,1,1]])

        # shs = np.random.random((num_pts, 3)) / 255.0 
        # colors = SH2RGB(shs)
        pcd_by3d = o3d.geometry.PointCloud()
        pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
        
        # pcd = o3d.io.read_point_cloud('/data/taoranyi/3dgs/threestudio_3dgs_4shape_sds2/outputs/dreamfusion3dgs-if/a_car@20230929-003953/save/it11-test.ply')

        # 获取点云的边界框
        bbox = pcd_by3d.get_axis_aligned_bounding_box()

        # 在边界框内随机生成点
        num_points = 1000000  # 你想要生成的点的数量
        points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))

        # 创建一个KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)

        # 检查每个点是否在物体内部
        points_inside = []
        color_inside= []
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
            if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                points_inside.append(point)
                color_inside.append(rgb[idx[0]])
        # import pdb; pdb.set_trace()
        # 创建一个新的点云
        # pcd_inside = o3d.geometry.PointCloud()
        # pcd_inside.points = o3d.utility.Vector3dVector(np.array(points_inside))
        # print("np.array(points_inside)",np.array(points_inside).shape)
        # # 保存点云
        # o3d.io.write_point_cloud("filled_point_cloud.ply", pcd_inside)
        all_coords = np.array(points_inside)
        all_rgb = np.array(color_inside)
        all_coords = np.concatenate([all_coords,coords],axis=0)
        all_rgb = np.concatenate([all_rgb,rgb],axis=0)
        # import pdb; pdb.set_trace()
        self.num_pts = all_coords.shape[0]
        print(f"Generating random point cloud ({self.num_pts})...")
        
        # 可视化结果
        # o3d.visualization.draw_geometries([pcd, pcd_inside])
        pcd = BasicPointCloud(points=all_coords *bound, colors=all_rgb, normals=np.zeros((self.num_pts, 3)))
        # import pdb; pdb.set_trace()
        return pcd
    
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # import pdb; pdb.set_trace()

        images = []
        depths = []
        self.viewspace_point_list = []
        for id in range(batch['c2w_3dgs'].shape[0]):
            # import pdb; pdb.set_trace()
            
            viewpoint_cam  = Camera(c2w = batch['c2w_3dgs'][id],FoVy = batch['fovy'][id],height = batch['height'],width = batch['width'])
            # import pdb; pdb.set_trace()

            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, self.background_tensor)
            image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)
            
            # import pdb ; pdb.set_trace()
            # (Pdb) self.viewspace_point_tensor.shape
            # torch.Size([4096, 3])
            # (Pdb) self.visibility_filter.shape # tensor([True, True, True],
            # torch.Size([4096])
            # (Pdb) self.radii.shape
            # torch.Size([4096])
            
            if id == 0:

                self.radii = radii
            else:


                self.radii = torch.max(radii,self.radii)
                
            
            depth = render_pkg["depth_3dgs"]
            depth =  depth.permute(1, 2, 0)
            
            image =  image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)
            

            #             import pdb; pdb.set_trace()
            #             (Pdb) render_pkg['depth'].shape
            # torch.Size([1, 64, 64])
            # (Pdb) render_pkg['depth'].max()
            # tensor(7.6642, device='cuda:0', grad_fn=<MaxBackward1>)
            # (Pdb) render_pkg['depth'].min()
            # tensor(0., device='cuda:0', grad_fn=<MinBackward1>)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        self.visibility_filter = self.radii>0.0
        # self.viewspace_point_tensor.retain_grad()
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.trainstep=0
        # self.trainstep = 0

    def training_step(self, batch, batch_idx):
        # optimizer = self.optimizers()
        self.trainstep = batch_idx
        # print('self.trainstep',self.true_global_step,self.global_step,batch_idx,self.trainstep)
        # import pdb; pdb.set_trace()
        # if self.trainstep % 1000 == 0 and self.trainstep >2000:
        #     self.gaussian.oneupSHdegree()

        
        if self.trainstep > 500:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)
            # self.guidance_zero.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.35)

        if self.trainstep > 800:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.25)


        # if self.trainstep >1500:
        #     self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.25)

        self.gaussian.update_learning_rate(self.trainstep)

        out = self(batch) 
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"].
        # batch dict_keys(['rays_o', 'rays_d', 'mvp_mtx', 'camera_positions', 'c2w', 'light_positions', 'elevation', 'azimuth', 'camera_distances', 'height', 'width'])
        # out dict_keys(['comp_rgb', 'comp_rgb_fg', 'comp_rgb_bg', 'opacity', 'depth', 'z_variance', 'weights', 't_points', 't_intervals', 't_dirs', 'ray_indices', 'points', 'density', 'features', 'normal', 'shading_normal'])
        prompt_utils = self.prompt_processor()

        # if self.trainstep>200:
        #     images = out["comp_rgb"]
        # else:
        #     images = out["opacity"].repeat(1, 1, 1, 3)


        images = out["comp_rgb"]
        
        # import pdb; pdb.set_trace()
        # print('self.trainstep',self.trainstep)

        guidance_eval = (self.trainstep % 100 == 0)
        # guidance_eval = False
        
        guidance_out = self.guidance(
            images, prompt_utils, **batch, rgb_as_latents=False,guidance_eval=guidance_eval
        )
        
        # rgb: Float[Tensor, "B H W C"],
        # prompt_utils: PromptProcessorOutput,
        # elevation: Float[Tensor, "B"],
        # azimuth: Float[Tensor, "B"],
        # camera_distances: Float[Tensor, "B"],
        # rgb_as_latents=False,
        # guidance_eval=False,

        loss = 0.0
        #guidance_out dict_keys(['loss_sds', 'grad_norm', 'min_step', 'max_step'])
        loss = loss + guidance_out['loss_sds'] *self.C(self.cfg.loss['lambda_sds'])
        
        # import pdb; pdb.set_trace()

        # if self.C(self.cfg.loss.lambda_orient) > 0:
        if False:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        # loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        # self.log("train/loss_sparsity", loss_sparsity)
        # loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        # opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        # loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        # self.log("train/loss_opaque", loss_opaque)
        # loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        
        # loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        # self.log("train/loss_sparsity", loss_sparsity)
        # loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        # opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        # loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        # self.log("train/loss_opaque", loss_opaque)
        # loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.manual_backward(loss)

        # if self.trainstep < 300:
        #     self.gaussian._features_dc.requires_grad = False
        #     self.gaussian._features_rest.requires_grad = False  

        with torch.no_grad():
            
            if self.trainstep < 1000: # 15000
                
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                    
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)


                if self.trainstep > 500 and self.trainstep % 100 == 0: # 500 100
                    size_threshold = 20 if self.trainstep > 1000 else None # 3000
                    self.gaussian.densify_and_prune(0.0001 , 0.01, self.cameras_extent, size_threshold) 
    

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none = True)
            
            # optimizer = self.optimizers()
            
            # optimizer.step()
            # optimizer.zero_grad(set_to_none = True)
        # self.global_step =self.global_step +1
        if self.all_train_step<=self.trainstep:
            os.kill(os.getpid(), signal.SIGINT)

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.trainstep}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0],
                        "kwargs": {},
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.trainstep,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.trainstep}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0],
                        "kwargs": {},
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.trainstep,
        )


    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.trainstep}-test",
            f"it{self.trainstep}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.trainstep,
        )
        save_path = self.get_save_path(f"it{self.trainstep}-test.ply")
        self.gaussian.save_ply(save_path)
        # self.pointefig.savefig(self.get_save_path("pointe.png"))
        self.save_gif_to_file(self.shapeimages, self.get_save_path("pointe.gif"))


    def configure_optimizers(self):
        # optim_1 = parse_optimizer(self.cfg.optimizer, self)


    
        return [],[]
    
    def guidance_evaluation_save(self, comp_rgb, guidance_eval_out):
        B, size = comp_rgb.shape[:2]
        resize = lambda x: F.interpolate(
            x.permute(0, 3, 1, 2), (size, size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        filename = f"it{self.trainstep}-train.png"

        def merge12(x):
            return x.reshape(-1, *x.shape[2:])

        self.save_image_grid(
            filename,
            [
                {
                    "type": "rgb",
                    "img": merge12(comp_rgb),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_noisy"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1step"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_1orig"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": merge12(resize(guidance_eval_out["imgs_final"])),
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
            ),
            name="train_step",
            step=self.trainstep,
            texts=guidance_eval_out["texts"],
        )
