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
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh
import io  
from PIL import Image  
import open3d as o3d


def load_ply(path,save_path):
    C0 = 0.28209479177387814
    def SH2RGB(sh):
        return sh * C0 + 0.5
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:,:,0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)

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


@threestudio.register("gaussiandreamer-system")
class GaussianDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        sh_degree: int = 0
        load_type: int = 0
        load_path: str = "./load/shapes/stand.obj"



    cfg: Config
    def configure(self) -> None:
        self.radius = self.cfg.radius
        self.sh_degree =self.cfg.sh_degree
        self.load_type =self.cfg.load_type
        self.load_path = self.cfg.load_path

        self.gaussian = GaussianModel(sh_degree = self.sh_degree)
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        self.background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    
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
        model.load_state_dict(torch.load('./load/shapE_finetuned_with_330kdata.pth', map_location=device)['model_state_dict'])
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


        skip = 1
        coords = pc.verts
        rgb = np.concatenate([pc.vertex_channels['R'][:,None],pc.vertex_channels['G'][:,None],pc.vertex_channels['B'][:,None]],axis=1) 

        coords = coords[::skip]
        rgb = rgb[::skip]

        self.num_pts = coords.shape[0]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(coords)
        point_cloud.colors = o3d.utility.Vector3dVector(rgb)
        self.point_cloud = point_cloud

        return coords,rgb,0.4
    
    def add_points(self,coords,rgb):
        pcd_by3d = o3d.geometry.PointCloud()
        pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
        

        bbox = pcd_by3d.get_axis_aligned_bounding_box()
        np.random.seed(0)

        num_points = 1000000  
        points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))


        kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)


        points_inside = []
        color_inside= []
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
            if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                points_inside.append(point)
                color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

                
                

        all_coords = np.array(points_inside)
        all_rgb = np.array(color_inside)
        all_coords = np.concatenate([all_coords,coords],axis=0)
        all_rgb = np.concatenate([all_rgb,rgb],axis=0)
        return all_coords,all_rgb

    def smpl(self):
        self.num_pts  = 50000
        mesh = o3d.io.read_triangle_mesh(self.load_path)
        point_cloud = mesh.sample_points_uniformly(number_of_points=self.num_pts)
        coords = np.array(point_cloud.points)
        shs = np.random.random((self.num_pts, 3)) / 255.0
        rgb = SH2RGB(shs)
        adjusment = np.zeros_like(coords)
        adjusment[:,0] = coords[:,2]
        adjusment[:,1] = coords[:,0]
        adjusment[:,2] = coords[:,1]
        current_center = np.mean(adjusment, axis=0)
        center_offset = -current_center
        adjusment += center_offset
        return adjusment,rgb,0.5
    
    def pcb(self):
        # Since this data set has no colmap data, we start with random points
        if self.load_type==0:
            coords,rgb,scale = self.shape()
        elif self.load_type==1:
            coords,rgb,scale = self.smpl()
        else:
            raise NotImplementedError
        
        bound= self.radius*scale

        all_coords,all_rgb = self.add_points(coords,rgb)
        

        pcd = BasicPointCloud(points=all_coords *bound, colors=all_rgb, normals=np.zeros((all_coords.shape[0], 3)))

        return pcd
    
    
    def forward(self, batch: Dict[str, Any],renderbackground = None) -> Dict[str, Any]:

        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        self.viewspace_point_list = []
        for id in range(batch['c2w_3dgs'].shape[0]):
       
            viewpoint_cam  = Camera(c2w = batch['c2w_3dgs'][id],FoVy = batch['fovy'][id],height = batch['height'],width = batch['width'])


            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            
            if id == 0:

                self.radii = radii
            else:


                self.radii = torch.max(radii,self.radii)
                
            
            depth = render_pkg["depth_3dgs"]
            depth =  depth.permute(1, 2, 0)
            
            image =  image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)
            



        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        self.visibility_filter = self.radii>0.0
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
    
    def training_step(self, batch, batch_idx):

        self.gaussian.update_learning_rate(self.true_global_step)
        
        if self.true_global_step > 500:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)

        self.gaussian.update_learning_rate(self.true_global_step)

        out = self(batch) 

        prompt_utils = self.prompt_processor()
        images = out["comp_rgb"]


        guidance_eval = (self.true_global_step % 200 == 0)
        # guidance_eval = False
        
        guidance_out = self.guidance(
            images, prompt_utils, **batch, rgb_as_latents=False,guidance_eval=guidance_eval
        )
        

        loss = 0.0

        loss = loss + guidance_out['loss_sds'] *self.C(self.cfg.loss['lambda_sds'])
        



        
        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))




        return {"loss": loss}



    def on_before_optimizer_step(self, optimizer):

        with torch.no_grad():
            
            if self.true_global_step < 900: # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > 300 and self.true_global_step % 100 == 0: # 500 100
                    size_threshold = 20 if self.true_global_step > 500 else None # 3000
                    self.gaussian.densify_and_prune(0.0002 , 0.05, self.cameras_extent, size_threshold) 






    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
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
            ),
            name="validation_step",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
        # self.gaussian.save_ply(save_path)
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if False else [0, 0, 0]

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch,testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
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
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
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
                step=self.true_global_step,
            )


    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_path = self.get_save_path(f"last_3dgs.ply")
        self.gaussian.save_ply(save_path)
        # self.pointefig.savefig(self.get_save_path("pointe.png"))
        if self.load_type==0:
            o3d.io.write_point_cloud(self.get_save_path("shape.ply"), self.point_cloud)
            self.save_gif_to_file(self.shapeimages, self.get_save_path("shape.gif"))
        load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-test-color.ply"))
        


    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
        
        opt = OptimizationParams(self.parser)
        point_cloud = self.pcb()
        self.cameras_extent = 4.0
        self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)

        self.pipe = PipelineParams(self.parser)
        self.gaussian.training_setup(opt)
        
        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret
