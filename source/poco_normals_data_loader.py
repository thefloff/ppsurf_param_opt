import typing

import numpy as np
import trimesh.transformations as trafo

from source.poco_data_loader import PocoDataModule, PocoDataset, PocoReconstructionDataset, get_data_poco
from source.base.container import dict_np_to_torch


class PocoNormalsDataModule(PocoDataModule):
    """Poco data module with normals support (6-channel input)"""
    
    def make_dataset(
            self, in_file: typing.Union[str, list], reconstruction: bool, patches_per_shape: typing.Optional[int],
            do_data_augmentation: bool):

        if reconstruction:
            dataset = PocoNormalsReconstructionDataset(
                in_file=in_file,
                padding_factor=self.padding_factor,
                seed=self.seed,
                use_ddp=self.use_ddp,
            )
        else:
            dataset = PocoNormalsDataset(
                in_file=in_file,
                padding_factor=self.padding_factor,
                seed=self.seed,
                use_ddp=self.use_ddp,
                manifold_points=self.manifold_points,
                patches_per_shape=self.patches_per_shape,
                do_data_augmentation=do_data_augmentation,
            )
        return dataset
    



class PocoNormalsDataset(PocoDataset):
    """Poco dataset with normals support (6-channel input)"""
    
    def augment_shape(self, shape_data: dict, rand_rot: np.ndarray) -> dict:
        """Augment shape data including both positions and normals"""
        def rot_arr(arr, rot):
            return trafo.transform_points(arr, rot).astype(np.float32)

        # Call parent method first to handle standard points
        shape_data = super().augment_shape(shape_data, rand_rot)
        
        # Then handle normals (which parent doesn't know about)
        if 'normals_ms' in shape_data:
            shape_data['normals_ms'] = rot_arr(shape_data['normals_ms'], rand_rot)
        
        return shape_data
    
    def load_shape_by_index(self, shape_ind, return_kdtree=True):
        """Override to handle normals during subsampling"""
        # Call parent to get basic shape loading
        shape_data, pts_ms_raw = super().load_shape_by_index(shape_ind, return_kdtree)
        
        # Override the subsampling to handle normals
        def sub_sample_point_cloud_with_normals(pts: np.ndarray, normals: np.ndarray, num_target_pts: typing.Optional[int]):
            if num_target_pts is None:
                return pts, normals
            replace = True if pts.shape[0] < num_target_pts else False
            choice_ids = self.rng.choice(np.arange(pts.shape[0]), size=num_target_pts, replace=replace)
            return pts[choice_ids], normals[choice_ids]

        # Apply normals-aware subsampling
        if 'normals_ms' in shape_data:
            pts_sub_sample, normals_sub_sample = sub_sample_point_cloud_with_normals(
                pts=shape_data['pts_ms'], 
                normals=shape_data['normals_ms'], 
                num_target_pts=self.manifold_points
            )
            shape_data['pts_ms'] = pts_sub_sample
            shape_data['normals_ms'] = normals_sub_sample
        
        return shape_data, pts_ms_raw
    
    def __getitem__(self, shape_id):
        shape_data, pts_ms_raw = self.load_shape_by_index(shape_id, return_kdtree=False)

        if self.do_data_augmentation:
            import trimesh
            rand_rot = trimesh.transformations.random_rotation_matrix(self.rng.rand(3))
            shape_data = self.augment_shape(shape_data, rand_rot)

        # Expect normals to be available - concatenate positions and normals
        pts_with_normals = np.concatenate([shape_data['pts_ms'], shape_data['normals_ms']], axis=-1)
        shape_data['pts_ms'] = pts_with_normals  # Replace with 6-channel data
        
        shape_data = dict_np_to_torch(shape_data)
        shape_data = get_data_poco(shape_data)
        return shape_data


class PocoNormalsReconstructionDataset(PocoReconstructionDataset):
    """Poco reconstruction dataset with normals support"""
    
    def load_shape_by_index(self, shape_ind, return_kdtree=True):
        """Override to handle normals during subsampling"""
        # Call parent to get basic shape loading
        shape_data, pts_ms_raw = super().load_shape_by_index(shape_ind, return_kdtree)
        
        # Override the subsampling to handle normals
        def sub_sample_point_cloud_with_normals(pts: np.ndarray, normals: np.ndarray, num_target_pts: typing.Optional[int]):
            if num_target_pts is None:
                return pts, normals
            replace = True if pts.shape[0] < num_target_pts else False
            choice_ids = self.rng.choice(np.arange(pts.shape[0]), size=num_target_pts, replace=replace)
            return pts[choice_ids], normals[choice_ids]

        # Apply normals-aware subsampling
        if 'normals_ms' in shape_data:
            pts_sub_sample, normals_sub_sample = sub_sample_point_cloud_with_normals(
                pts=shape_data['pts_ms'], 
                normals=shape_data['normals_ms'], 
                num_target_pts=self.manifold_points
            )
            shape_data['pts_ms'] = pts_sub_sample
            shape_data['normals_ms'] = normals_sub_sample
        
        return shape_data, pts_ms_raw
    
    def __getitem__(self, shape_id):
        shape_data, pts_ms_raw = self.load_shape_by_index(shape_id, return_kdtree=False)
        
        # Expect normals to be available - concatenate positions and normals
        pts_with_normals = np.concatenate([shape_data['pts_ms'], shape_data['normals_ms']], axis=-1)
        shape_data['pts_ms'] = pts_with_normals  # Replace with 6-channel data
        
        shape_data = dict_np_to_torch(shape_data)
        return shape_data