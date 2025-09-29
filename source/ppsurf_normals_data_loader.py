import typing

import numpy as np
from overrides import overrides

from source.ppsurf_data_loader import PPSurfDataModule, PPSurfDataset
from source.poco_normals_data_loader import PocoNormalsDataset, PocoNormalsReconstructionDataset
from source.poco_data_loader import get_data_poco
from source.base.container import dict_np_to_torch
from source.base.proximity import query_kdtree


class PPSurfNormalsDataModule(PPSurfDataModule):
    """PPSurf data module with normals support (6-channel input)"""
    
    def make_dataset(
            self, in_file: typing.Union[str, list], reconstruction: bool, patches_per_shape: typing.Optional[int],
            do_data_augmentation: bool):

        if reconstruction:
            dataset = PPSurfNormalsReconstructionDataset(
                in_file=in_file,
                num_pts_local=self.num_pts_local,
                padding_factor=self.padding_factor,
                seed=self.seed,
                use_ddp=self.use_ddp,
            )
        else:
            dataset = PPSurfNormalsDataset(
                in_file=in_file,
                num_pts_local=self.num_pts_local,
                padding_factor=self.padding_factor,
                seed=self.seed,
                patches_per_shape=self.patches_per_shape,
                do_data_augmentation=do_data_augmentation,
                use_ddp=self.use_ddp,
                manifold_points=self.manifold_points,
            )
        return dataset


class PPSurfNormalsDataset(PocoNormalsDataset):
    """PPSurf dataset with normals support (6-channel input)"""

    def __init__(self, in_file, num_pts_local, padding_factor, seed, use_ddp,
                 manifold_points, patches_per_shape: typing.Optional[int], do_data_augmentation=True):
        super(PPSurfNormalsDataset, self).__init__(
            in_file=in_file, padding_factor=padding_factor, seed=seed,
            use_ddp=use_ddp, manifold_points=manifold_points,
            patches_per_shape=patches_per_shape, do_data_augmentation=do_data_augmentation)

        self.num_pts_local = num_pts_local

    def __getitem__(self, shape_id):
        shape_data, pts_ms_raw = self.load_shape_by_index(shape_id, return_kdtree=True)
        kdtree = shape_data.pop('kdtree')

        if self.do_data_augmentation:
            import trimesh
            rand_rot = trimesh.transformations.random_rotation_matrix(self.rng.rand(3))
            shape_data = self.augment_shape(shape_data, rand_rot)

        # must be after augmentation
        shape_data = PPSurfNormalsDataset.get_local_subsamples(shape_data, kdtree, pts_ms_raw, self.num_pts_local)

        # Include normals in patch normalization - expect normals to be available
        pts_local_ps = self.normalize_patches_with_normals(
            pts_local_ms=shape_data['pts_local_ms'], 
            pts_query_ms=shape_data['pts_query_ms'],
            normals_local_ms=shape_data['normals_local_ms']
        )

        shape_data['pts_local_ps'] = pts_local_ps
        shape_data = dict_np_to_torch(shape_data)  # must be before poco part
        shape_data = get_data_poco(shape_data)
        return shape_data

    @staticmethod
    def get_local_subsamples(shape_data, kdtree, pts_raw_ms, num_pts_local):
        _, patch_pts_ids = query_kdtree(kdtree=kdtree, pts_query=shape_data['pts_query_ms'],
                                        k=num_pts_local, sqr_dists=True)
        patch_pts_ids = patch_pts_ids.astype(np.int64)
        shape_data['pts_local_ms'] = pts_raw_ms[patch_pts_ids]
        
        # Expect normals to be available - let it error if missing
        shape_data['normals_local_ms'] = shape_data['normals_ms'][patch_pts_ids]
        
        return shape_data
    
    @staticmethod
    def normalize_patches_with_normals(pts_local_ms, pts_query_ms, normals_local_ms):
        """Normalize patches and concatenate with normals"""
        patch_radius_ms = PPSurfDataset.get_patch_radii(pts_local_ms, pts_query_ms)
        pts_local_ps = PPSurfDataset.model_space_to_patch_space(
            pts_to_convert_ms=pts_local_ms, pts_patch_center_ms=pts_query_ms,
            patch_radius_ms=patch_radius_ms)
        
        # Normals don't need translation, only rotation (which is already applied during augmentation)
        # Concatenate positions and normals along the feature dimension
        pts_with_normals_ps = np.concatenate([pts_local_ps, normals_local_ms], axis=-1)
        return pts_with_normals_ps


class PPSurfNormalsReconstructionDataset(PocoNormalsReconstructionDataset):
    """PPSurf reconstruction dataset with normals support"""

    def __init__(self, in_file, num_pts_local, padding_factor, seed, use_ddp):
        super(PPSurfNormalsReconstructionDataset, self).__init__(
            in_file=in_file, padding_factor=padding_factor, seed=seed, use_ddp=use_ddp)
        
        self.num_pts_local = num_pts_local

    @overrides
    def __getitem__(self, shape_id):
        shape_data, pts_ms_raw = self.load_shape_by_index(shape_id, return_kdtree=False)
        shape_data['pts_raw_ms'] = pts_ms_raw  # collate issue for batch size > 1
        shape_data = dict_np_to_torch(shape_data)
        return shape_data