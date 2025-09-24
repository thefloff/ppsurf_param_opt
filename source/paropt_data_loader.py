import os
import os.path
import typing

import torch.utils.data as torch_data
import numpy as np
import torch
import trimesh
from overrides import EnforceOverrides, overrides

from source.occupancy_data_module import OccupancyDataModule, get_training_data_dir
from source.base.container import dict_np_to_torch
from .poco_data_loader import get_fkaconv_ids, get_proj_ids


# Adapted from POCO: https://github.com/valeoai/POCO
# which is published under Apache 2.0: https://github.com/valeoai/POCO/blob/main/LICENSE





def get_data_paropt(batch_data: dict, k: int):
    import torch

    fkaconv_data = {
        'pts': torch.transpose(batch_data['pts_ms'], -1, -2),
        'pts_query': torch.transpose(batch_data['pts_query_ms'], -1, -2),
    }

    # if 'imp_surf_dist_ms' in batch_data.keys():
    #     occ_sign = torch.sign(batch_data['imp_surf_dist_ms'])
    #     occ = torch.zeros_like(occ_sign, dtype=torch.int64)
    #     occ[occ_sign > 0.0] = 1
    #     fkaconv_data['occ'] = occ
    # else:
    fkaconv_data['occ'] = torch.zeros(fkaconv_data['pts_query'].shape[:1])

    with torch.no_grad():
        net_data = get_fkaconv_ids(fkaconv_data)
        proj_data = get_proj_ids(fkaconv_data, k)  # TODO: put k in param
        net_data['proj_ids'] = proj_data['proj_ids']

    # need points also for poco ids
    for k in fkaconv_data.keys():
        batch_data[k] = fkaconv_data[k]
    for k in net_data.keys():
        batch_data[k] = net_data[k]

    return batch_data


class ParoptDataModule(OccupancyDataModule):

    def __init__(self, in_file, workers, use_ddp, padding_factor, seed, manifold_points,
                 patches_per_shape: typing.Optional[int], do_data_augmentation: bool, batch_size: int, split: str = "train"):
        super(ParoptDataModule, self).__init__(
            use_ddp=use_ddp, workers=workers, in_file=in_file, patches_per_shape=patches_per_shape,
            do_data_augmentation=do_data_augmentation, batch_size=batch_size)
        self.in_file = in_file
        self.padding_factor = padding_factor
        self.seed = seed
        self.manifold_points = manifold_points
        self.patches_per_shape = patches_per_shape
        self.do_data_augmentation = do_data_augmentation
        self.split = split

    def make_dataset(
            self, in_file: typing.Union[str, list], reconstruction: bool, patches_per_shape: typing.Optional[int],
            do_data_augmentation: bool):
        return ParoptDataset(split = self.split, k = 10000)


class ParoptDataset(torch_data.Dataset, EnforceOverrides):
    def __init__(self, split: str = "train", k: int = 64):
        root = "datasets/abc_normals"
        self.npoints = 1000
        self.root = root
        self.split = split
        self.fns = []
        self.test_every = 1 / 0.1
        self.k = k

        self.gt_dir = os.path.join(root, "06_opt_params")
        self.pts_dir = os.path.join(root, "04_pts_vis")

        self.cloud_data = []

        count = 0
        for gt in os.listdir(self.gt_dir):
            if (self.split == "test" and count%self.test_every == 0) or (self.split != "test" and count%self.test_every != 0):
                gt_data = {}
                with open(os.path.join(self.gt_dir, gt)) as file:
                    for line in file:
                        ls = line.strip().split(sep=': ')
                        if ls[1] == "False":
                            gt_data[ls[0]] = False
                        elif ls[1] == "True":
                            gt_data[ls[0]] = True
                        else:
                            gt_data[ls[0]] = float(ls[1])
                pts_path = os.path.join(self.pts_dir, os.path.splitext(gt)[0] + ".ply")
                self.cloud_data.append([pts_path, gt_data, os.path.splitext(gt)[0]])
            count += 1

    def __getitem__(self, index):
        # index = 0
        cloud = self.cloud_data[index]
        # pts = np.load(cloud[0]).astype(np.float32)
        pts = trimesh.load(cloud[0]).vertices.astype(np.float32)

        #resample
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        pts_ms = pts[choice, :]

        gt = [
            cloud[1]["cgDepth"], 
            (cloud[1]["depth"] - 4) / 4,
            (cloud[1]["fullDepth"] - 5) / 3,
            (cloud[1]["iters"] - 6) / 4,
            (cloud[1]["pointWeight"] - 2) / 6,
            (cloud[1]["samplesPerNode"] - 1) / 8,
            (cloud[1]["scale"] - 0.9) / 0.8,
        ]

        pts = torch.from_numpy(pts)
        pts_ms = torch.from_numpy(pts_ms)
        gt = torch.from_numpy(np.array(gt).astype(np.float32))
        pts_query_ms = torch.from_numpy(np.array([[0, 0, 0]]).astype(np.float32))
        shape_data = {'pts': pts, 'pts_ms': pts_ms, 'pts_query_ms': pts_query_ms, 'gt': gt, 'id': cloud[0]}
        shape_data = get_data_paropt(shape_data, self.k)
        return shape_data
    
    # pts_ms 10000 subsample
    # pts_query_ms 1 point [0,0,0] -> deactivate decoder? (segmentation aus)

    def un_norm(cfg):
        cfg[1] = cfg[1] * 4 + 4
        cfg[2] = cfg[2] * 3 + 5
        cfg[3] = cfg[3] * 4 + 6
        cfg[4] = cfg[4] * 6 + 2
        cfg[5] = cfg[5] * 8 + 1
        cfg[6] = cfg[6] * 0.8 + 0.9
        return cfg

    def __len__(self):
        return len(self.cloud_data)
