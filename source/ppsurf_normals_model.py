from source.ppsurf_model import PPSurfModel, PPSurfNetwork, count_parameters


class PPSurfNormalsModel(PPSurfModel):
    """PPSurf model with normals support (6-channel input)"""

    def __init__(self,
                 pointnet_latent_size,
                 output_names, in_channels, out_channels, k,
                 lambda_l1, debug, in_file, results_dir, padding_factor, name, network_latent_size,
                 gen_subsample_manifold_iter, gen_subsample_manifold, gen_resolution_global, num_pts_local,
                 rec_batch_size, gen_refine_iter, workers
                 ):
        super(PPSurfNormalsModel, self).__init__(
            pointnet_latent_size=pointnet_latent_size,
            output_names=output_names, in_channels=in_channels,
            out_channels=out_channels, k=k,
            lambda_l1=lambda_l1, debug=debug, in_file=in_file, results_dir=results_dir,
            padding_factor=padding_factor, name=name, workers=workers, rec_batch_size=rec_batch_size,
            gen_refine_iter=gen_refine_iter, gen_subsample_manifold=gen_subsample_manifold,
            gen_resolution_global=gen_resolution_global,
            gen_subsample_manifold_iter=gen_subsample_manifold_iter,
            network_latent_size=network_latent_size, num_pts_local=num_pts_local
        )

        # Replace the network with normals-aware version
        self.network = PPSurfNormalsNetwork(in_channels=self.in_channels, latent_size=self.network_latent_size,
                                           out_channels=self.out_channels, k=self.k,
                                           num_pts_local=self.num_pts_local,
                                           pointnet_latent_size=self.pointnet_latent_size)


class PPSurfNormalsNetwork(PPSurfNetwork):
    """PPSurf network with normals support (6-channel input)"""

    def __init__(self, in_channels, latent_size, out_channels, k, num_pts_local, pointnet_latent_size):
        # Initialize the parent class but we'll override the point_net
        super().__init__(in_channels, latent_size, out_channels, k, num_pts_local, pointnet_latent_size)
        
        # Replace PointNet with specialized normals version (positions + normals)
        from source.base.nn import PointNetfeatNormals
        self.point_net = PointNetfeatNormals(net_size_max=pointnet_latent_size, num_points=num_pts_local, use_point_stn=False,
                                            use_feat_stn=True, output_size=latent_size, sym_op='att', dim=6)

        print(f'PPSurfNormalsNetwork -- PointNetNormals with 6D input (positions + normals) -- {count_parameters(self.point_net)} parameters')