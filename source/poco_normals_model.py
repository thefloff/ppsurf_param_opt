from source.poco_model import PocoModel, PocoNetwork


class PocoNormalsModel(PocoModel):
    """Poco model with normals support (6-channel input)"""

    def __init__(self, output_names, in_channels, out_channels, k,
                 lambda_l1, debug, in_file, results_dir, padding_factor, name, network_latent_size,
                 gen_subsample_manifold_iter, gen_subsample_manifold, gen_resolution_global,
                 rec_batch_size, gen_refine_iter, workers):
        super(PocoNormalsModel, self).__init__(
            output_names=output_names, in_channels=in_channels,
            out_channels=out_channels, k=k,
            lambda_l1=lambda_l1, debug=debug, in_file=in_file, results_dir=results_dir,
            padding_factor=padding_factor, name=name, workers=workers, rec_batch_size=rec_batch_size,
            gen_refine_iter=gen_refine_iter, gen_subsample_manifold=gen_subsample_manifold,
            gen_resolution_global=gen_resolution_global,
            gen_subsample_manifold_iter=gen_subsample_manifold_iter,
            network_latent_size=network_latent_size
        )

        # Replace the network with normals-aware version
        self.network = PocoNormalsNetwork(in_channels=self.in_channels, latent_size=self.network_latent_size,
                                         out_channels=self.out_channels, k=self.k)


class PocoNormalsNetwork(PocoNetwork):
    """Poco network with normals support (6-channel input)"""
    pass  # The FKAConvNetwork already handles variable input channels via in_channels parameter