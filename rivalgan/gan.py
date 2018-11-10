"""  Factory method to create GANs """
from rivalgan.least_square_gan import LeastSquaresGAN
from rivalgan.vanilla_gan import VanillaGAN
from rivalgan.wgan import WassersteinGAN


def create_gan(gan_config):
    """
    Factory method to create GANs
    :param gan_config: configuration of the GAN
    :return: a GAN
   """
    if gan_config.name == "VGAN":
        gan = VanillaGAN(gan_config)

    elif gan_config.name in ["WGAN", "IWGAN"]:
        gan = WassersteinGAN(gan_config)

    elif gan_config.name == "LGAN":
        gan = LeastSquaresGAN(gan_config)

    # TODO: enable AAE
    # elif gan_config.name  == "AAE":
    #     gan = AdversarialAE(gan_name, batches=batch_size, X_nodes=num_features,
    #                         y_nodes=y_output, z_dims=prior_z_dim)

    gan.generator()
    gan.discriminator()

    gan.optimise()
    return gan
