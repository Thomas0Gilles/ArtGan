# ArtGan
Creating Art with Generative Adversarial Networks

Implementations of DC-GANs and WGANs in order to generate Art. Art images from Wikiart. MNIST and CIFAR-10 are also avalaible as toy datasets.
The networks are trained to not only learn the dataset distribution, but also be able to chose the art style generated. 

## Example of generated art
![Batch of Different Art Styles](https://github.com/Thomas0Gilles/ArtGan/blob/master/art_wgan_100.png)

## Mixing of different art style generated

The network is asked to generate 2 art styles at the same time.
![Mixing matrix](https://github.com/Thomas0Gilles/ArtGan/blob/master/mix_art.png)

## To train your own network

Download wikipaintings at www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/wikipaintings_full.tgz.

Update the data path in main_art_wgan.py

Run the command `python main_art_wgan.py`
