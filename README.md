# Creating Album Cover Art using Generative Adversarial Networks(GANs)

This project began as my final projecct for the CS231N Class at Stanford but after failing to succeed 
in arriving to satisfactory results I continued to work on it in the weeks following the end of spring quarter.

This project builds utilizes the work of [Erik Linder-Norén](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations)
for implementations of wgan and acgan as well as the work of [Rafael Bidese](https://github.com/rafaelbidese/LOGAN) for implentation of the
LOGAN archirecture.

## Description

The objective of this project is to utilize state of the art GAN architectures on a newer more expansive image dataset in hopes of achieving state of the art performance in generating modern album cover art. In order to accomplish this, this project consists of two builds and a custom dataset. 

#### Dataset

The dataset was built using the Spotify API and the compatibly python library [spotipy](https://spotipy.readthedocs.io/en/2.18.0/) I gathered over 1,000,000 datapoints each consisting of an album name, track name, track image, and set of numerical indicators provided by sspotify known as audio features. The objective was to use these numerical indicators as labels or conditions to implement in the model architecture. Before using the dataset it should be noted that it is not completely clean some datapoints have images with only 3 channels or an odd dimensionality(width/height) so additionally cleaning may be necessary.

#### Model Builds
Both builds are based on a combination the popular [WGAN architecture](https://arxiv.org/pdf/1701.07875.pdf) and the afformentioned [LOGAN architecture](https://arxiv.org/abs/1912.00953) and both make use of Convolutions as introduced in the [DCGAN architecture](https://arxiv.org/abs/1511.06434). The primary difference lies in the fact that on is conditional and one is not. The reasoning behind this is the two final builds are based off of my "broken" build. In order to debug this build I decided to remove the conditional aspect from the GAN and began seeing satisfactory performance giving rise to the non-conditional model. Given the original purpose of the model was to improve accuracy by including audio features as a condition I decided to give it another go but instead this time utilize an [ACGAN architecture](https://arxiv.org/abs/1610.09585) instead of the classical CGAN infastructure. 

#### Results



