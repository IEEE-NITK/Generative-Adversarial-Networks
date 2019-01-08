# Generative Adversarial Networks

The aim of the project was to learn about Generative Adversarial Networks and explore the various types of GAN objectives and architectures. We also studied the problem of Multi-Domain image to image translation, and implemented StarGAN to perform the same on the celebA dataset.


## Structure
* `learning-phase` contains the models implemented during the learning phase of the project. This includes:
    * Convolutional Neural Network for classification of MNIST dataset.
    * [Vanilla GAN](https://arxiv.org/abs/1406.2661)
    * [DCGAN](https://arxiv.org/abs/1511.06434)
* `archived` contains a Tensorflow implementation of StarGAN that was archived due to the issues with the data loader.
* `StarGAN` contains a Tensorflow implementation of [StarGAN](https://arxiv.org/abs/1711.09020) for multi-domain image to image translation on the [celebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).


## Requirements
* Tensorflow

## Team
* Moksh Jain
* Mahim Agrawal
* Mahir Jain
* Palak Singhal
