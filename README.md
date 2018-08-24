# Denoising-Kernels-MLPs-Autoencoders

Denoising kernels, multilayer perceptrons and autoencoders for electron microscopy.

This repository is in the works. A paper, more code, examples and more neural networks will be published here soon (by around the end of August).

## Example Usage 

Example applications of autoencoders to compression/decompression and noise removal and kernels/multilayer perceptrons to noise removal are provided in the examples. Each of the example scripts demonstrates how a network only needs to be loaded once to be used for inference multiple times.

Example applications of the every autoencoder, kernel and multilayer perceptron are provided as an appendix in the paper.

## Download

To get the training and inference scripts, simply copy the files from or clone this repository:

```
git clone https://github.com/Jeffrey-Ede/Denoising-Kernels-MLPs-Autoencoders.git
cd Denoising-Kernels-MLPs-Autoencoders
```

The last saved checkpoints for our fully trained autoencoders are available [here](https://drive.google.com/open?id=13_xQvTOJqYu4WoEUF5lUiM4hAkYX4yM3).

The last saved checlpoints for our fully trained kernels and multilayer perceptrons are available [here](https://drive.google.com/open?id=1sweKWAL9quwAqIclgfM2d0Ia2-rgmu0Y).

You will see that kernel and multilayer perceptron checkpoint files are labelled with with d<x>w<y>. These indicate the depths and sizes of the kernels and MLPs the file stores parameters for. depth = hidden layers+1 so d1w3 is a simple kernel of input size 3 whereas d4w5 is a multilayer perceptron with 3 hidden layers and input size 5.

## Dependencies

This neural network was trained using TensorFlow and requires it and other common python libraries. Most of these libraries come with modern python distributions by default. If you don't have some of these libraries, they can be installed using pip or another package manager. 

Libraries you need for both training and inference:

* tensorFlow
* numpy
* cv2
* functools
* itertools
* collections
* six
* os

For training you also need:

* argparse
* random
* scipy
* Image

The network was scripted for python 3.6 using Windows 10. Small adjustments may need to be made for other environments or operating systems.

## Training

To continue training the neural networks; either from scratch or to fine-tune the fully trained ones we provide, you will need to adjust some of the variables at the top of the training file. Specifically, variables indicating the location of your datasets and locations to save logs and checkpoints to.
