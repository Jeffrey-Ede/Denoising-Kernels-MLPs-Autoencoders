# Denoising-Kernels-MLPs-Autoencoders

This repository is in the works. A paper, more code, examples and more neural networks will be published here soon (by around the end of August).

Repository for the paper <- add paper

Denoising kernels, multilayer perceptrons and autoencoders for electron microscopy

Fully trained autoencoders are available [here](https://drive.google.com/open?id=13_xQvTOJqYu4WoEUF5lUiM4hAkYX4yM3).

Fully trained denoising kernels and MLPs are available [here](https://drive.google.com/open?id=1sweKWAL9quwAqIclgfM2d0Ia2-rgmu0Y)

Kerenel MLP files are labelled with with d<x>w<y>. These indicate the depths and widths of the kernels and MLPs the file stores parameters for. depth = hidden layers+1 so d1w3 is a simple kernel and input width 3 whereas d4w5 is a multilayer perceptron with 3 hidden layers and input width 5. 

##Example Usage 
```python
"""
import numpy as np
from apply_autoencoders import Denoiser, disp

#Create a 1000x1000 image from random numbers for demonstration
#Try replacing this with your own image!
img = np.random.rand(1500, 1500)

#Initialize the denoising neural network
noise_remover = Denoiser()

#Denoise a 512x512 crop from the image
crop = img[:512,:512]
denoised_crop = noise_remover.denoise_crop(crop)

#Denoise the entire image
denoised_img = noise_remover.denoise(img)

disp(crop) #Crop before denoising
disp(denoised_crop) #Crop after denoising
disp(img) #Image before denoising
disp(denoised_img) #Image after denoising
"""
'''

Note that this repository is in the works!
```
