import numpy as np
from apply_autoencoders import Micrograph_Autoencoder, disp

#Create a 1000x1000 image from random numbers for demonstration
#Try replacing this with your own image!
img = np.random.rand(1000, 1000)

#Initialize the autoencoder. 
#We are using a feature depth 16 autoencoder in this example
ckpt_loc = "my/checkpoint/location/for/feature/depth/16/"
feature_depth = 16
nn = Micrograph_Autoencoder(ckpt_loc=ckpt_loc, encoding_features=feature_depth)

#Denoise the image
denoised_img = nn.denoise(img)

#Display the original and decompressed image for comparison
disp(img)
disp(denoised_img)

##Repeat to demonstrate that the autoencoder only needs to be loaded once
##to be used for denoising multiple times

#Create a 1000x1000 image from random numbers for demonstration
#Try replacing this with your own image!
img2 = np.random.rand(1000, 1000)

#Denoise the image
denoised_img2 = nn.denoise(img2)

#Display the original and decompressed image for comparison
disp(img2)
disp(denoised_img2)