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

#Compress the image
compressed_img = nn.compress(img)
print("Image is {}x its original size", compressed_img.size/img.size)

#Decompress the image
decompressed_img = nn.decompress(compressed_img)

#Display the original and decompressed images for comparison
disp(img)
disp(decompressed_img)

##Repeat to demonstrate that the autoencoder only needs to be loaded once
##to be used for compression multiple times

#Create a 1000x1000 image from random numbers for demonstration
#Try replacing this with your own image!
img2 = np.random.rand(1000, 1000)

#Compress the image
compressed_img2 = nn.compress(img2)
print("Image 2 is {}x its original size", compressed_img.size/img.size)

#Decompress the image
decompressed_img2 = nn.decompress(compressed_img2)

#Display the original and decompressed images for comparison
disp(img2)
disp(decompressed_img2)