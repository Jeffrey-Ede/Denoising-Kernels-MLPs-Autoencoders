import numpy as np
from apply_autoencoders import Micrograph_Autoencoder, disp

#Create a 1000x1000 image from random numbers for demonstration
#Try replacing this with your own image!
img = np.random.rand(1000, 1000)

#Load kernel or multilayer perceptron. 
#We are using a size 5 multilayer perceptron with 1 hidden layer in this example
ckpt_loc = "my/checkpoint/location/for/size/5/with/1/hidden/layer"
feature_depth = 16
denoiser = Kernels_and_MLPS(ckpt_loc=ckpt_loc,
                            depth=2, #Depth is the number of hidden layers + 1
                            width=5)

#Denoise the image
denoised_img = denoiser.denoise(img)

#Display the original and decompressed image for comparison
disp(img)
disp(denoised_img)

##Repeat to demonstrate that the neural network only needs to be loaded once
##to be used for denoising multiple times

#Create a 1000x1000 image from random numbers for demonstration
#Try replacing this with your own image!
img2 = np.random.rand(1000, 1000)

#Denoise the image
denoised_img2 = denoiser.denoise(img2)

#Display the original and decompressed image for comparison
disp(img2)
disp(denoised_img2)
