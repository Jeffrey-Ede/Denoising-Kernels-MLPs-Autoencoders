from apply_autoencoders import Micrograph_Autoencoder

from scipy.misc import imread
from PIL import Image
import os
import numpy as np

cropsize = 20

ckpt_loc = 'G:/noise-removal-kernels-TEM+STEM/autoencoder/16/model/'
nn = Micrograph_Autoencoder(checkpoint_loc=ckpt_loc,
                            visible_cuda='1',
                            encoding_features=16)

data_loc1 = "G:/unaltered_TEM_crops-171x171/"
data_loc2 = "G:/unaltered_STEM_crops-171x171/"
save_loc0 = "G:/noise-removal-kernels-TEM+STEM/data/orig/"
save_loc = "G:/noise-removal-kernels-TEM+STEM/data/16/"

files = ([data_loc1+f for f in os.listdir(data_loc1)[:6077//2]] + 
         [data_loc1+f for f in os.listdir(data_loc2)[:(6077//2+6077%2)]])
num_files = len(files)
print("Num files: {}".format(num_files))

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

counter = 1
for k, file in enumerate(files):
    print("Train file {} of {}".format(k, num_files))

    try:
        img = imread(file, mode="F")
        img = img[:160, :160]
        nn_img = nn.denoise_crop(img)

        c = np.min(img)
        m = np.mean(img)-c
        img = (img - c) / m

        c = np.min(nn_img)
        m = np.mean(nn_img)-c
        nn_img = (nn_img - c) / m

        if img.shape[0] >= cropsize and img.shape[1] >= cropsize:
            
            #for i in range(0, img.shape[0]-cropsize+1, cropsize):
            #    for j in range(0, img.shape[1]-cropsize+1, cropsize):

            i = np.random.randint(20, 160-20-20)
            j = np.random.randint(20, 160-20-20)

            Image.fromarray(nn_img[i:(i+cropsize), j:(j+cropsize)]).save( save_loc+str(counter)+".tif" )
            Image.fromarray(img[i:(i+cropsize), j:(j+cropsize)]).save( save_loc0+str(counter)+".tif" )
            counter += 1
    except:
        print('error')

    if counter >= 2*6077:
        break