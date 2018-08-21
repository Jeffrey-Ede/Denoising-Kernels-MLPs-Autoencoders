import numpy as np
import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
fontsize = 10
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['axes.titlepad'] = 10
mpl.rcParams['savefig.dpi'] = 300

import matplotlib.mlab as mlab

import scipy.stats as stats

import cv2
from scipy.misc import imread

cols = 5
rows = 6

max_cols = 5

# width as measured in inkscape
scale = 1.0
ratio = 1.3 # 1.618
width = scale * 2.2 * 3.487
height = 4.35*(width / 1.618) / 2.2
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

##Make a list of all the example images

#TEM examples
TEM_loc = "G:/noise-removal-kernels-TEM/examples/kernels+mlps/"
TEM_filtered_loc = "G:/noise-removal-kernels-TEM/examples/filtered/"
TEM_orig_locs = ["G:/noise-removal-kernels-TEM/examples/orig/"]
TEM_orig_locs = [TEM_orig_locs[0]+str(i)+".tif" for i in range(1, 6)]
TEM_autoenc_locs = [TEM_filtered_loc+str(i)+'/' for i in [1, 4, 16, 64]]
TEM_autoenc_locs = [[l+str(i)+".tif" for i in range(1, 6)] for l in TEM_autoenc_locs]
TEM_autoenc_locs = [item for sublist in TEM_autoenc_locs for item in sublist]
TEM_kernel_locs = [TEM_loc+loc+'/' for loc in ['d1w3', 'd1w5', 'd1w7', 'd1w11', 'd1w15']]
TEM_kernel_locs = [[l+str(i)+".tif" for i in range(1, 6)] for l in TEM_kernel_locs]
TEM_kernel_locs = [item for sublist in TEM_kernel_locs for item in sublist]
TEM_MLP_locs = [TEM_loc+loc+'/' for loc in ['d2w3', 'd2w5', 'd2w7', 'd3w5', 'd3w7']]
TEM_MLP_locs = [[l+str(i)+".tif" for i in range(1, 6)] for l in TEM_MLP_locs]
TEM_MLP_locs = [item for sublist in TEM_MLP_locs for item in sublist]

#STEM examples
STEM_loc = "G:/noise-removal-kernels-STEM/examples/kernels+mlps/"
STEM_filtered_loc = "G:/noise-removal-kernels-STEM/examples/filtered/"
STEM_orig_locs = ["G:/noise-removal-kernels-STEM/examples/orig/"]
STEM_orig_locs = [STEM_orig_locs[0]+str(i)+".tif" for i in range(1, 6)]
STEM_autoenc_locs = [STEM_filtered_loc+str(i)+'/' for i in [4, 16, 64]]
STEM_autoenc_locs = [[l+str(i)+".tif" for i in range(1, 6)] for l in STEM_autoenc_locs]
STEM_autoenc_locs = [item for sublist in STEM_autoenc_locs for item in sublist]
STEM_kernel_locs = [STEM_loc+loc+'/' for loc in ['d1w3', 'd1w5', 'd1w7', 'd1w11', 'd1w15']]
STEM_kernel_locs = [[l+str(i)+".tif" for i in range(1, 6)] for l in STEM_kernel_locs]
STEM_kernel_locs = [item for sublist in STEM_kernel_locs for item in sublist]
STEM_MLP_locs = [STEM_loc+loc+'/' for loc in ['d2w3', 'd2w5', 'd2w7']]
STEM_MLP_locs = [[l+str(i)+".tif" for i in range(1, 6)] for l in STEM_MLP_locs]
STEM_MLP_locs = [item for sublist in STEM_MLP_locs for item in sublist]

#TEM+STEM examples
TEM_and_STEM_loc = "G:/noise-removal-kernels-TEM+STEM/examples/kernels+mlps/"
TEM_and_STEM_filtered_loc = "G:/noise-removal-kernels-TEM+STEM/examples/filtered/"
TEM_and_STEM_orig_locs = ["G:/noise-removal-kernels-TEM+STEM/examples/orig/"]
TEM_and_STEM_orig_locs = [TEM_and_STEM_orig_locs[0]+str(i)+".tif" for i in range(1, 6)]
TEM_and_STEM_autoenc_locs = [TEM_and_STEM_filtered_loc+str(i)+'/' for i in [2**j for j in range(7)]]
TEM_and_STEM_autoenc_locs = [[l+str(i)+".tif" for i in range(1, 6)] for l in TEM_and_STEM_autoenc_locs]
TEM_and_STEM_autoenc_locs = [item for sublist in TEM_and_STEM_autoenc_locs for item in sublist]
TEM_and_STEM_kernel_locs = [TEM_and_STEM_loc+loc+'/' for loc in ['d1w3', 'd1w5', 'd1w7', 'd1w11', 'd1w15']]
TEM_and_STEM_kernel_locs = [[l+str(i)+".tif" for i in range(1, 6)] for l in TEM_and_STEM_kernel_locs]
TEM_and_STEM_kernel_locs = [item for sublist in TEM_and_STEM_kernel_locs for item in sublist]
TEM_and_STEM_MLP_locs = [TEM_and_STEM_loc+loc+'/' for loc in ['d2w3', 'd2w5', 'd2w7', 'd2w11', 'd3w3', 'd3w7']]
TEM_and_STEM_MLP_locs = [[l+str(i)+".tif" for i in range(1, 6)] for l in TEM_and_STEM_MLP_locs]
TEM_and_STEM_MLP_locs = [item for sublist in TEM_and_STEM_MLP_locs for item in sublist]

#Put data in order
data_locs = TEM_orig_locs+TEM_autoenc_locs+TEM_kernel_locs+TEM_MLP_locs + \
    STEM_orig_locs+STEM_autoenc_locs+STEM_kernel_locs+STEM_MLP_locs + \
    TEM_and_STEM_orig_locs+TEM_and_STEM_autoenc_locs+TEM_and_STEM_kernel_locs+TEM_and_STEM_MLP_locs

#Labels
TEM_labels = ["TEM Originals", "TEM Latent Depth 1", "TEM Latent Depth 4", "TEM Latent Depth 16", "TEM Latent Depth 64",
              "TEM $h$=0, $w$=3", "TEM $h$=0, $w$=5", "TEM $h$=0, $w$=7",
              "TEM $h$=0, $w$=11", "TEM $h$=0, $w$=15", "TEM $h$=1, $w$=3", "TEM $h$=1, $w$=5", "TEM $h$=1, $w$=7", 
              "TEM $h$=2, $w$=5", "TEM $h$=2, $w$=7"]
STEM_labels = ["STEM Originals", "STEM Latent Depth 4", "STEM Latent Depth 16", " Latent Depth 64",
               "STEM $h$=0, $w$=3", "STEM $h$=0, $w$=5", "STEM $h$=0, $w$=7",
               "STEM $h$=0, $w$=11", "STEM $h$=0, $w$=15", "STEM $h$=1, $w$=3", "STEM $h$=1, $w$=5", "STEM $h$=1, $w$=7"]
TEM_and_STEM_labels = ["TEM+STEM Originals", "TEM Latent Depth 1", "TEM Latent Depth 2", "TEM Latent Depth 4",
                       "TEM Latent Depth 8", "TEM Latent Depth 16", "TEM Latent Depth 32", "TEM Latent Depth 64",
                       "TEM+STEM $h$=0, $w$=3", "TEM+STEM $h$=0, $w$=5", "TEM+STEM $h$=0, $w$=7",
                       "TEM+STEM $h$=0, $w$=11", "TEM+STEM $h$=0, $w$=15", "TEM+STEM $h$=1, $w$=3", 
                       "TEM+STEM $h$=1, $w$=5", "TEM+STEM $h$=1, $w$=7", "TEM+STEM $h$=1, $w$=11",
                       "TEM+STEM $h$=2, $w$=3", "TEM+STEM $h$=2, $w$=7"]

#Put labels in order
labels = TEM_labels + STEM_labels + TEM_and_STEM_labels

for l in data_locs:
    print(l)

num_labels = len(labels)
print("Num labels: {}".format(len(labels)))

for set_num in range(8, 9):

    save_loc = "G:/noise-removal-kernels-TEM/image_displays/examples"+str(set_num)+".png"

    #Select subset of data
    start_idx = (set_num-1)*rows*max_cols
    end_idx = start_idx + cols

    label_start_idx = (set_num-1)*rows

    data_locs_to_use = []
    labels_to_use = []
    num_rows = min( set_num*rows, num_labels ) - (set_num-1)*rows
    for i in range(num_rows):
        labels_to_use.append(labels[label_start_idx+i])
        data_locs_to_use += data_locs[(start_idx+i*max_cols):(end_idx+i*max_cols)]

    for l in data_locs_to_use:
        print(l)

    #Load data
    data_to_use = [imread(loc, mode='F') for loc in data_locs_to_use]

    #codes = [(num, 2, x+1) for x in range(2*num)]

    f = plt.figure(1)
    #f, big_axes = plt.subplots( figsize=(15.0, 15.0),nrows=1, ncols=1, sharey=True)

    w = h = 160

    subplot_cropsize = 16
    subplot_prop_of_size = 0.6
    subplot_side = int(subplot_prop_of_size*w)
    subplot_prop_outside = 0.2
    out_len = int(subplot_prop_outside*subplot_side)
    side = w+out_len
    increase_factor = 16
    truesize = int(0.6*w)

    def rescale_discrete(img, cropsize, increase_factor):

        newsize = increase_factor*cropsize

        tmp = img[:cropsize,:cropsize]
        sub = np.zeros((newsize, newsize))
        for y in range(newsize):
            for x in range(newsize):
                sub[y,x] = tmp[y//increase_factor, x//increase_factor]

        return sub

    def plot_data(data, label, pos):

        ax = f.add_subplot(rows,cols,pos)

        img = np.ones(shape=(side,side), dtype=np.float32)
    
        img[:w, :w] = data

        discrete_rescale = rescale_discrete(data, subplot_cropsize, increase_factor)
        img[(side-subplot_side):,(side-subplot_side):] = cv2.resize(discrete_rescale, (truesize, truesize))
    
        img = img.clip(0., 1.)

        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        ax.set_frame_on(False)

        if pos % 5 == 1:
            plt.ylabel(label, rotation=90)

        return

    #Plot the data
    for i, data in enumerate(data_to_use, 1):
        idx = (i-1)//cols
        label = labels_to_use[idx]

        print(label, idx)

        plot_data(data, label, i)

    f.subplots_adjust(wspace=0.07, hspace=-0.0)
    f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)

    f.set_size_inches(width, height)

    #plt.show()

    f.savefig(save_loc, bbox_inches='tight', )

    del f