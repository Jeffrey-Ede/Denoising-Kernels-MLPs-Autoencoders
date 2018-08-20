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
mpl.rcParams['savefig.dpi'] = 600

import matplotlib.mlab as mlab

import scipy.stats as stats

# width as measured in inkscape
scale = 1.0
ratio = 1.3 # 1.618
width = scale * 2.2 * 3.487
height = 1.7*2.2*(width / ratio) / 2.2
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

#Graph labels
TEM_kernel_labels = ["Size 3", "Size 5", "Size 7", "Size 11", "Size 15"]
STEM_kernel_labels = ["Size 3", "Size 5", "Size 7", "Size 11", "Size 15"]
TEM_and_STEM_kernel_labels = ["Size 3", "Size 5", "Size 7", "Size 11", "Size 15"]

TEM_MLP_labels = ["1 Hidden Layer, Size 3", "1 Hidden Layer, Size 5", "1 Hidden Layer, Size 7",
                  "2 Hidden Layers, Size 5", "2 Hidden Layers, Size 7"]
STEM_MLP_labels = ["1 Hidden Layer, Size 3", "1 Hidden Layer, Size 5", "1 Hidden Layer, Size 7"]
TEM_and_STEM_MLP_labels = ["1 Hidden Layer, Size 3", "1 Hidden Layer, Size 5",
                           "1 Hidden Layer, Size 7", "1 Hidden Layer, Size 11",
                           "2 Hidden Layers, Size 3", "2 Hidden Layers, Size 7"]

#Numbers of files
TEM_kernel_num = len(TEM_kernel_labels)
STEM_kernel_num = len(STEM_kernel_labels)
TEM_and_STEM_kernel_num = len(TEM_and_STEM_kernel_labels)

TEM_MLP_num = len(TEM_MLP_labels)
STEM_MLP_num = len(STEM_MLP_labels)
TEM_and_STEM_MLP_num = len(TEM_and_STEM_MLP_labels)

#Data locations
TEM_kernel_data_files = [r"G:/noise-removal-kernels-TEM/results/1/log"+str(i)+".npy" for i in range(1, 6)]
STEM_kernel_data_files = [r"G:/noise-removal-kernels-STEM/results/1/log"+str(i)+".npy" for i in range(1, 6)]
TEM_and_STEM_kernel_data_files = [r"G:/noise-removal-kernels-TEM+STEM/results/1/log"+str(i)+".npy" for i in range(1, 6)]

TEM_MLP_data_files = [r"G:/noise-removal-kernels-TEM/results/2/log"+str(i)+".npy" for i in range(1, 4)] + \
    [r"G:/noise-removal-kernels-TEM/results/3/log"+str(i)+".npy" for i in range(1, 3)]
STEM_MLP_data_files = [r"G:/noise-removal-kernels-STEM/results/2/log"+str(i)+".npy" for i in range(1, 4)]
TEM_and_STEM_MLP_data_files = [r"G:/noise-removal-kernels-TEM+STEM/results/2/log"+str(i)+".npy" for i in range(1, 5)] + \
    [r"G:/noise-removal-kernels-TEM+STEM/results/3/log"+str(i)+".npy" for i in range(1, 4) if i != 2]

#Load data
TEM_kernel_data = [np.load(f) for f in TEM_kernel_data_files]
STEM_kernel_data = [np.load(f) for f in STEM_kernel_data_files]
TEM_and_STEM_kernel_data = [np.load(f) for f in TEM_and_STEM_kernel_data_files]

TEM_MLP_data = [np.load(f) for f in TEM_MLP_data_files]
STEM_MLP_data = [np.load(f) for f in STEM_MLP_data_files]
TEM_and_STEM_MLP_data = [np.load(f) for f in TEM_and_STEM_MLP_data_files]

#codes = [(num, 2, x+1) for x in range(2*num)]

f, big_axes = plt.subplots( figsize=(15.0, 15.0),nrows=3, ncols=1, sharey=True)

#titles = ["Kernels (0 hidden layers)", "Multilayer Perceptrons (> 0 hidden layers)"]
titles = ["TEM", "STEM", "TEM+STEM"]
for row, big_ax in enumerate(big_axes):
    big_ax.set_title(titles[row], fontsize=fontsize)

    # Turn off axis lines and ticks of the big subplot 
    # obs alpha is 0 in RGBA string!
    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top=False, bottom=False, left=False, right=False)
    # removes the white frame
    big_ax._frameon = False

def plot_data(data, data_labels, pos):

    pos = 2*pos-1 if pos <= 3 else 2*(pos-3)
    ax = f.add_subplot(3,2,pos)

    for i, data in enumerate(data):
        xpoints = np.linspace(1, data.size, data.size)
        plt.plot(xpoints, np.log10(data), linewidth=1., label=data_labels[i])

    plt.xlabel('Batches $\\times$10$^3$')
    plt.ylabel('Log$_{{10}}$(MSE)')
    plt.minorticks_on()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    plt.legend(loc='upper right', frameon=False)

    return

#Plot the data
plot_data(TEM_kernel_data, TEM_kernel_labels, 1)
plot_data(STEM_kernel_data, STEM_kernel_labels, 2)
plot_data(TEM_and_STEM_kernel_data, TEM_and_STEM_kernel_labels, 3)

plot_data(TEM_MLP_data, TEM_MLP_labels, 4)
plot_data(STEM_MLP_data, STEM_MLP_labels, 5)
plot_data(TEM_and_STEM_MLP_data, TEM_and_STEM_MLP_labels, 6)


f.subplots_adjust(wspace=0.18, hspace=0.22)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)

f.set_size_inches(width, height)

#plt.show()

f.savefig('kernel+MLP_learning_curves.png', bbox_inches='tight', )


