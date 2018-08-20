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
width = scale * 2.2 * 3.487 / 2.4
height = 3.7*2.2*(width / ratio) / 2.2
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

#Graph labels
TEM_labels = ["Latent Depth 1", "Latent Depth 4", "Latent Depth 16", "Latent Depth 64"]
STEM_labels = ["Latent Depth 4", "Latent Depth 16", "Latent Depth 64"]
TEM_and_STEM_labels = ["Latent Depth 1", "Latent Depth 2", "Latent Depth 4", 
                       "Latent Depth 8", "Latent Depth 16", "Latent Depth 32",
                       "Latent Depth 64"]

#Data locations
TEM_autoenc_data_files = ["G:/noise-removal-kernels-TEM/autoencoder/"+str(i)+"/log.npy" for i in [1, 4, 16, 64]]
STEM_autoenc_data_files = ["G:/noise-removal-kernels-STEM/autoencoder/"+str(i)+"/log.npy" for i in [4, 16, 64]]
TEM_and_STEM_autoenc_data_files = ["G:/noise-removal-kernels-TEM+STEM/autoencoder/"+str(i)+"/log.npy" for i in 
                                   [1, 2, 4, 8, 16, 32, 64]]

#Load data
TEM_data = [np.load(f) for f in TEM_autoenc_data_files]
STEM_data = [np.load(f) for f in STEM_autoenc_data_files]
TEM_and_STEM_data = [np.load(f) for f in TEM_and_STEM_autoenc_data_files]
#codes = [(num, 2, x+1) for x in range(2*num)]

f = plt.figure(1)
#f, big_axes = plt.subplots( figsize=(15.0, 15.0),nrows=1, ncols=1, sharey=True)

#titles = ["Kernels (0 hidden layers)", "Multilayer Perceptrons (> 0 hidden layers)"]
titles = ["TEM", "STEM", "TEM+STEM"]
#for row, big_ax in enumerate(big_axes):
#    #big_ax.set_title(titles[row], fontsize=fontsize)

#    # Turn off axis lines and ticks of the big subplot 
#    # obs alpha is 0 in RGBA string!
#    big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top=False, bottom=False, left=False, right=False)
#    # removes the white frame
#    big_ax._frameon = False

def plot_data(data, data_labels, pos):

    ax = f.add_subplot(3,1,pos)

    for i, data in enumerate(data):
        xpoints = np.linspace(1, data.size, data.size)
        plt.plot(xpoints, np.log10(data), linewidth=1., label=data_labels[i])

    plt.xlabel('Batches $\\times$10$^3$')
    plt.ylabel('Log$_{{10}}$(MSE)')
    #plt.title(titles[pos-1], fontsize=10)
    plt.minorticks_on()
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.xticks(np.arange(0, 60000, 15000))

    ax.text(.5,.9,titles[pos-1],
        horizontalalignment='center',
        transform=ax.transAxes)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(0.3, -0.35))
    ax.legend(loc='lower left', framealpha=0.8)
    #plt.legend(loc='upper right', frameon=False)

    return

#Plot the data
plot_data(TEM_data, TEM_labels, 1)
plot_data(STEM_data, STEM_labels, 2)
plot_data(TEM_and_STEM_data, TEM_and_STEM_labels, 3)

f.subplots_adjust(wspace=0.25, hspace=0.19)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)

f.set_size_inches(width, height)

#plt.show()

f.savefig('autoenc_learning_curves.png', bbox_inches='tight', )


