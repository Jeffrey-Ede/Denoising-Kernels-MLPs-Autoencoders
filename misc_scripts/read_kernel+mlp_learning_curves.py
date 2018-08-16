import re
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""For reading restoration kernel and MLP learning curves"""

take_ln = True
moving_avg = True
save = True
window_size = 500
dataset_num = 4
num_losses = 2
type = 'TEM+STEM'
log_loc = "G:/noise-removal-kernels-"+type+"/results/"+str(dataset_num)+"/"
#log_loc = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-13-general/"
log_file = log_loc+"log.txt"

switch = False
losses = [[] for _ in range(num_losses)]
losses_iters = []
with open(log_file, "r") as f:
    for line in f:
        numbers = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
        #numbers = numbers[len(numbers)-40000:]
        #numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        print(len(numbers))

        for count, i in enumerate(range(0, len(numbers)-num_losses-1, num_losses+1)):
            
            losses_iters.append(float(numbers[count]))
            for n in range(num_losses):
                losses[n].append(float(numbers[i+n+1]))

            if not count%1000:
                print(count, losses[n][count])

def moving_average(a, n=window_size):
    ret = np.cumsum(np.insert(a,0,0), dtype=float)
    return (ret[n:] - ret[:-n]) / float(n)

losses_iters = moving_average(np.array(losses_iters[:])) if moving_avg else np.array(losses_iters[:])
for n in range(num_losses):
    losses[n] = moving_average(np.array(losses[n][:])) if moving_avg else np.array(losses[n][:])

#print(losses[0][7000:7030])

if save:
    for n in range(num_losses):
        save_loc = (log_loc+"log"+str(n+1)+".npy")
        np.save(save_loc, losses[n])

#Show the first loss log as an example
#plt.plot(losses_iters, np.log(losses[0]) if take_ln else losses[0])
plt.plot(np.log(losses[1]) if take_ln else losses[1])
plt.show()


