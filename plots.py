
#%% Modules
import numpy as np
import matplotlib.pyplot as plt
from CreatePsuedoChip import PsuedoChip

# Convolution 
new_chip = PsuedoChip()
(data_conv, x_cent, y_cent) = new_chip.create_chip_rand_centroid_noise()
pixel_plot = plt.figure()
plt.title("pixel_plot")
plt.imshow(data_conv, interpolation='nearest')
plt.plot(x_cent, y_cent, 'o')
plt.colorbar()
plt.show()

# Simulating Exceedence thresholding
new_chip.create_exceedence_chip()
pixel_plot = plt.figure()
plt.title("exceedence_plot")
plt.imshow(new_chip.exceedence_chip, interpolation='nearest')
plt.plot(x_cent, y_cent, 'o')
plt.colorbar()
plt.show()

# %%
