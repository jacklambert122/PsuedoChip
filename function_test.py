#%% Modules
import numpy as np
import matplotlib.pyplot as plt
from CreatePsuedoChip import PsuedoChip

# %% Picture the continueous gaussian
edge = np.linspace(-0.5, 4.5, 100)
X, Y = np.meshgrid(edge, edge)
Gauss_val = PsuedoChip.gauss_fn(X, Y, 2, 2)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Gauss_val)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Gaussian Contours Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# %% Picture the continueous gaussian in 3d
edge = np.linspace(-0.5, 4.5, 1000)
X, Y = np.meshgrid(edge, edge)
Gauss_val = PsuedoChip.gauss_fn(X, Y, 2, 2)
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Gauss_val, 1000, cmap='binary')
ax.plot_wireframe(X, Y, Gauss_val)
ax.set_title('Gaussian Contours Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# %% Pixel space with value equal to center value from continuous guassian
def create_chip(side_len_sq, centroid_x = None, centroid_y = None):
    side_len = np.linspace(0, side_len_sq-1, side_len_sq)
    X, Y = np.meshgrid(side_len, side_len)

    # Default to random
    if centroid_x is None:
        centroid_x = int(side_len_sq/2)
    if centroid_y is None:
        centroid_y = int(side_len_sq/2)

    return PsuedoChip.gauss_fn(X, Y, centroid_x, centroid_y)

Gauss_val = create_chip(5)
pixel_plot = plt.figure()
plt.title("pixel_plot")
plt.imshow(Gauss_val, interpolation='nearest')
plt.colorbar()
plt.show()

# %% Need to represent a guassian using the intrgated sum within each pixel
side_len_sq = 5
Gauss_val = PsuedoChip.my_trapz(side_len_sq)
pixel_plot = plt.figure()
plt.title("pixel_plot")
plt.imshow(Gauss_val, interpolation='nearest')
plt.colorbar()
plt.show()

#%% Moving centroid in center pixel
side_len_sq = 5
(Gauss_val, x_cent, y_cent) = PsuedoChip.create_chip_rand_centroid(side_len_sq, PsuedoChip.integrate_pixel_count)
pixel_plot = plt.figure()
plt.title("pixel_plot")
plt.imshow(Gauss_val, interpolation='nearest')
plt.plot(x_cent, y_cent, 'o')
plt.colorbar()
plt.show()

# %% Gaussian in discrete pixels
def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    # return kernel / np.sum(kernel) # sum to 1
    return kernel

