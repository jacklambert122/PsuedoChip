#%% Modules
import numpy as np
from scipy import integrate

############################################################################################
# %% Class for chip with gaussian pixel distribution + random noise (w/ normal max level)
class PsuedoChip:
    def __init__(self):

        # Matrix size
        self.sq_matrix_len = 5

        # Gauss peak
        self.peak_max = 20
        self.gauss_peak = np.random.randint(1, self.peak_max)

        # Noise params
        noise_mean = 0.1    # In percent of total relative to gauss dist.
        noise_sigma = 0.03
        self.num_bins = 1000
        self.noise_level = np.random.normal(noise_mean, noise_sigma, self.num_bins)

        # Method to create chip 
        # self.fn=self.my_trapz
        self.fn=self.double_quadrature

        # Exceedance thresholding
        self.count_lim = 4

        # Pixel Data
        self.data_conv = np.zeros((self.sq_matrix_len, self.sq_matrix_len))
        self.data_rand = np.zeros((self.sq_matrix_len, self.sq_matrix_len))
        self.data_gauss = np.zeros((self.sq_matrix_len, self.sq_matrix_len))
        self.exceedence_chip = np.zeros((self.sq_matrix_len, self.sq_matrix_len))


    @staticmethod
    def gauss_fn(x, y, xc=0, yc=0, scalar=1, sig=1):
        return ( 1 /  (2 * np.pi * sig**2) )  * np.exp( -((x-xc)**2 + (y-yc)**2) / (2 * sig**2) )

    @staticmethod
    def double_Integral(xmin, xmax, ymin, ymax, nx, ny, A):
        '''
        Integrate surface using trapizoidal method
        '''
        # area of each sub unit
        dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))
        # internal pts
        A_Internal = A[1:-1, 1:-1]
        # sides: up, down, left, right
        (A_u, A_d, A_l, A_r) = (A[0, 1:-1], A[-1, 1:-1], A[1:-1, 0], A[1:-1, -1])
        # corners
        (A_ul, A_ur, A_dl, A_dr) = (A[0, 0], A[0, -1], A[-1, 0], A[-1, -1])
        # per dS, internal pts show up 4 times, edges 2 times, corners 1 time
        return dS * (np.sum(A_Internal)\
                    + 0.5 * (np.sum(A_u) + np.sum(A_d) + np.sum(A_l) + np.sum(A_r))\
                    + 0.25 * (A_ul + A_ur + A_dl + A_dr))

    @staticmethod
    def my_trapz(side_len_sq, centroid_x = None, centroid_y = None):
        '''
        Need to represent a guassian using the intrgated sum within each pixel
        '''
        scaler = 8
        steps = side_len_sq * int(scaler) # Must be divisible by original frame
        side_len = np.linspace(-0.5, side_len_sq - 0.5, steps+1)
        X, Y = np.meshgrid(side_len, side_len)

        # Default to random
        if centroid_x is None:
            centroid_x = int(side_len_sq/2)
        if centroid_y is None:
            centroid_y = int(side_len_sq/2)
        
        gauss = PsuedoChip.gauss_fn(X, Y, centroid_x, centroid_y)

        pixel_grid = np.zeros((5,5))
        for x in range(side_len_sq):
            for y in range(side_len_sq):
                x_low = x * scaler
                x_high = (x+1) * scaler + 1
                y_low = y * scaler
                y_high = (y+1) * scaler + 1
                pixel_grid[x,y] = PsuedoChip.double_Integral(x-0.5, x+0.5, y-0.5, y+0.5, scaler, scaler, gauss[x_low:x_high, y_low:y_high])
                # pixel_grid[x][y] = np.sum([gauss[x_low:x_high][y_low:y_high]])

        return pixel_grid

    @staticmethod
    def double_quadrature(side_len_sq, centroid_x = None, centroid_y = None):
        '''
        Need to represent a guassian using the intrgated sum within each pixel
        '''
        # Default to random
        if centroid_x is None:
            centroid_x = int(side_len_sq/2)
        if centroid_y is None:
            centroid_y = int(side_len_sq/2)

        pixel_grid = np.zeros((5,5))
        for x in range(side_len_sq):
            for y in range(side_len_sq):
                (integral, error) = integrate.dblquad(PsuedoChip.gauss_fn, x-0.5, x+0.5, y-0.5, y+0.5, args=(centroid_x, centroid_y))
                pixel_grid[x,y] = integral

        return pixel_grid

    @staticmethod
    def create_chip_rand_centroid(side_len_sq, fn):
        '''
        Moving centroid in center pixel
        '''
        x_centroid = np.random.random() + int(side_len_sq/2) - 0.5
        y_centroid = np.random.random() + int(side_len_sq/2) - 0.5
        return (fn(side_len_sq, x_centroid, y_centroid), x_centroid, y_centroid)

    def create_chip_rand_centroid_noise(self):
        '''
        Gaussian pixel distribution + random noise (w/ normal max level)
        '''
        # create new each call
        gauss_peak = np.random.randint(1, self.peak_max)

        # create new choosing from same norm. dist
        noise_ind = np.random.randint(0, self.num_bins)

        # Do convolution
        (Gauss_val, x_cent, y_cent) = self.create_chip_rand_centroid(self.sq_matrix_len, self.fn)
        self.data_gauss = Gauss_val * (1 / np.max(Gauss_val)) * gauss_peak # Photon counts discretize signal even futher 
        self.data_rand = np.random.random((self.sq_matrix_len, self.sq_matrix_len)) * self.noise_level[noise_ind] * np.max(self.data_gauss)
        self.data_conv = np.ceil(self.data_rand + self.data_gauss)
        return (self.data_conv, x_cent, y_cent)

    def create_exceedence_chip(self):
        '''
        Simulating Exceedence thresholding
        '''
        for r, row in enumerate(self.data_conv):
            for c, count in enumerate(row):
                if count > self.count_lim:
                    self.exceedence_chip[r, c] = count
        return self.exceedence_chip, self.data_conv


########################################################################################