import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import extractor as ex
# Read in an image
# You can also read cutout2, 3, 4 etc. to see other examples
image = mpimg.imread('cutout1.jpg')

feature_vec = ex.bin_spatial(image, color_space='RGB', size=(32, 32))

# Plot features
plt.plot(feature_vec)
plt.title('Spatially Binned Features')
plt.show()