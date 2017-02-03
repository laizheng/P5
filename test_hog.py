import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import extractor as ex
import cv2
import matplotlib.image as mpimg
from extractor import Extractor

ex = Extractor()

#image = color.rgb2gray(data.astronaut())
path = './labeled_data_smallset/vehicles_smallset/cars1/1.jpeg'
image = mpimg.imread(path)
image = color.rgb2gray(image)

fd, hog_image = ex.get_hog_features(image, orient=8, pix_per_cell=8,
                                    cell_per_block=2, vis=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
#ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
#ax1.set_adjustable('box-forced')
plt.show()