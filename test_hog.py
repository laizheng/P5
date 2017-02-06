import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import extractor as ex
import cv2
import matplotlib.image as mpimg
from extractor import Extractor

ex = Extractor()

#image = color.rgb2gray(data.astronaut())
image = cv2.imread(ex.car_paths[0])
image = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
image_channeled = image[:,:,1]
fd, hog_image = ex.get_hog_features_one_channel(image_channeled, vis=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image_channeled, cmap=plt.cm.gray)
ax1.set_title('Input image')
#ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
#ax1.set_adjustable('box-forced')
plt.show()