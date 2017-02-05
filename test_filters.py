from filters import Filter
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    filter = Filter(model_file="model.p",scaler_file="scaler.p")

    #filter.predict_batch(image_path=glob("./labeled_data_smallset/vehicles_smallset/**/*.*"))
    #filter.predict_batch(image_path=filter.test_clf_image_paths)

    image = cv2.cvtColor(cv2.imread(filter.test_video_images_path[7]), cv2.COLOR_BGR2RGB)
    image_res, centroids_and_sizes = filter.sliding_box_multi_level(image, level=2)
    #filter.sliding_box_single_level(image,**filter.sliding_box_param_level1)
    #image_res, centroids_and_sizes = filter.sliding_box_multi_level(image,level='all')
    #heatmap = filter.add_heat(image.shape,centroids_and_sizes)
    #heatmap = filter.apply_heat_threshold(heatmap)
    final_image = filter.pipepine(image)
    plt.imshow(final_image)
    plt.show()

    #filter.extract_half_image_hog(image)

    """
    filter.draw_one_box(image,150,(500,500))
    plt.imshow(image)
    plt.show()
    """

if __name__ == "__main__":
    main()