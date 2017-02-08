from filters import Filter
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    filter = Filter(model_file="model.p",scaler_file="scaler.p")
    #filter.predict_batch(image_path=glob("./labeled_data_smallset/vehicles_smallset/**/*.*"))
    #filter.predict_batch(image_path=filter.test_clf_image_paths)
    frame = None
    cnt = 0
    for path in filter.test_video_images_path:
        cnt += 1
        if frame != None and cnt == frame:
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            final_image = filter.pipepine(image)
            plt.imshow(final_image)
            plt.show()
            break
        elif frame == None:
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            final_image = filter.pipepine(image)
            plt.imshow(final_image)
            plt.show()
    # image_res, centroids_and_sizes = filter.sliding_box_multi_level(image, level=2)

    """
    filter.draw_one_box(image,150,(500,500))
    plt.imshow(image)
    plt.show()
    """

if __name__ == "__main__":
    main()