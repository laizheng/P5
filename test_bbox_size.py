from filters import Filter
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    filter = Filter(model_file="model.p",scaler_file="scaler.p")
    image = cv2.cvtColor(cv2.imread(filter.test_video_images_path[0]), cv2.COLOR_BGR2RGB)
    filter.draw_one_box(image,120,(500,500))
    filter.draw_one_box(image, 140, (500, 500))
    filter.draw_one_box(image, 160, (500, 500))
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()