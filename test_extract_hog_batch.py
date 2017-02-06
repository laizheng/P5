from filters import Filter
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    filter = Filter(model_file="model.p",scaler_file="scaler.p")
    image = cv2.imread(filter.ex.car_paths[0])
    filter.extract_half_image_hog(image)
if __name__ == "__main__":
    main()