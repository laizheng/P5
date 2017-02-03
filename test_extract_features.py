import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob
from extractor import Extractor

def main():
    ex = Extractor()
    features = ex.extract_features_all_images()

if __name__ == "__main__":
    main()
