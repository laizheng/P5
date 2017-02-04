import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import time
import pickle
from extractor import Extractor

class Filter():
    def __init__(self,model_file=None,scaler_file=None):
        if model_file==None or scaler_file==None:
            raise ValueError("Need to provide model file and scaler filer.")
        f_model = open(model_file, 'rb')
        self.svc = pickle.load(f_model)
        f_scaler = open(scaler_file, 'rb')
        self.scaler = pickle.load(f_scaler)
        self.img_col = 64
        self.img_row = 64
        self.ex = Extractor()
        self.test_clf_image_paths = glob("./clf_test_image/*.*")
        self.test_video_images_path = glob("./video_images_1/*.*")

    def predict_one_image(self,image):
        resize = cv2.resize(image,(self.img_col,self.img_row))
        X = self.ex.extract_features_one_image(resize)
        X = X.astype(np.float64)
        X_scaled = self.scaler.transform(X)
        return self.svc.predict([X_scaled])

    def predict_batch(self,image_path):
        for path in image_path:
            image = cv2.imread(path)
            pred = self.predict_one_image(image)
            plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            plt.title("Pred:{}".format(pred))
            plt.show()

    def draw_one_box(self,image,size,centroid):
        topLeft = (int(centroid[1] - size / 2), int(centroid[0] - size / 2))
        bottomLeft = (int(centroid[1] - size / 2), int(centroid[0] + size / 2))
        topRight = (int(centroid[1] + size / 2), int(centroid[0] - size / 2))
        bottomRight = (int(centroid[1] + size / 2), int(centroid[0] + size / 2))
        C = (0, 255, 0)
        #cv2.line(image, topLeft, bottomLeft, C, 2)
        #cv2.line(image, bottomLeft, bottomRight, C, 2)
        #cv2.line(image, bottomRight, topRight, C, 2)
        #cv2.line(image, topLeft, topRight, C, 2)
        cv2.rectangle(image, bottomLeft, topRight, C, 4)

    def extract_half_image_hog(self, image):
        image_cropped = np.copy(image[-int(image.shape[0] / 2):, :])
        hog_features = self.ex.get_hog_features_multi_channels(
            image_cropped, orient=self.ex.hog_orient,
            pix_per_cell=self.ex.hog_pix_per_cell,
            cell_per_block=self.ex.hog_cell_per_block,
            hog_channel=self.ex.hog_hog_channel,
            feature_vec=False)
        pass
