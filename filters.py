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
from scipy.ndimage.measurements import label
from extractor import Extractor
from moviepy.editor import VideoFileClip

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
        self.sliding_box_param_level = [
            #{'size': 25, 'x_step': 12.5, 'y_step': 12.5, 'portion': 1 / 4},
            #{'size': 50, 'x_step': 25, 'y_step': 25, 'portion': 2 / 4},
            {'size': 100, 'x_step': 50, 'y_step': 50, 'portion': 0.5},
            {'size': 120, 'x_step': 60, 'y_step': 60, 'portion': 0.6},
            {'size': 140, 'x_step': 70, 'y_step': 70, 'portion': 0.7},
            {'size': 160, 'x_step': 80, 'y_step': 80, 'portion': 0.8},
            {'size': 200, 'x_step': 100, 'y_step': 100, 'portion': 0.9}
        ]
        self.heatmap_threshold = 1
        self.font = cv2.FONT_HERSHEY_COMPLEX

    def resetDiag(self):
        self.mainDiagScreen = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag1 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag2 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diag3 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.diagScreen = np.zeros((720, 1280, 3), dtype=np.uint8)

    def to3D(self, img):
        if len(img.shape) < 3:
            img = img / np.max(img) * 255
            return np.dstack((img, img, img))
        else:
            return img

    def diagScreenUpdate(self):
        self.mainDiagScreen = self.to3D(self.mainDiagScreen)
        self.diag1 = self.to3D(self.diag1)
        self.diag2 = self.to3D(self.diag2)
        self.diag3 = self.to3D(self.diag3)
        # assemble the screen example
        self.diagScreen = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.diagScreen[0:360, 0:640] = cv2.resize(self.mainDiagScreen, (640, 360), interpolation=cv2.INTER_AREA)

        self.diagScreen[0:360, 640:1280] = cv2.resize(self.diag1, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(self.diagScreen[0:360, 640:1280], 'diag1', (30, 60), self.font, 1, (255, 0, 0), 2)

        self.diagScreen[360:720, 0:640] = cv2.resize(self.diag2, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(self.diagScreen[360:720, 0:640], 'diag2', (30, 60), self.font, 1, (255, 0, 0), 2)

        self.diagScreen[360:720, 640:1280] = cv2.resize(self.diag3, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(self.diagScreen[360:720, 640:1280], 'diag3', (30, 60), self.font, 1, (255, 0, 0), 2)

    def predict_one_image(self,image):
        resize = cv2.resize(image,(self.img_col,self.img_row))
        X = self.ex.extract_features_one_image(resize)
        X = X.astype(np.float64)
        X_scaled = self.scaler.transform(X.reshape(1, -1))
        return self.svc.predict(X_scaled)

    def predict_batch(self,image_path):
        for path in image_path:
            image = cv2.imread(path)
            pred = self.predict_one_image(image)
            plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            plt.title("Pred:{}".format(pred))
            plt.show()

    def draw_one_box(self,image,size,centroid):
        topLeft = (int(centroid[0] - size / 2), int(centroid[1] - size / 2))
        bottomLeft = (int(centroid[0] - size / 2), int(centroid[1] + size / 2))
        topRight = (int(centroid[0] + size / 2), int(centroid[1] - size / 2))
        bottomRight = (int(centroid[0] + size / 2), int(centroid[1] + size / 2))
        C = (0, 255, 0)
        #cv2.line(image, topLeft, bottomLeft, C, 2)
        #cv2.line(image, bottomLeft, bottomRight, C, 2)
        #cv2.line(image, bottomRight, topRight, C, 2)
        #cv2.line(image, topLeft, topRight, C, 2)
        cv2.rectangle(image, bottomLeft, topRight, C, 4)

    def draw_boxes(self,image,size,centroids):
        for centroid in centroids:
            self.draw_one_box(image,size,centroid)

    def sliding_box_single_level(self,image,size,x_step,y_step,portion):
        # Portion is how much the algorithm is to be run on the image.
        # Eg: if portion = 1/4, only the top 1/4 portions of the images are to be slidied
        image_cropped = image[:int(image.shape[0]*portion),:]
        pos_list = self.generate_box_pos(image_cropped.shape,size,x_step=x_step,y_step=y_step)
        detected_true = []
        for pos in pos_list:
            topLeft = (int(pos[0] - size / 2), int(pos[1] - size / 2))
            bottomLeft = (int(pos[0] - size / 2), int(pos[1] + size / 2))
            topRight = (int(pos[0] + size / 2), int(pos[1] - size / 2))
            bottomRight = (int(pos[0] + size / 2), int(pos[1] + size / 2))
            image_boxed = image_cropped[topLeft[1]:bottomLeft[1],topLeft[0]:topRight[0]]
            y_pred = self.predict_one_image(image_boxed)
            if y_pred[0]:
                detected_true.append(pos)
        return detected_true

    def sliding_box_multi_level(self,image,level = 2):
        image_copy = np.copy(image)
        y_offset = int(image_copy.shape[0]/2)
        image_copy_cropped = image_copy[y_offset:,:]
        centroids_and_sizes = [] # member element is a dictionary
        if level != 'ALL':
            if level > 3:
                raise ValueError("Level cannot exceed {}".format(len(self.sliding_box_param_level)))
            centroids = self.sliding_box_single_level(image_copy_cropped,**(self.sliding_box_param_level[level]))
            centroids = np.array(centroids)
            centroids[:, 1] = centroids[:, 1] + y_offset
            kwargs = {"centroids":centroids,"size":self.sliding_box_param_level[level]["size"]}
            centroids_and_sizes.append(kwargs)
            self.draw_boxes(image_copy,**kwargs)
        else:
            for level in range(len(self.sliding_box_param_level)):
                centroids = self.sliding_box_single_level(image_copy_cropped, **(self.sliding_box_param_level[level]))
                centroids = np.array(centroids)
                if len(centroids)>0:
                    centroids[:, 1] = centroids[:, 1] + y_offset
                    kwargs = {"centroids": centroids, "size": self.sliding_box_param_level[level]["size"]}
                    centroids_and_sizes.append(kwargs)
                    self.draw_boxes(image_copy, **kwargs)
        return image_copy, centroids_and_sizes

    def generate_box_pos(self,image_shape,size,x_step,y_step):
        x_min, x_max = 0, image_shape[1]
        y_min, y_max = 0, image_shape[0]
        pos = [x_min + int(size / 2), y_min + int(size / 2)]
        pos_list = []
        while True:
            if pos[1] + int(size / 2) > y_max:
                if pos_list[-1][1] + int(size / 2) != y_max:
                    pos[1] = y_max - int(size / 2)
                    continue
                else:
                    break
            while True:
                if pos[0] + int(size / 2) > x_max:
                    if pos_list[-1][0] + int(size / 2) != x_max:
                        pos[0] = x_max - int(size / 2)
                        pos_list.append(list(pos))
                        pos[0] = x_min + int(size / 2)
                        break
                    else:
                        pos[0] = x_min + int(size / 2)
                        break
                pos_list.append(list(pos))
                pos[0] += x_step
            pos[1] += y_step
        return pos_list

    def add_heat(self, image_shape, centroids_and_sizes):
        heatmap = np.zeros(image_shape[:2])
        # Iterate through list of bboxes
        for i in range(len(centroids_and_sizes)):
            centroids = centroids_and_sizes[i]["centroids"]
            size = centroids_and_sizes[i]["size"]
            for centroid in centroids:
                topLeft = (int(centroid[0] - size / 2), int(centroid[1] - size / 2))
                bottomLeft = (int(centroid[0] - size / 2), int(centroid[1] + size / 2))
                topRight = (int(centroid[0] + size / 2), int(centroid[1] - size / 2))
                bottomRight = (int(centroid[0] + size / 2), int(centroid[1] + size / 2))
                heatmap[topLeft[1]:bottomLeft[1], topLeft[0]:topRight[0]] += 1
        # Return updated heatmap
        return heatmap

    def apply_heat_threshold(self,heatmap):
        heatmap_copy = np.zeros_like(heatmap)
        heatmap_copy[heatmap >= self.heatmap_threshold] = 1
        return heatmap_copy

    def draw_final_bbox(self,original_image, heatmap):
        original_image_copy = np.copy(original_image)
        labels = label(heatmap)
        num_car_found = labels[1]
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(original_image_copy, bbox[0], bbox[1], (0, 0, 255), 6)
        return original_image_copy, num_car_found

    def pipepine(self,image):
        image_res, centroids_and_sizes = self.sliding_box_multi_level(image, level='ALL')
        self.diag1 = image_res
        heatmap = self.add_heat(image.shape, centroids_and_sizes)
        self.diag2 = heatmap
        heatmap = self.apply_heat_threshold(heatmap)
        self.diag3 = heatmap
        final_image, num_car_found = self.draw_final_bbox(image, heatmap)
        self.mainDiagScreen = final_image
        self.diagScreenUpdate()
        return self.diagScreen

    def extract_half_image_hog(self, image):
        image_cropped = np.copy(image[-int(image.shape[0] / 2):, :])
        hog_features = self.ex.get_hog_features_multi_channels(
            image_cropped, orient=self.ex.hog_orient,
            pix_per_cell=self.ex.hog_pix_per_cell,
            cell_per_block=self.ex.hog_cell_per_block,
            hog_channel=self.ex.hog_hog_channel,
            feature_vec=False)
        pass

    def toVideo(self, input_video_file_name, output_video_file_name):
        clipInput = VideoFileClip(input_video_file_name)
        clipOutput = clipInput.fl_image(self.pipepine)
        clipOutput.write_videofile(output_video_file_name, audio=False)