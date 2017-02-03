import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
import pickle

class Extractor():
    def __init__(self):
        car_pattern = "./labeled_data_smallset/vehicles_smallset/**/*.*"
        noncar_pattern = "./labeled_data_smallset/non-vehicles_smallset/**/*.*"
        #car_pattern = "./labeled_data/vehicles/**/*.*"
        #noncar_pattern = "./labeled_data/non-vehicles/**/*.*"
        self.car_paths = glob(car_pattern,recursive = True)
        self.noncar_paths = glob(noncar_pattern,recursive = True)
        self.hog_cspace = 'HLS'
        self.hog_orient = 8
        self.hog_pix_per_cell = 8
        self.hog_cell_per_block = 2
        self.hog_hog_channel = 2

    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the RGB channels separately
        ch1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        ch2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        ch3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # Generating bin centers
        bin_edges = ch1_hist[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features, bin_centers, ch1_hist, ch2_hist, ch3_hist

    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    def get_hog_features(self, img, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), visualise=vis, \
                                      feature_vector=feature_vec)
            return features, hog_image
        else:
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), visualise=vis, \
                                      feature_vector=feature_vec)
            return features

    def extract_features_colorhist_binspatial(self, image_paths, cspace='RGB', spatial_size=(32, 32),
                            hist_bins=32, hist_range=(0, 256)):
        # Create a list to append feature vectors to
        features = []
        for path in image_paths:
            image = cv2.imread(path)
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            else:
                feature_image = np.copy(image)
            bin_spatial_features = self.bin_spatial(feature_image,size=spatial_size)
            color_hist_features, _, _, _, _ = self.color_hist(feature_image,nbins=hist_bins,bins_range=hist_range)
            features.append(np.concatenate((bin_spatial_features, color_hist_features)))
        return features

    def extract_features_hog(self, image_paths, cspace='RGB', orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0):
        print("Performing hog feature extrations... # of images to process = {}".format(len(image_paths)))
        print("Parameters: cspace={},orient={},"
              "pix_per_cell={},cell_per_block={},"
              "hog_channel={}".format(cspace, \
                                      orient,
                                      pix_per_cell,
                                      cell_per_block,
                                      hog_channel))
        features = []
        for path in image_paths:
            image = cv2.imread(path)
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                elif cspace == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            else:
                feature_image = np.copy(image)
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(self.get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = self.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            features.append(hog_features)
        print("Hog feature extraction concluded")
        return features

    def normalize(self, car_features, noncar_features):
        if len(car_features)>0 and len(noncar_features)>0:
            print("Performing normalization...")
            # Create an array stack of feature vectors
            X = np.vstack((car_features, noncar_features)).astype(np.float64)
            # Fit a per-column scaler
            X_scaler = StandardScaler().fit(X)
            # Apply the scaler to X
            scaled_X = X_scaler.transform(X)
            y = np.hstack((np.ones(len(car_features)),
                           np.zeros(len(noncar_features))))
            print("Normalization concluded...")
            return scaled_X, y
        else:
            raise ValueError("Empty features found.")

    def train(self,model_file):
        #extract = self.extract_features_colorhist_binspatial
        #car_features = extract(self.car_paths, cspace='RGB', spatial_size=(32, 32),
        #                                               hist_bins=32, hist_range=(0, 256))
        #noncar_features = extract(self.noncar_paths, cspace='RGB', spatial_size=(32, 32),
        #                                                  hist_bins=32, hist_range=(0, 256))
        car_features = self.extract_features_hog(self.car_paths,
                                                 cspace=self.hog_cspace,
                                                 orient=self.hog_orient,
                                                 pix_per_cell=self.hog_pix_per_cell,
                                                 cell_per_block=self.hog_cell_per_block,
                                                 hog_channel=self.hog_hog_channel)
        noncar_features = self.extract_features_hog(self.noncar_paths,
                                                    cspace=self.hog_cspace,
                                                    orient=self.hog_orient,
                                                    pix_per_cell=self.hog_pix_per_cell,
                                                    cell_per_block=self.hog_cell_per_block,
                                                    hog_channel=self.hog_hog_channel)
        print("# of cars:{}, # of non-cars :{}".format(len(car_features), len(noncar_features)))
        scaled_X, y = self.normalize(car_features,noncar_features)
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
        print("In training set, there are {} cars and {} noncars".format(len(y_train[y_train == 1]),
                                                                         len(y_train[y_train == 0])))
        print("In test set, there are {} cars and {} noncars".format(len(y_test[y_test == 1]),
                                                                         len(y_test[y_test == 0])))
        print('Feature vector length:', len(X_train[0]))
        #svc = LinearSVC()
        svc = SVC(C=3.981, gamma=0.0004641, verbose=True) # test acc = 0.9791
        t = time.time()
        print("Start Training...")
        svc.fit(X_train, y_train)
        t2 = time.time()
        print("\n",round(t2 - t, 2), 'Seconds to train SVC...')
        print('Train Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
        with open(model_file, 'wb') as f:
            pickle.dump(svc, f)

    def grid_search(self):
        car_features = self.extract_features_hog(self.car_paths,
                                                 cspace=self.hog_cspace,
                                                 orient=self.hog_orient,
                                                 pix_per_cell=self.hog_pix_per_cell,
                                                 cell_per_block=self.hog_cell_per_block,
                                                 hog_channel=self.hog_hog_channel)
        noncar_features = self.extract_features_hog(self.noncar_paths,
                                                    cspace=self.hog_cspace,
                                                    orient=self.hog_orient,
                                                    pix_per_cell=self.hog_pix_per_cell,
                                                    cell_per_block=self.hog_cell_per_block,
                                                    hog_channel=self.hog_hog_channel)
        print("# of cars:{}, # of non-cars :{}".format(len(car_features), len(noncar_features)))
        scaled_X, y = self.normalize(car_features, noncar_features)
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
        print("In training set, there are {} cars and {} noncars".format(len(y_train[y_train == 1]),
                                                                         len(y_train[y_train == 0])))
        print("In test set, there are {} cars and {} noncars".format(len(y_test[y_test == 1]),
                                                                     len(y_test[y_test == 0])))
        print('Feature vector length:', len(X_train[0]))

        rand_state = np.random.randint(0, 100)
        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=rand_state)
        C_range = np.logspace(-1, 3, 6)
        gamma_range = np.logspace(-4, -3, 4)
        param_grid = dict(gamma=gamma_range, C=C_range)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=sss,verbose=5)
        grid.fit(X_train,y_train)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        plt.show()
        pass

    def predict(self,model_file):
        print("\n\n----------------Peforming Test-------------\n\n")
        with open(model_file, 'rb') as f:
            svc = pickle.load(f)
            car_features = self.extract_features_hog(self.car_paths,
                                                     cspace=self.hog_cspace,
                                                     orient=self.hog_orient,
                                                     pix_per_cell=self.hog_pix_per_cell,
                                                     cell_per_block=self.hog_cell_per_block,
                                                     hog_channel=self.hog_hog_channel)
            noncar_features = self.extract_features_hog(self.noncar_paths,
                                                        cspace=self.hog_cspace,
                                                        orient=self.hog_orient,
                                                        pix_per_cell=self.hog_pix_per_cell,
                                                        cell_per_block=self.hog_cell_per_block,
                                                        hog_channel=self.hog_hog_channel)
            scaled_X, y = self.normalize(car_features, noncar_features)
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X, y, test_size=0.2, random_state=rand_state)
            print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
