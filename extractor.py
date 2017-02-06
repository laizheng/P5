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
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

class Extractor():
    def __init__(self):
        car_pattern = "./labeled_data/vehicles/**/*.*"
        noncar_pattern = "./labeled_data/non-vehicles/**/*.*"
        self.car_paths = glob(car_pattern, recursive=True)
        self.noncar_paths = glob(noncar_pattern, recursive=True)
        self.hog_cspace = 'HLS'
        self.hog_orient = 9
        self.hog_pix_per_cell = 8
        self.hog_cell_per_block = 3
        self.hog_hog_channel = 1

    def use_small_set(self):
        car_pattern = "./labeled_data_smallset/vehicles_smallset/**/*.*"
        noncar_pattern = "./labeled_data_smallset/non-vehicles_smallset/**/*.*"
        self.car_paths = glob(car_pattern, recursive=True)
        self.noncar_paths = glob(noncar_pattern, recursive=True)

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

    def get_hog_features_one_channel(self, img, vis=False, feature_vec=True):
        orient = self.hog_orient
        pix_per_cell = self.hog_pix_per_cell
        cell_per_block = self.hog_cell_per_block
        if vis:
            features, hog_image = hog(img,
                                      orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      visualise=vis,
                                      feature_vector=feature_vec)
            return features, hog_image
        else:
            features = hog(img,
                           orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           visualise=vis,
                           feature_vector=feature_vec)
            return features

    def get_hog_features_multi_channels(self, feature_image, hog_channel=0, feature_vec=True):
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(self.get_hog_features_one_channel(
                    feature_image[:, :, channel],
                    vis=False, feature_vec=feature_vec))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = self.get_hog_features_one_channel(
                feature_image[:, :, hog_channel], vis=False, feature_vec=feature_vec)
        return hog_features

    def extract_features_one_image(self, image):
        if self.hog_cspace != 'RGB':
            if self.hog_cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif self.hog_cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif self.hog_cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif self.hog_cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif self.hog_cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hog_features = self.get_hog_features_multi_channels(feature_image, hog_channel=self.hog_hog_channel)
        #bin_spatial_features = self.bin_spatial(image, size=(16, 16))
        #color_hist_features, _, _, _, _ = self.color_hist(image, nbins=32, bins_range=(0, 256))
        #color_hist_features = color_hist_features.astype(float)
        #return np.concatenate((color_hist_features, bin_spatial_features, hog_features))
        return hog_features

    def extract_features_batch(self, image_paths):
        print("Performing feature extrations... # of images to process = {}".format(len(image_paths)))
        print("Parameters: cspace={},orient={},"
              "pix_per_cell={},cell_per_block={},"
              "hog_channel={}".format(self.hog_cspace,
                                      self.hog_orient,
                                      self.hog_pix_per_cell,
                                      self.hog_cell_per_block,
                                      self.hog_hog_channel))
        features = []
        for path in image_paths:
            image = cv2.imread(path)
            features.append(self.extract_features_one_image(image))
        print("feature extraction concluded")
        return features

    def normalize(self, car_features, noncar_features):
        if len(car_features) > 0 and len(noncar_features) > 0:
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
            return scaled_X, y, X_scaler
        else:
            raise ValueError("Empty features found.")

    def train(self, model_file, scaler_file):
        # extract = self.extract_features_colorhist_binspatial
        # car_features = extract(self.car_paths, cspace='RGB', spatial_size=(32, 32),
        #                                               hist_bins=32, hist_range=(0, 256))
        # noncar_features = extract(self.noncar_paths, cspace='RGB', spatial_size=(32, 32),
        #                                                  hist_bins=32, hist_range=(0, 256))
        car_features = self.extract_features_batch(self.car_paths)
        noncar_features = self.extract_features_batch(self.noncar_paths)
        print("# of cars:{}, # of non-cars :{}".format(len(car_features), len(noncar_features)))
        scaled_X, y, scaler = self.normalize(car_features, noncar_features)
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
        print("In training set, there are {} cars and {} noncars".format(len(y_train[y_train == 1]),
                                                                         len(y_train[y_train == 0])))
        print("In test set, there are {} cars and {} noncars".format(len(y_test[y_test == 1]),
                                                                     len(y_test[y_test == 0])))
        print('Feature vector length:', len(X_train[0]))
        # The best parameters are {'gamma': 0.0001, 'C': 3.9810717055349731} with a score of 1.00
        # Train Accuracy of SVC =  0.9882
        # Test Accuracy of SVC =  0.9756

        # svc = LinearSVC(C=4)
        svc = SVC(C=4, gamma=0.0001, verbose=True)
        t = time.time()
        print("Start Training...")
        svc.fit(X_train, y_train)
        t2 = time.time()
        print("\n", round(t2 - t, 2), 'Seconds to train SVC...')
        print('Train Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        y_test_pred = svc.predict(X_test)
        recall = precision_score(y_test, y_test_pred)
        print('precision Accuracy of SVC on test set = ', round(recall, 4))
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
        with open(model_file, 'wb') as f:
            pickle.dump(svc, f)
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)

    def grid_search_linSVC(self):
        car_features = self.extract_features_batch(self.car_paths)
        noncar_features = self.extract_features_batch(self.noncar_paths)
        print("# of cars:{}, # of non-cars :{}".format(len(car_features), len(noncar_features)))
        scaled_X, y, _ = self.normalize(car_features, noncar_features)
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
        C_range = np.logspace(-1, 2, 10)
        param_grid = dict(C=C_range)
        grid = GridSearchCV(LinearSVC(), param_grid=param_grid, cv=sss, verbose=5)
        grid.fit(X_train, y_train)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        scores = grid.cv_results_['mean_test_score']
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.plot(scores)
        plt.xlabel('C')
        plt.ylabel('CV Score')
        plt.xticks(np.arange(len(C_range)), C_range, rotation=45)
        plt.title('Validation accuracy')
        plt.show()
        pass

    def grid_search(self):
        car_features = self.extract_features_batch(self.car_paths)
        noncar_features = self.extract_features_batch(self.noncar_paths)
        print("# of cars:{}, # of non-cars :{}".format(len(car_features), len(noncar_features)))
        scaled_X, y, _ = self.normalize(car_features, noncar_features)
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
        C_range = np.logspace(-1, 1, 6)
        gamma_range = np.logspace(-4, -3, 4)
        param_grid = dict(gamma=gamma_range, C=C_range)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=sss, verbose=5)
        grid.fit(X_train, y_train)
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

    def predict(self, model_file, scaler_file):
        print("\n\n----------------Peforming Test-------------\n\n")
        f_model = open(model_file, 'rb')
        svc = pickle.load(f_model)
        f_scaler = open(scaler_file, 'rb')
        scaler = pickle.load(f_scaler)
        car_features = self.extract_features_batch(self.car_paths)
        noncar_features = self.extract_features_batch(self.noncar_paths)
        X = np.vstack((car_features, noncar_features)).astype(np.float64)
        scaled_X = scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
        y_test_pred =svc.predict(X_test)
        recall = precision_score(y_test, y_test_pred)
        print('precision Accuracy of SVC on test set = ', round(recall, 4))
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
