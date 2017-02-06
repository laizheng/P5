from extractor import Extractor
import matplotlib.pyplot as plt
import cv2
def main():
    ex = Extractor()
    # ex.grid_search_linSVC()
    # ex.use_small_set()
    ex.train(model_file="model.p",scaler_file="scaler.p")
    #ex.predict(model_file="model.p",scaler_file="scaler.p")
    #ex.grid_search()

if __name__ == "__main__":
    main()