from extractor import Extractor
import matplotlib.pyplot as plt
import cv2
def main():
    ex = Extractor()
    ex.train(model_file="model.p",scaler_file="scaler.p")
    #ex.gen_prediction(model_file="model.p", scaler_file="scaler.p",n=3)
if __name__ == "__main__":
    main()