from filters import Filter
from glob import glob
import cv2
import matplotlib.pyplot as plt
def main():
    filter = Filter(model_file="model.p",scaler_file="scaler.p")

    #filter.predict_batch(image_path=glob("./labeled_data_smallset/vehicles_smallset/**/*.*"))
    #filter.predict_batch(image_path=filter.test_clf_image_paths)

    image = cv2.cvtColor(cv2.imread(filter.test_video_images_path[0]), cv2.COLOR_BGR2RGB)
    filter.extract_half_image_hog(image)

    """
    filter.draw_one_box(image,150,(500,500))
    plt.imshow(image)
    plt.show()
    """

if __name__ == "__main__":
    main()