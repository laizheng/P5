from filters import Filter
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

def main():
    filter = Filter(model_file="model.p",scaler_file="scaler.p")
    clip = VideoFileClip("project_video_short3.mp4")
    cnt = 0
    stop_frame_num = 113
    for img in clip.iter_frames():
        cnt += 1
        if (cnt == stop_frame_num):
            if img.shape[2] == 4:
                img = img[:, :, :3]
            ret = filter.pipepine(img)
            plt.figure(figsize=(16, 10))
            plt.imshow(filter.diagScreen)
            plt.subplots_adjust(left=0.03, bottom=0.03, right=1, top=1)
            plt.show()

if __name__ == "__main__":
    main()