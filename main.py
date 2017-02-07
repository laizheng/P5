from filters import Filter


def main():
    filter = Filter(model_file="model.p",scaler_file="scaler.p")
    filter.toVideo("project_video.mp4",output_video_file_name="out.mp4")
if __name__ == "__main__":
    main()