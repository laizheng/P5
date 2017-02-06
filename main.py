from filters import Filter


def main():
    filter = Filter(model_file="model.p",scaler_file="scaler.p")
    filter.toVideo("project_video_short.mp4",output_video_file_name="out_short.mp4")
if __name__ == "__main__":
    main()