from extractor import Extractor

def main():
    ex = Extractor()
    ex.train(model_file="model.p")
    ex.predict(model_file="model.p")
    #ex.grid_search()

if __name__ == "__main__":
    main()