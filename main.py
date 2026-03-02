from src.config import load_config
from src.preprocessing import Preprocessing
from src.model import Model


config = load_config("./configs/config.yaml")

preprocess = Preprocessing(config)
model = Model(config)


def main():
    embedding = preprocess.run_preprocessing()
    output = model.forward(embedding)


if __name__ == "__main__":
    main()
    
