
from src.train import train_model
from src.detect import detect_mask

def main():

    print("Training model...")
    train_model()

    print("Running mask detection...")
    detect_mask()

if __name__ == "__main__":
    main()
