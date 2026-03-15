
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.cluster import run_kmeans
from src.evaluate import evaluate_clusters

def main():

    data = load_data("data/online_shoppers_intention.csv")

    data = clean_data(data)

    x, labels_pred = run_kmeans(data)

    evaluate_clusters(data, labels_pred)

if __name__ == "__main__":
    main()
