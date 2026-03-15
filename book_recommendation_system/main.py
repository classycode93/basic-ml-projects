
from src.data_loader import load_data
from src.preprocessing import build_features
from src.model import train_model, recommend_books

def main():

    df = load_data("data/books.csv")

    features, df_processed = build_features(df)

    model, idlist = train_model(features)

    results = recommend_books("Harry Potter and the Half-Blood Prince (Harry Potter  #6)", df_processed, idlist)

    print("Recommended Books:")
    for r in results:
        print("-", r)

if __name__ == "__main__":
    main()
