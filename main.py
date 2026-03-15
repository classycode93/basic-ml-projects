
from src.data_loader import load_data
from src.preprocessing import prepare_data
from src.train import train_model
from src.predict import predict_prices

def main():

    df = load_data("data/prices.csv", "GOOG")

    forecast_col = "close"
    forecast_out = 5
    test_size = 0.2

    X_train, X_test, y_train, y_test, X_lately = prepare_data(
        df, forecast_col, forecast_out, test_size
    )

    model = train_model(X_train, y_train)

    predict_prices(model, X_test, y_test, X_lately)

if __name__ == "__main__":
    main()
