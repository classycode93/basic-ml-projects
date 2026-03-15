
import numpy as np

def predict_prices(model, X_test, y_test, X_lately):

    score = model.score(X_test, y_test)

    forecast = model.predict(X_lately)

    response = {
        "test_score": float(score),
        "forecast_set": forecast.tolist()
    }

    print(response)
