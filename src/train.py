
from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    joblib.dump(model, "models/linear_regression_stock.pkl")

    return model
