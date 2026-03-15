
def evaluate_model(model, X_test, y_test):

    loss, acc = model.evaluate(X_test, y_test)

    print("Test Loss:", loss)
    print("Test Accuracy:", acc)
