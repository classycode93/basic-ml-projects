
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model(X_train, y_train):

    model = Sequential()

    model.add(Dense(16, activation='relu', input_shape=(3,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(
        optimizer='SGD',
        loss='squared_hinge',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=10, batch_size=10)

    return model
