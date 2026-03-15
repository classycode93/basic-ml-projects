
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

def create_model(max_sequence_len, total_words):

    input_len = max_sequence_len - 1

    model = Sequential()

    model.add(Embedding(total_words, 10, input_length=input_len))

    model.add(LSTM(100))
    model.add(Dropout(0.1))

    model.add(Dense(total_words, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model
