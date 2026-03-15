
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras.utils as ku

def build_corpus(input_sequences, total_words):

    max_sequence_len = max([len(x) for x in input_sequences])

    input_sequences = np.array(
        pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
    )

    predictors = input_sequences[:, :-1]
    label = input_sequences[:, -1]

    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len
