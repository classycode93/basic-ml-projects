
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

def create_sequences(corpus):

    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1

    input_sequences = []

    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    return input_sequences, total_words
