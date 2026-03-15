
from src.data_loader import load_data
from src.preprocessing import build_corpus
from src.sequence_generator import create_sequences
from src.model import create_model
from src.generator import generate_text

def main():

    corpus = load_data("data")

    input_sequences, total_words = create_sequences(corpus)

    predictors, label, max_sequence_len = build_corpus(input_sequences, total_words)

    model = create_model(max_sequence_len, total_words)

    model.fit(predictors, label, epochs=10)

    print(generate_text("spiderman", 5, model, max_sequence_len))

if __name__ == "__main__":
    main()
