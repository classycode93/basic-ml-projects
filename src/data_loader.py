
import pandas as pd

def load_data(path, symbol):

    df = pd.read_csv(path)

    df = df[df.symbol == symbol]

    return df
