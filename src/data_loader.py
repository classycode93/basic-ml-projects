
import pandas as pd

def load_data(path):

    data = pd.read_csv(path)

    data = data[['Date','Time','Latitude','Longitude','Depth','Magnitude']]

    return data
