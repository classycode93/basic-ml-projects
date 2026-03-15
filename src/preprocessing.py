
import pandas as pd
import datetime
import time
from sklearn.model_selection import train_test_split

def preprocess_data(data):

    timestamp = []

    for d, t in zip(data['Date'], data['Time']):

        try:
            ts = datetime.datetime.strptime(d+' '+t,'%m/%d/%Y %H:%M:%S')
            timestamp.append(time.mktime(ts.timetuple()))
        except:
            timestamp.append(None)

    data['Timestamp'] = timestamp

    data = data.drop(['Date','Time'], axis=1)

    data = data.dropna()

    X = data[['Timestamp','Latitude','Longitude']]
    y = data[['Magnitude','Depth']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
