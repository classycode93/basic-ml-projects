
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def build_features(df):

    df2 = df.copy()

    df2.loc[(df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "0-1"
    df2.loc[(df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "1-2"
    df2.loc[(df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "2-3"
    df2.loc[(df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "3-4"
    df2.loc[(df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "4-5"

    rating_df = pd.get_dummies(df2['rating_between'])
    language_df = pd.get_dummies(df2['language_code'])

    features = pd.concat([
        rating_df,
        language_df,
        df2['average_rating'],
        df2['ratings_count']
    ], axis=1)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    return features, df2
