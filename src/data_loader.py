
import pandas as pd
import json

def category_extractor(data):
    ids = [data['items'][i]['id'] for i in range(len(data['items']))]
    titles = [data['items'][i]['snippet']['title'] for i in range(len(data['items']))]
    ids = list(map(int, ids))
    return dict(zip(ids, titles))

def load_data(path):

    df1 = pd.read_csv(f"{path}/USvideos.csv")
    df2 = pd.read_csv(f"{path}/CAvideos.csv")
    df3 = pd.read_csv(f"{path}/GBvideos.csv")

    data1 = json.load(open(f"{path}/US_category_id.json"))
    data2 = json.load(open(f"{path}/CA_category_id.json"))
    data3 = json.load(open(f"{path}/GB_category_id.json"))

    df1['category_title'] = df1['category_id'].map(category_extractor(data1))
    df2['category_title'] = df2['category_id'].map(category_extractor(data2))
    df3['category_title'] = df3['category_id'].map(category_extractor(data3))

    df = pd.concat([df1, df2, df3])
    df = df.drop_duplicates("video_id")

    entertainment = df[df["category_title"] == "Entertainment"]["title"]

    return entertainment.tolist()
