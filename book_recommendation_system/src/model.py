
from sklearn import neighbors

def train_model(features):

    model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    model.fit(features)

    dist, idlist = model.kneighbors(features)

    return model, idlist

def recommend_books(book_name, df, idlist):

    book_list = []

    book_id = df[df['title'] == book_name].index[0]

    for new_id in idlist[book_id]:
        book_list.append(df.loc[new_id].title)

    return book_list
