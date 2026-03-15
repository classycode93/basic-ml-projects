
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def run_kmeans(data):

    x = data.iloc[:, [5,6]].values

    wcss = []

    for i in range(1,11):
        km = KMeans(
            n_clusters=i,
            init='k-means++',
            max_iter=300,
            n_init=10,
            random_state=0
        )

        km.fit(x)
        wcss.append(km.inertia_)

    plt.plot(range(1,11), wcss)
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()

    km = KMeans(n_clusters=2, random_state=0)

    labels_pred = km.fit_predict(x)

    plt.scatter(x[labels_pred == 0,0], x[labels_pred == 0,1], label="Uninterested")
    plt.scatter(x[labels_pred == 1,0], x[labels_pred == 1,1], label="Target Customers")
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], label="Centroids")

    plt.legend()
    plt.show()

    return x, labels_pred
