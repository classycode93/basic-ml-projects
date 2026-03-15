
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import scikitplot as skplt
import matplotlib.pyplot as plt

def evaluate_clusters(data, labels_pred):

    le = LabelEncoder()

    labels_true = le.fit_transform(data["Revenue"])

    score = metrics.adjusted_rand_score(labels_true, labels_pred)

    print("Adjusted Rand Index:", score)

    skplt.metrics.plot_confusion_matrix(labels_true, labels_pred)
    plt.show()
