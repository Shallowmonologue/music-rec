import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib

def kmeans(X, clusters, archive=None):

    # 构造K均值函数
    kmeans = KMeans(n_clusters=clusters).fit(X)
    # 获取分类
    classes = kmeans.labels_.reshape(-1,1)
    # 添加到输入矩阵
    X = np.hstack((X, kmeans.labels_.reshape(-1,1)))
    if archive is not None:
        joblib.dump(kmeans, archive+'/kmeans/model.joblib')

    return X



def find_optimal_k(X):

    print('Trying K = 1 through 21...')
    results = {k: KMeans(n_clusters=k).fit(X).inertia_ for k in range(1, 21)}
    print('Plotting...')
    sns.lineplot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel('K-value')
    plt.ylabel('Distortion')
    plt.title('Finding Optimal K-value')
    plt.grid()
    plt.show()