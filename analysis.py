import sys
import kmeans
import symnmf
import numpy as np
from sklearn.metrics import silhouette_score

def handle_error():
    """
    handle error according to the instructions
    """
    print("An Error Has Occurred")
    exit()


def kmeans_centroids(k, x):
    """
    Calculate centroids using kmeans

    Parameters:
    x (List): A matrix of data
    k (int): number of clusters

    Returns:
    List: indices of the cluster for each data point in x
    """
    centroids = kmeans.calc_centroids(k, x)
    min_distances = [kmeans.find_min(row, centroids, k, len(x[0])) for row in x]
    return min_distances


def symnmf_centroids(k, x):
    """
    Calculate centroids using symnmf

    Parameters:
    x (List): A matrix of data
    k (int): number of clusters

    Returns:
    List: indices of the cluster for each data point in x
    """
    H = np.array(symnmf.symnmf(x, k))
    return H.argmax(axis=1)

def get_data(file_name):
    """
    read the data from the given filename

    Parameters:
    filename (string): the path to the file

    Returns:
    List: the data in the file
    """

    data = []
    with open(file_name, 'r') as file:
        for line in file:
            row = [float(num) for num in line.strip().split(',')]
            data.append(row)
    return data

if __name__ == "__main__":
    
    # check number of arguments
    if len(sys.argv) != 3:  
        handle_error()

    # Initialize arguments
    try:
        K = int(sys.argv[1])
        filename = sys.argv[2]
    except:
        handle_error()

    # check filename
    if not filename.endswith('.txt'):  
        handle_error()
    X = get_data(filename)

    # check number of clusters
    if(K >= len(X) or K <= 1):
        handle_error()

    kmeans_centroid = kmeans_centroids(K, X)
    symnmf_centroid = symnmf_centroids(K, X)  

    print("nmf: %.4f" % silhouette_score(X, symnmf_centroid))
    print("kmeans: %.4f" % silhouette_score(X, kmeans_centroid))
