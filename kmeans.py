import sys
import math

def find_min(data, centroids, K, d):
  min_dist = get_euclidean_dist(data, centroids[0], d)
  min_cluster = 0
  for i in range(1, K):
    curr_dist = get_euclidean_dist(data, centroids[i], d)
    if curr_dist < min_dist:
      min_dist = curr_dist
      min_cluster = i
  return min_cluster

# calculate the euclidean distance between two vectors
def get_euclidean_dist(X,Y, d):
    sum = 0
    for i in range(d):
        sum += math.pow(X[i]-Y[i], 2)
    return math.sqrt(sum)

# assign a given data point to its closest cluster
def get_closest_cluster(X ,centroids, d):
  argminDistValues = []

  for centroid in centroids:
    d_X_centroid = get_euclidean_dist(X,centroid, d)
    argminDistValues.append(d_X_centroid)

  closestCluster = argminDistValues.index(min(argminDistValues))  
        
  return closestCluster

# calculate the centroid of a given cluster 
def calc_centroid(cluster, d):
  if not cluster:
    return [0] * d
  sum = [0 for j in range(d)]
  for i in range(d):
      for X in cluster:
        sum[i] += X[i]
  centroid = [x/len(cluster) for x in sum]
  return centroid

# validate the euclidean distance between the previous centroids and the new ones for every cluster
def check_euclidean_dist_for_every_centroid(prevC, centroids, K, d):
  for i in range(K):
    dist = get_euclidean_dist(prevC[i], centroids[i], d)
    if  dist >= 0.001:
      return True
  return False

def calc_centroids(K, matrix_data, iter=300):
  d = len(matrix_data[0])
  # init centroids as first K data points
  centroids = matrix_data[:K]
  # init prevCentroids as the max value in each centroid
  prevCentroids = [[max(centroids[i]) for j in range(d)] for i in range(K)]
    
  # create an empty clusters list
  clusters = [[] for i in range(K)]

  # the Algorithm
  i = 0
  while(check_euclidean_dist_for_every_centroid(prevCentroids, centroids, K, d) and (i < iter)):
         
    # assign every data point to its closest cluster
    for X in matrix_data:
      closest = get_closest_cluster(X, centroids, d)
      clusters[closest].append(X)
        
    # update the centroids
    for j in range(K):
      prevCentroids[j] = centroids[j]
      new_centroid = calc_centroid(clusters[j], d)
      centroids[j] = new_centroid
            
    # update i
    i+=1

    # empty clusters
    for j in range(K):
      clusters[j] = []

  return centroids