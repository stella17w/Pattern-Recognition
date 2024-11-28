from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import KMeans
from sklearn import metrics  

#get training data

#support vector machine
alg_svm=svm.LinearSvC()

#random forest with 2000 trees
alg_rf=RandomForestClassifier(n_estimators = 2000)  

#k-nearest neighbor
alg_kn=NearestNeighbors(n_neighbors=2, algorithm='auto')

#k-means clustering with 2 clusters
alg_km=KMeans(n_clusters = 2, random_state = 0, n_init='auto')