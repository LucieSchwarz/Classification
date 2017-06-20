from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np

X = np.array([[181, 80, 44], [177, 70, 43], [160, 60, 38],
	[154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
	[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]])


y = ['male', 'female', 'female', 'female', 'male', 'male',
	'male', 'female', 'male', 'female', 'male']

clf = NearestCentroid()
clf.fit(X, y)

print (clf.predict([190, 70, 43]))

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, y)

print (neigh.predict_proba([190, 70, 43]))

# Ne donne res male qu'a partir de 5 voisins ..