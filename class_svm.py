from sklearn.svm import SVC

# SVM = Support Vector Machine

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38],
	[154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
	[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
	
y = ['male', 'female', 'female', 'female', 'male', 'male',
	'male', 'female', 'male', 'female', 'male']

clf1 = SVC()
clf1.fit(X, y)

print("Résultat clf1 avec SVC : ")
print(clf1.predict([190, 70, 43]))

from sklearn.svm import NuSVC

clf2 = NuSVC()
clf2.fit(X, y)

print ("Résultat clf2 avec NuSVC : ")
print (clf2.predict([190, 70, 43]))