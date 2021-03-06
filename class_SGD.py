from sklearn.linear_model import SGDClassifier

# SGD = Stochastic Gradient Descent

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38],
	[154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
	[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
	
y = ['male', 'female', 'female', 'female', 'male', 'male',
	'male', 'female', 'male', 'female', 'male']

clf = SGDClassifier()
clf.fit(X, y)

print(clf.predict([190, 70, 43]))