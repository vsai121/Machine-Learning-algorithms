from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

# Create data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Instantiate
abc = AdaBoostClassifier()

# Fit
abc.fit(X_train, y_train)

# Predict
y_pred = abc.predict(X_test)

# Accuracy
accuracy_score(y_pred, y_test)


