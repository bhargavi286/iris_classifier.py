# iris_classifier.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Dataset load
iris = load_iris()
X = iris.data   # features (measurements)
y = iris.target # labels (species)

# 2. Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train a model (Logistic Regression)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Predict on test data
y_pred = model.predict(X_test)

# 5. Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Example prediction
sample = [[5.1, 3.5, 1.4, 0.2]]  # sepal length, sepal width, petal length, petal width
prediction = model.predict(sample)
print("Predicted species:", iris.target_names[prediction][0])

