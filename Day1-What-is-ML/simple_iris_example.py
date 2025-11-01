# simple_iris_example.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1) Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2) Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 4) Predict and evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Simple Iris model accuracy: {acc:.2f}")
