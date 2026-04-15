from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the data
# This dataset contains measurements of 150 iris flowers
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Choose a Model (Random Forest is great for beginners)
model = RandomForestClassifier()

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions and check accuracy
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 6. Predict a single "mystery" flower
# Measurements: [sepal length, sepal width, petal length, petal width]
mystery_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(mystery_flower)
print(f"The mystery flower is predicted to be: {iris.target_names[prediction][0]}")