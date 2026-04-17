import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Define the dataset manually 
# Format: [sepal length, sepal width, petal length, petal width]
# This is a small sample of the 150 original rows for brevity
X = np.array([
    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], # Setosa (0)
    [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], # Versicolor (1)
    [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1]  # Virginica (2)
])

# Labels: 0 = Setosa, 1 = Versicolor, 2 = Virginica
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# Map names for the final output
target_names = ['setosa', 'versicolor', 'virginica']

# 2. Split data 
# Note: With only 9 samples, we use a small test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Choose a Model
model = RandomForestClassifier(n_estimators=10) # Small forest for small data

# 4. Train the model
model.fit(X_train, y_train)
#

# 5. Make predictions and check accuracy
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 6. Predict a single "mystery" flower
# Using measurements that look like a Setosa
mystery_flower = [[5.0, 3.4, 1.5, 0.2]]
prediction = model.predict(mystery_flower)
print(f"The mystery flower is predicted to be: {target_names[prediction[0]]}")




''' from sklearn.datasets import load_iris
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
print(f"The mystery flower is predicted to be: {iris.target_names[prediction][0]}") '''