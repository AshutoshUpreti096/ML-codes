import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]  # Selecting a subset of features
y = iris.target  # We'll treat one of the numerical features as the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Scatter plots for the selected features
plt.figure(figsize=(12, 4))

# Scatter plot for the 'sepal length (cm)' feature
plt.subplot(131)
plt.scatter(X_test['sepal length (cm)'], y_test, color='blue', label='Actual')
plt.scatter(X_test['sepal length (cm)'], y_pred, color='red', label='Predicted')
plt.xlabel('sepal length (cm)')
plt.ylabel('Target')
plt.title('Actual vs. Predicted (sepal length)')

# Scatter plot for the 'sepal width (cm)' feature
plt.subplot(132)
plt.scatter(X_test['sepal width (cm)'], y_test, color='blue', label='Actual')
plt.scatter(X_test['sepal width (cm)'], y_pred, color='red', label='Predicted')
plt.xlabel('sepal width (cm)')
plt.ylabel('Target')
plt.title('Actual vs. Predicted (sepal width)')

# Scatter plot for the 'petal length (cm)' feature
plt.subplot(133)
plt.scatter(X_test['petal length (cm)'], y_test, color='blue', label='Actual')
plt.scatter(X_test['petal length (cm)'], y_pred, color='red', label='Predicted')
plt.xlabel('petal length (cm)')
plt.ylabel('Target')
plt.title('Actual vs. Predicted (petal length)')

plt.tight_layout()
plt.show()

