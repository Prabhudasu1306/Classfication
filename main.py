import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from joblib import dump

# Load the dataset
df = pd.read_csv("Advertising.csv")
print(df.head())
print(df.info())
print(df.describe())

# Pairplot for visual exploration
sns.pairplot(df)
plt.show()

# Correlation matrix
print(df.corr())

# Check for null values
print(df.isnull().sum())

# Define features (X) and target (y)
X = df[["TV", "radio", "newspaper"]]
y = df["sales"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Arrays to store R² scores for different polynomial degrees
train_r2 = []
test_r2 = []

# Loop over different polynomial degrees (1 to 9)
for degree in range(1, 10):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Transform the training and testing data
    X_train_poly = pd.DataFrame(poly.fit_transform(X_train))
    X_test_poly = pd.DataFrame(poly.transform(X_test))
    
    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Store R² scores for both training and testing sets
    train_r2.append(model.score(X_train_poly, y_train))
    test_r2.append(model.score(X_test_poly, y_test))

# Plotting the R² scores for both training and testing sets
plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), train_r2, label='Train R²', marker='o')
plt.plot(range(1, 10), test_r2, label='Test R²', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('R² Score vs Polynomial Degree')
plt.legend()
plt.grid(True)
plt.show()

# Display the best polynomial degree based on test R² score
best_degree = np.argmax(test_r2) + 1  # +1 because index starts from 0
print(f"The best polynomial degree is: {best_degree}")
print(f"Train R² at best degree: {train_r2[best_degree-1]}")
print(f"Test R² at best degree: {test_r2[best_degree-1]}")

# Apply the best polynomial degree to the entire dataset
poly = PolynomialFeatures(degree=best_degree, include_bias=False)
X_poly = poly.fit_transform(X)

# Cross-validation with the best polynomial degree
model.fit(X_poly, y)  # Fit the model on the entire dataset
scores = cross_val_score(model, X_poly, y, cv=5)
print("Cross-validation scores with best polynomial degree:", scores)
print("Mean cross-validation score:", scores.mean())

# Predict sales for new input data
input_data = pd.DataFrame([[149000, 22000, 12000]], columns=["TV", "radio", "newspaper"])  # Ensure feature names match
transformed_data = poly.transform(input_data)  # Use the poly transformer
prediction = model.predict(transformed_data)  # Make prediction
print(f"Predicted sales for input {input_data}: {prediction}")

# Save the model and polynomial transformer
dump(model, "Advertising_poly_model.joblib")
dump(poly, "poly_transformer.joblib")
