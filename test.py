import numpy as np
import matplotlib.pyplot as plt

# Data points (Table 1: Data Set)
x_i = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y_i = [20.26, 5.61, 3.14, -30.00, -40.00, -8.13, -11.73, -16.08, -19.95, -24.03]

# Create design matrix X (with a column of 1s for intercept) and target vector Y
X = np.array([[1, x] for x in x_i])  
Y = np.array([[y] for y in y_i])      

# print("X =", X)
# print("Y =", Y)

# Apply OLS formula: w = (X^T X)^(-1) X^T Y
X_T = X.T
w_OLS = np.linalg.inv(X_T @ X) @ X_T @ Y

# Extract coefficients
intercept = w_OLS[0, 0]
slope = w_OLS[1, 0]

# Predict y values using the regression model
x_range = np.linspace(min(x_i), max(x_i), 100)
y_predicted = intercept + slope * x_range

# Step 4: Plot original data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(x_i, y_i, color='blue', label='Data points')
plt.plot(x_range, y_predicted, color='red', label=f'Regression line: y = {intercept:.3f} + {slope:.3f}x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

print(f"Regression Line Equation: y = {intercept:.3f} + {slope:.3f}x")