import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Dataset
# -----------------------------
X = np.array([1, 2, 3, 4])
y = np.array([2, 2.8, 3.6, 4.5])

# -----------------------------
# 2. Initialize Parameters
# -----------------------------
theta0 = 0.0  # Intercept
theta1 = 0.0  # Slope
learning_rate = 0.01
epochs = 100
losses = []

# -----------------------------
# 3. Training using SGD
# -----------------------------
for epoch in range(epochs):
    # Loop over each data point (stochastic updates)
    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]

        # Prediction
        y_pred = theta0 + theta1 * x_i

        # Error
        error = y_pred - y_i

        # Update parameters (SGD step)
        theta0 -= learning_rate * error
        theta1 -= learning_rate * error * x_i

    # Compute Mean Squared Error after each epoch
    mse = np.mean((theta0 + theta1 * X - y) ** 2)
    losses.append(mse)

# -----------------------------
# 4. Results and Visualization
# -----------------------------
y_pred_final = theta0 + theta1 * X

# Plot Loss Curve
plt.subplot(1, 2, 1)
plt.plot(range(epochs), losses, color='purple')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Loss Reduction Over Epochs")

# Plot Regression Fit
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred_final, color='red', label='Fitted Line')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression (SGD)")
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# 5. Final Parameters
# -----------------------------
print(f"Final Intercept (theta0): {theta0:.4f}")
print(f"Final Slope (theta1): {theta1:.4f}")
