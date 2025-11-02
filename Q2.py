import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([1, 2, 3, 4])
y = np.array([2, 2.8, 3.6, 4.5])

# Hyperparameters
learning_rate = 0.01
epochs = 100

# Initialize parameters
theta_0 = 0.0
theta_1 = 0.0
loss_history = []
n_samples = len(X)

# Training loop using Stochastic Gradient Descent (SGD)
for epoch in range(epochs):

    # Combine and shuffle data for stochastic updates
    data = np.c_[X, y]
    np.random.shuffle(data)

    X_shuffled = data[:, 0]
    y_shuffled = data[:, 1]

    epoch_loss_sum = 0

    # Iterate over each sample for an SGD update
    for i in range(n_samples):
        x_i = X_shuffled[i]
        y_i = y_shuffled[i]

        # Prediction
        y_pred = theta_0 + theta_1 * x_i

        # Error calculation
        error = y_pred - y_i

        # Gradient for the single sample (based on SSE derivative)
        grad_theta_0 = error
        grad_theta_1 = error * x_i

        # Parameter update
        theta_0 -= learning_rate * grad_theta_0
        theta_1 -= learning_rate * grad_theta_1

        # Sum of squared error for the epoch monitoring
        epoch_loss_sum += error ** 2

    # Mean Squared Error (MSE) for the current epoch
    mse = epoch_loss_sum / n_samples
    loss_history.append(mse)

# Final prediction line for plotting
final_predictions = theta_0 + theta_1 * X

# Plotting the results
plt.figure(figsize=(12, 5))

# Plot 1: Loss over Epochs
plt.subplot(1, 2, 1)
plt.plot(range(epochs), loss_history, label="MSE")
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Loss Reduction over Epochs (SGD)')
plt.legend()

# Plot 2: Linear Regression Fit
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, final_predictions, color='red', label='Regression Line (SGD)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Fit with SGD')
plt.legend()
plt.tight_layout()
plt.show()

# Final parameters output
print(f"Final theta_0 (Intercept): {theta_0:.4f}")
print(f"Final theta_1 (Coefficient): {theta_1:.4f}")