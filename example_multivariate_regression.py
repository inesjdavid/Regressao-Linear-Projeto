"""
Multivariate Linear Regression Example
Demonstrates training, prediction, and visualization of the model
"""

import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

#Data Generation
np.random.seed(42)

x1 = np.random.uniform(0, 35, 200)
x2 = np.random.uniform(-5, 17, 200)
X = np.column_stack([x1, x2])

Y = 2.0 * x1 - 1.5 * x2 + 4.5 + np.random.normal(0, 0.5, 200)

#Model Training
model = LinearRegression(
    learning_rate=0.005,
    n_iterations_max=30000,
    tol=1e-6,
    min_loss=2)

model.fit(X, Y)
print(model)

#Prediction for new data points
X_new = np.array([
    [2, -8],
    [7, 4.5],
    [5, 1]
])
Y_pred_new = model.predict(X_new)

#Output Predictions + Stopping Criteria
print("\nPredictions for new input values:")
for x, y in zip(X_new, Y_pred_new):
    print(f"  For X = {x}, predicted Y = {y:.2f}")
print()
print(f"Stop reason: {model.stop_reason_}")

#Plot Regression Line
r2 = model.score(X, Y)
Y_pred_train = model.predict(X)
plt.figure(figsize=(10, 6))
plt.gca().set_axisbelow(True)

#Training data
plt.scatter(
    Y,
    Y_pred_train,
    color="darkorange",
    alpha=0.6,
    label="Training data")

#New Predictions
min_val = min(Y.min(), Y_pred_train.min())
max_val = max(Y.max(), Y_pred_train.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    label="Perfect fit")

#Labels and formatting
plt.xlabel("True Y")
plt.ylabel("Predicted Y")
plt.title("Multivariate Linear Regression: True vs Predicted")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.grid(True)

#Model Information
info_text = (
    f"Y = {model.coef_[0]:.2f}·X₁ "
    f"+ {model.coef_[1]:.2f}·X₂ "
    f"+ {model.intercept_:.2f}\n"
    f"$R^2$ = {r2:.3f}")

plt.text(
    1.04, 0.78,
    info_text,
    transform=plt.gca().transAxes,
    ha="left",
    va="top",
    bbox=dict(boxstyle="round", edgecolor="deeppink", facecolor="white"))

plt.subplots_adjust(right=0.72)
plt.savefig("example_multivariate_regression.png", dpi=150)
plt.close()

#Plot Loss Evolution
initial_loss = model.loss_history_[0]
final_loss = model.loss_history_[-1]

plt.figure(figsize=(8, 4))
plt.plot(model.loss_history_, color="deeppink")
plt.yscale("log")

plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Training Error Evolution")
plt.grid(True)

#Loss Information
mse_text = (
    f"Initial MSE: {initial_loss:.4f}\n"
    f"Final MSE: {final_loss:.4f}")

plt.text(
    0.01, 1.10,
    mse_text,
    transform=plt.gca().transAxes,
    ha="left",
    bbox=dict(boxstyle="round", edgecolor="deeppink", facecolor="white"))

plt.subplots_adjust(top=0.80)
plt.savefig("example_multivariate_loss_curve.png", dpi=150)
plt.close()
