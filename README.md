# Linear Regression with Gradient Descent
This project implements a Linear Regression model using Gradient Descent.

## Model Overview
- **Gradient Descent Optimization** - Iteratively minimizes Mean Squared Error (MSE)

- **Stopping Criteria**:
  - Maximum number of iterations
  - Minimum error threshold
  - Minimum improvement between iterations

- **Features** - Can handle both univariate and multivariate regression

- **R² Score** - Evaluated model performance

- **Parameters:**
- `learning_rate` (float): Step size for gradient descent
- `n_iterations_max` (int): Maximum number of training iterations
- `tol` (float, optional): Stops if improvement < tol
- `min_loss` (float, optional): Stops if MSE < min_loss

- **Attributes:**
- `coef_` (array): Learned coefficients
- `intercept_` (float): Learned bias term
- `loss_history_` (list): MSE at each iteration
- `n_iter_` (int): Number of iterations performed
- `stop_reason_` (str): Why training stopped

- **Methods:**
- `fit(X, y)`: Trains the model
- `predict(X)`: Makes predictions
- `score(X, y)`: Calculates the R squared score

## Installation
```bash
git clone https://github.com/yourusername/linear-regression.git
cd linear-regression
```
Requirements: `numpy`, `matplotlib` (for examples)

## Example Application
### Univariate Regression
An example application of the model is provided where we train a model on synthetic data with noise:

![Linear Regression Example](example_linear_regression.png)

### Training Convergence

![Training Error Evolution](example_loss_curve.png)


## Author
Inês David - Linear Regression Progamming Project