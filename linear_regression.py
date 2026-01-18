
import numpy as np

class LinearRegression:
    """
    Linear Regression Model (y = wX + b) with Gradient Descent as to minimize Mean Squared Error loss function
    
    Parameters:
    learning_rate: default=0.01
        Step size for gradient descent updates
    n_iterations_max: default=1000
        Maximum number of training iterations (stopping criteria 1)
    tol: optional
        Minimum improvement threshold between iterations (stopping criteria 2)
        Training stops if |loss[i-1] - loss[i]| < tol
    min_loss: optional
        Minimum acceptable loss value (stopping criteria 3)
        Training stops if MSE <= min_loss
    """
    
    def __init__(
        self,
        learning_rate=0.01,
        n_iterations_max=1000,
        tol=None,
        min_loss=None):
    
        #Hyperparameters
        self.learning_rate = learning_rate
        self.n_iterations_max = n_iterations_max
        self.tol = tol
        self.min_loss = min_loss

        #Model parameters from training
        self.coef_ = None        #weight
        self.intercept_ = None   #bias
        
        #Training history
        self.loss_history_ = []
        self.n_iter_ = 0
        self.stop_reason_ = None
        
        # Scaling parameters
        self.X_mean_ = None
        self.X_std_ = None

    def __repr__(self):
        """ Showcases the parameters of the Linear Regression Model """
        trained = self.coef_ is not None
        return (
            f"LinearRegression("f"learning_rate={self.learning_rate}, "f"n_iterations_max={self.n_iterations_max}, "f"trained={trained}, "f"n_iter={self.n_iter_})" )


    def _mean_squared_error(self, y_true, y_pred):
        """ Calculate Mean Squared Error """
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y):
        """ Model Training with gradient descent
        
        Parameters: 
        X (feature matrix) with shape [n_samples, n_features]
        Y (target values) with shape [n_samples,]
        """ 
        #Convert to numpy arrays
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        #Input validation
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix with the shape [n_samples, n_features]")
        if y.ndim != 1:
            raise ValueError("Y must be a 1D array of target values.")

        n_samples, n_features = X.shape
        
        #Scaling
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0)
        self.X_std_[self.X_std_ == 0] = 1.0  #as to avoid division by zero
        X_scaled = (X - self.X_mean_) / self.X_std_
        
        #Parameters initialization
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        #Reset Training
        self.loss_history_.clear()
        self.n_iter_ = 0
        self.stop_reason_ = None

        prev_loss = None

        #Loop Gradient Descent
        for i in range(self.n_iterations_max):  
            y_pred = X_scaled @ self.coef_ + self.intercept_

            loss = self._mean_squared_error(y, y_pred)
            self.loss_history_.append(loss)
            self.n_iter_ = i + 1

            #Stopping criterion 3: minimum loss threshold
            if self.min_loss is not None and loss <= self.min_loss:
                self.stop_reason_ = "Minimum error threshold reached"
                break

            #gradients
            grad_w = (-2 / n_samples) * (X_scaled.T @ (y - y_pred))
            grad_b = (-2 / n_samples) * np.sum(y - y_pred)

            #Parameters update
            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b

            #Stopping criterion 2: minimum improvement threshold
            if self.tol is not None and prev_loss is not None:
                if abs(prev_loss - loss) < self.tol:
                    self.stop_reason_ = "Minimum improvement threshold reached"
                    break

            prev_loss = loss

        # topping criterion 1: maximum iterations
        if self.stop_reason_ is None:
            self.stop_reason_ = "Maximum number of iterations reached"

        return self

    def predict(self, X):
        """ Prediction of output values with the training model """
        X = np.array(X, dtype=float)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix with the shape [n_samples, n_features]")
        
        X_scaled = (X - self.X_mean_) / self.X_std_
        
        return X_scaled @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Computes the coefficient of determination R squared
        """

        y = np.array(y, dtype=float)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot