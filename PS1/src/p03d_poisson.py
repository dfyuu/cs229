import numpy as np
import util

from linear_model import LinearModel 
import matplotlib.pyplot as plt

def plot(y_label, y_pred, title):
    plt.plot(y_label, 'go', label='label')
    plt.plot(y_pred, 'rx', label='prediction')
    plt.suptitle(title, fontsize=12)
    plt.legend(loc='upper left') 


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset('data/ds4_train.csv', add_intercept=True)
    # The line below is the original one from Stanford. It does not include the intercept, but this should be added.
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_valid, y_valid = util.load_dataset('data/ds4_valid.csv', add_intercept=True)    
    
    clf = PoissonRegression(step_size=2e-7)
    clf.fit(x_train, y_train)
    
    y_train_pred = clf.predict(x_train)
    plt.figure(1)
    plot(y_train, y_train_pred, 'Training Set')
    
    y_valid_pred = clf.predict(x_valid)
    plt.figure(2)
    plot(y_valid, y_valid_pred, 'Validation Set')
    
    plt.show()
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def h(self, theta, x):
        return np.exp(x @ theta)

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        
        if self.theta is None:
            theta = np.zeros(n)
        else:
            theta = self.theta 
            
        def next_step(theta):
            return self.step_size / m * x.T @ (y - self.h(theta, x)) 
        
        step = next_step(theta)
        while np.linalg.norm(step, 1) >= self.eps:
            theta += step
            step = next_step(theta)
        
        self.theta = theta 
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return self.h(self.theta, x)
        # *** END CODE HERE ***

