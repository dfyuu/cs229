import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    
    clf.fit(x_train, y_train)
    
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    preds = clf.predict(x_eval)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        
        if self.theta is None:
            self.theta = np.zeros(n) 
        
        for i in range(self.max_iter):
            theta_old = np.copy(self.theta)
            
            z = x @ self.theta
            h = 1 / (1 + np.exp(-z))
            grad = (1 / m) * x.T @ (h - y)
            d_diag = h * (1 - h)
            D = np.diag(d_diag)
            H = (1 / m) * x.T @ D @ x
            H_inv = np.linalg.inv(H)
            self.theta = self.theta - H_inv @ grad
            
            if np.linalg.norm(self.theta - theta_old, ord = 1) < self.eps:
                break
            
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = x @ self.theta
        h = 1 / (1 + np.exp(-z))
        
        preds = (h >= 0.5).astype(int)
        
        return preds
        # *** END CODE HERE ***
