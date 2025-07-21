import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    
    clf.fit(x_train, y_train)
    
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    predictions = clf.predict(x_eval) 
    
    np.savetxt(pred_path, predictions, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n + 1)
        
        phi = (1 / m) * np.sum(y)
        mu_0 = np.mean(x[y == 0], axis = 0)
        mu_1 = np.mean(x[y == 1], axis = 0)
        
        mu_map = np.where(y.reshape(-1, 1) == 1, mu_1, mu_0)
        
        centered_x = x - mu_map
        sigma = (1 / m) * centered_x.T @ centered_x
        
        sigma_inv = np.linalg.inv(sigma)
        
        theta = sigma_inv @ (mu_1 - mu_0)
        theta_0 = (0.5) * mu_0.T @ sigma_inv @ mu_0 - (0.5) * mu_1.T @ sigma_inv @ mu_1 - np.log((1 - phi) / phi)
        
        self.theta[0] = theta_0
        self.theta[1:] = theta
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x_with_intercept = util.add_intercept(x)
        
        z = x_with_intercept @ self.theta
        
        predictions = (z > 0).astype(int)

        return predictions    
        # *** END CODE HERE ***
