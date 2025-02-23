import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_random_state, resample
from joblib import Parallel, delayed
import math



###############################################################################
# Optimized Bagging Classifier
###############################################################################
class OptimizedBaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    An optimized Bagging classifier that trains an ensemble of base estimators
    in parallel using bootstrap samples.

    Key enhancements:
      1. Parallel training using joblib.Parallel to accelerate model deployment.
      2. Bootstrap sampling for robust variance reduction and improved generalization.
      3. Flexibility to choose any base estimator and customize ensemble parameters.

    Parameters:
    -----------
    base_estimator : object
        The base estimator to fit on random subsets of the dataset.
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    max_samples : float or int, default=1.0
        If float, then draw max_samples * X.shape[0] samples. If int, then draw that many samples.
    n_jobs : int, default=1
        The number of jobs to run in parallel for fitting the base estimators.
    random_state : int or None, default=None
        Controls the random resampling of the original dataset.
    """

    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0, n_jobs=1, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _fit_single_estimator(self, X, y, random_state):
        """Helper function to fit one estimator on a bootstrap sample."""
        # Determine sample size
        n_samples = X.shape[0]
        if isinstance(self.max_samples, float):
            sample_size = int(self.max_samples * n_samples)
        else:
            sample_size = self.max_samples

        # Create a bootstrap sample with replacement
        rs = check_random_state(random_state)
        indices = rs.choice(n_samples, size=sample_size, replace=True)
        X_sample, y_sample = X[indices], y[indices]

        estimator = clone(self.base_estimator)
        estimator.fit(X_sample, y_sample)
        return estimator

    def fit(self, X, y):
        """Fit the bagging ensemble."""
        self.estimators_ = []
        rs = check_random_state(self.random_state)

        # Parallel fitting of base estimators
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_estimator)(X, y, rs.randint(0, 10000))
            for _ in range(self.n_estimators)
        )
        return self

    def predict(self, X):
        """Predict class labels for X by aggregating predictions from all base estimators."""
        # Collect predictions from each estimator
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        # Majority vote
        maj_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return maj_vote

    def predict_proba(self, X):
        """Predict class probabilities for X by averaging probabilities from all base estimators."""
        probas = np.array([estimator.predict_proba(X) for estimator in self.estimators_])
        avg_probas = np.mean(probas, axis=0)
        return avg_probas


###############################################################################
# Optimized Boosting Classifier (AdaBoost-like)
###############################################################################
class OptimizedBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    An optimized Boosting classifier implementing an AdaBoost-like algorithm.

    Key enhancements:
      1. Iterative weight adaptation to focus on hard-to-classify samples,
         which improves overall accuracy.
      2. Flexibility in base estimator choice and integration of a learning rate.
      3. Simple yet effective boosting mechanism that is well-suited for complex deployments.

    Parameters:
    -----------
    base_estimator : object
        The base estimator to be boosted.
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
    learning_rate : float, default=1.0
        Weight applied to each classifier's vote. A higher learning rate increases the influence
        of each classifier.
    random_state : int or None, default=None
        Controls the random seed given to each base estimator.
    """

    def __init__(self, base_estimator, n_estimators=50, learning_rate=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the boosting ensemble."""
        self.estimators_ = []
        self.estimator_weights_ = []

        n_samples = X.shape[0]
        # Initialize sample weights uniformly
        sample_weights = np.full(n_samples, 1 / n_samples)
        rs = check_random_state(self.random_state)

        # Assume binary classification with labels {0, 1}
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("This implementation supports binary classification only.")

        for m in range(self.n_estimators):
            # Clone base estimator for each iteration
            estimator = clone(self.base_estimator)
            estimator.set_params(random_state=rs.randint(0, 10000) if hasattr(estimator, 'random_state') else None)
            # Fit estimator with the current sample weights
            estimator.fit(X, y, sample_weight=sample_weights)
            y_pred = estimator.predict(X)

            # Compute weighted error rate
            incorrect = (y_pred != y)
            error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)

            # If error is 0 or greater than 0.5, stop early
            if error <= 0:
                alpha = 1e10
                self.estimators_.append(estimator)
                self.estimator_weights_.append(alpha)
                break
            elif error >= 0.5:
                break

            # Compute estimator weight (alpha) using AdaBoost formula
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)

            # Update sample weights: increase for misclassified, decrease for correctly classified
            sample_weights *= np.exp(-alpha * ((y == y_pred) * 2 - 1))
            sample_weights /= np.sum(sample_weights)  # Normalize

        return self

    def predict(self, X):
        """Predict class labels for X using the weighted majority vote."""
        # Sum the weighted predictions (convert class labels to -1 and +1)
        agg = np.zeros(X.shape[0])
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            prediction = estimator.predict(X)
            # Map original classes to -1 and +1 for voting
            mapped_pred = np.where(prediction == self.classes_[0], -1, 1)
            agg += weight * mapped_pred
        # Final prediction: choose class based on sign
        final_pred = np.where(agg < 0, self.classes_[0], self.classes_[1])
        return final_pred

    def predict_proba(self, X):
        """Estimate class probabilities for X.

        This method approximates probabilities based on the weighted sum of votes.
        """
        agg = np.zeros(X.shape[0])
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            prediction = estimator.predict(X)
            mapped_pred = np.where(prediction == self.classes_[0], -1, 1)
            agg += weight * mapped_pred
        # Transform the aggregated score to probability (using a sigmoid-like mapping)
        prob_positive = 1.0 / (1 + np.exp(-agg))
        prob_negative = 1 - prob_positive
        return np.vstack([prob_negative, prob_positive]).T


###############################################################################
# Example usage:
###############################################################################
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    # Create a toy dataset
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15, random_state=42)

    # Define a base estimator
    base_est = DecisionTreeClassifier(max_depth=3)

    # Instantiate and fit the optimized bagging classifier
    bagging_clf = OptimizedBaggingClassifier(base_estimator=base_est, n_estimators=20, max_samples=0.8, n_jobs=-1,
                                             random_state=42)
    bagging_clf.fit(X, y)
    y_pred_bag = bagging_clf.predict(X)
    print("Bagging Accuracy:", accuracy_score(y, y_pred_bag))

    # Instantiate and fit the optimized boosting classifier
    boosting_clf = OptimizedBoostingClassifier(base_estimator=base_est, n_estimators=50, learning_rate=1.0,
                                               random_state=42)
    boosting_clf.fit(X, y)
    y_pred_boost = boosting_clf.predict(X)
    print("Boosting Accuracy:", accuracy_score(y, y_pred_boost))
