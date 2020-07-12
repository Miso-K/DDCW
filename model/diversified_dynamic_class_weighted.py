import copy as cp
import numpy as np

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.lazy import KNNClassifier
from utils import diversity
import time

class DiversifiedDynamicClassWeightedClassifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Diversified Dynamic Class Weighted ensemble classifier.

    Parameters
    ----------
    min_estimators: int (default=5)
        Minimum number of estimators to hold.
    max_estimators: int (default=20)
        Maximum number of estimatorst to hold.
    base_estimators: List of StreamModel or sklearn.BaseEstimator (default=[NaiveBayes(), KNNClassifier(), HoeffdingTreeClassifier())
        Each member of the ensemble is an instance of the base estimator.
    period: int (default=100)
        Period between expert removal, creation, and weight update.
    alpha: float (default=0.02)
        Factor for which to decrease weights on experts lifetime
    beta: float (default=3)
        Factor for which to increase weights by.
    theta: float (default=0.02)
        Minimum fraction of weight per model.
    enable_diversity: bool (default=True)
        If true, calculate diversity of experts and weights update.

    Notes
    -----
    The diversified dynamic class weighted (DDCW) [1]_, uses five mechanisms to
    cope with concept drift: It trains online learners of the ensemble, it uses weights per class,
    it update class weights for those learners based on their performance diversity and time spend in ensemble,
    it removes them, also based on their performance, and it adds new experts based on the
    global performance of the ensemble.

    """

    class WeightedExpert:
        """
        Wrapper that includes an estimator and its class weights.

        Parameters
        ----------
        estimator: StreamModel or sklearn.BaseEstimator
            The estimator to wrap.
        weight: float
            The estimator's weight.
        num_classes: int
            The number of actual target classes
        """
        def __init__(self, estimator, weight, num_classes):
            self.estimator = estimator
            self.weight_class = np.full(num_classes, weight, dtype=float)
            self.lifetime = 0


    def __init__(self, min_estimators=5, max_estimators=20, base_estimators=[NaiveBayes(), KNNClassifier(), HoeffdingTreeClassifier()],
                 period=100, alpha=0.02, beta=3, theta=0.02, enable_diversity=True):
        """
        Creates a new instance of DiversifiedDynamicClassWeightedClassifier.
        """
        super().__init__()

        self.enable_diversity = enable_diversity
        self.min_estimators = min_estimators
        self.max_estimators = max_estimators
        self.base_estimators = base_estimators

        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.period = period

        self.p = -1

        self.n_estimators = max_estimators
        self.epochs = None
        self.num_classes = None
        self.experts = None
        self.div = []

        self.window_size = None
        self.X_batch = None
        self.y_batch = None
        self.y_batch_experts = None

        # custom measurements atributes
        self.custom_measurements = []
        self.custom_time = []

        self.reset()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fits the model on the supplied X and y matrices.

        Since it's an ensemble learner, if X and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the class labels of all samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class labels. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies depending on the base estimator.

        Returns
        -------
        DiversifiedDynamicClassWeightedClassifier
            self
        """
        for i in range(len(X)):
            start_time = time.time()
            self.fit_single_sample(
                X[i:i+1, :], y[i:i+1], classes, sample_weight
            )
            self.custom_time.append(time.time() - start_time)
        return self

    def predict(self, X):
        """ predict

        The predict function will take an average of the predictions of its
        learners, weighted by their respective class weights, and return the most
        likely class.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        """
        predictions_class = np.zeros((len(X), self.num_classes))

        for exp in self.experts:
            Y_hat = exp.estimator.predict(X)
            for i, y_hat in enumerate(Y_hat):
                predictions_class[i][y_hat] += exp.weight_class[y_hat]
        y_hat_final = np.argmax(predictions_class, axis=1)
        return y_hat_final


    def predict_proba(self, X):
        raise NotImplementedError

    def fit_single_sample(self, X, y, classes=None, sample_weight=None):
        """
        Fits a single sample of shape `X.shape=(1, n_attributes)` and `y.shape=(1)`

        Aggregates all experts' predictions, diminishes weight of experts whose
        predictions were wrong, and may create or remove experts every _period_
        samples.

        Finally, trains each individual expert on the provided data.


        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Features matrix used for partially updating the model.

        y: Array-like
            An array-like of all the class labels for the samples in X.

        classes: list
            List of all existing classes. This is an optional parameter.

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Applicability
            depends on the base estimator.

        """

        start_time = time.time()

        if self.p <= 0:
            for exp in self.experts:
                exp.estimator = self.train_model(cp.copy(exp.estimator), X, y, classes, sample_weight)


        N, D = X.shape
        self.window_size = self.period

        if self.p <= 0:
            self.X_batch = np.zeros((self.window_size, D), dtype=int)
            self.y_batch = np.zeros(self.window_size, dtype=int)
            self.y_batch_experts = np.zeros((len(self.experts), self.window_size), dtype=int)
            self.p = 0


        self.epochs += 1
        self.num_classes = max(
            len(classes) if classes is not None else 0,
            (int(np.max(y)) + 1), self.num_classes)


        predictions_class = np.zeros((self.num_classes,))
        sum_weight_class = np.zeros((self.num_classes,))

        weakest_expert_weight_class = 1 * self.num_classes
        weakest_expert_index = None


        for i, exp in enumerate(self.experts):

            y_hat = exp.estimator.predict(X)
            self.y_batch_experts[i] = y_hat

            if len(exp.weight_class) < self.num_classes:
                exp.weight_class = np.pad(exp.weight_class, (0,self.num_classes - len(exp.weight_class)), 'constant', constant_values=(1/self.n_estimators))

            predictions_class[y_hat] += exp.weight_class[y_hat]

            if np.any(y_hat == y) and (self.epochs % self.period == 0):
                exp.weight_class[y_hat] = exp.weight_class[y_hat] * self.beta


        self.X_batch[self.p] = X
        self.y_batch[self.p] = y
        self.p = self.p + 1

        if self.p >= self.window_size:
            if len(self.experts) > 1:
                self._calculate_diversity(self.y_batch_experts, self.y_batch)
            else:
                self.div = []

        if self.p >= self.window_size:
            for i, exp in enumerate(self.experts):
                exp.lifetime += 1

                exp.weight_class = exp.weight_class - (np.exp(self.alpha * exp.lifetime) - 1)/10

                if len(self.div) > 0 and self.enable_diversity:
                    exp.weight_class = exp.weight_class * (1 - self.div[i])

                exp.weight_class[exp.weight_class <= 0] = 0.001

                for j in range(len(exp.weight_class)):
                    sum_weight_class[j] += exp.weight_class[j]

                if sum(exp.weight_class) <= weakest_expert_weight_class:
                    weakest_expert_index = i
                    weakest_expert_weight_class = sum(exp.weight_class)


        y_hat_final = np.array([np.argmax(predictions_class)])

        if self.p >= self.window_size:
            self.p = 0
            self._normalize_weights_class(sum_weight_class)

            if np.any(y_hat_final != y):

                if len(self.experts) >= self.max_estimators:
                    self.experts.pop(weakest_expert_index)

                if len(self.experts) < self.max_estimators:
                    new_exp = self._construct_new_expert(len(self.experts))
                    self.experts.append(new_exp)

            self._remove_experts_class()

            if len(self.experts) < self.min_estimators:
                new_exp = self._construct_new_expert(len(self.experts))
                self.experts.append(new_exp)


        for exp in self.experts:
            exp.estimator = self.train_model(cp.copy(exp.estimator), X, y, classes, sample_weight)

        if self.p == 0:
            # save custom measurements
            data = {'id_period': self.epochs / self.period, 'n_experts': len(self.experts), 'diversity': np.mean(self.div), 'train_time': (start_time - time.time())}
            self.custom_measurements.append(data)

    def get_expert_predictions(self, X):
        """
        Returns predictions of each class for each expert.
        In shape: (n_experts, n_samples)
        """
        return [exp.estimator.predict(X) for exp in self.experts]

    def reset(self):
        """
        Reset this ensemble learner.
        """
        self.epochs = 0
        self.num_classes = 2    # Minimum of 2 classes
        self.experts = [
            self._construct_new_expert(1)
        ]

    def _normalize_weights_class(self, sum_weight_class):
        """
        Normalize the experts' weights such that the sum per class is 1.
        """

        for exp in self.experts:
            for i in range(len(exp.weight_class)):
                exp.weight_class[i] /= sum_weight_class[i]

    def _calculate_diversity(self, y_experts, y):
        """
        Calculate Q stat pairwise diversity in actual model
        """
        self.div = diversity.compute_pairwise_diversity(y, y_experts, diversity.Q_statistic)

    def _remove_experts_class(self):
        """
        Removes all experts whose score (sum weights per class) is lower than self.theta.
        """
        self.experts = [ex for ex in self.experts if sum(ex.weight_class) >= self.theta * self.num_classes]


    def _construct_new_expert(self, ln):
        """
        Constructs a new WeightedExpert randomly from list of provided base_estimators.
        """
        x = np.random.randint(0, len(self.base_estimators))
        weight = 1 if ln == 0 else 1 / ln,
        return self.WeightedExpert(cp.deepcopy(self.base_estimators[x]), weight, self.num_classes)

    @staticmethod
    def train_model(model, X, y, classes=None, sample_weight=None):
        """ Trains a model, taking care of the fact that either fit or partial_fit is implemented
        Parameters
        ----------
        model: StreamModel or sklearn.BaseEstimator
            The model to train
        X: numpy.ndarray of shape (n_samples, n_features)
            The data chunk
        y: numpy.array of shape (n_samples)
            The labels in the chunk
        classes: list or numpy.array
            The unique classes in the data chunk
        sample_weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.
        Returns
        -------
        StreamModel or sklearn.BaseEstimator
            The trained model
        """
        try:
            model.partial_fit(X, y, classes, sample_weight)
        except NotImplementedError:
            model.fit(X, y)
        return model