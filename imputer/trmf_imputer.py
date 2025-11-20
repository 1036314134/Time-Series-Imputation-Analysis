import time
import numpy as np


def trmf_impute(incomp_data, lags=[], K=-1, lambda_f=0.1, lambda_x=0.1, lambda_w=0.1, eta=0.1, alpha=1000, max_iter=100, logs=True, verbose=True):
    """
    Perform imputation using the Temporal Regularized Matrix Factorization (TRMF) algorithm.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    lags : array-like, optional
        Set of lag indices to use in model.
    K : int, optional
        Length of latent embedding dimension
    lambda_f : float, optional
        Regularization parameter used for matrix F.
    lambda_x : float, optional
        Regularization parameter used for matrix X.
    lambda_w : float, optional
        Regularization parameter used for matrix W.
    alpha : float, optional
        Regularization parameter used for make the sum of lag coefficient close to 1.
        That helps to avoid big deviations when forecasting.
    eta : float, optional
        Regularization parameter used for X when undercovering autoregressive dependencies.
    max_iter : int, optional
        Number of iterations of updating matrices F, X and W.
    verbose : bool, optional
        Whether to display the contamination information (default is True).
    logs : bool, optional
        Whether to log the execution time (default is True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.

    Notes
    -----
    The MRNN algorithm is a machine learning-based approach for time series imputation, where missing values are recovered using a recurrent neural network structure.

    This function logs the total execution time if `logs` is set to True.

    Example
    -------
        >>> recov_data = trmf(incomp_data, lags=[], K=-1, lambda_f=1.0, lambda_x=1.0, lambda_w=1.0, eta=1.0, alpha=1000.0, max_iter=100)
        >>> print(recov_data)

    References
    ----------
    H.-F. Yu, N. Rao, and I. S. Dhillon, "Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction," in *Advances in Neural Information Processing Systems*, vol. 29, 2016. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2016/file/85422afb467e9456013a2a51d4dff702-Paper.pdf
    """
    start_time = time.time()  # Record start time

    recov_data = recoveryTRMF(data=incomp_data, lags=lags, K=K, lambda_f=lambda_f, lambda_x=lambda_x, lambda_w=lambda_w, eta=eta, alpha=alpha, max_iter=max_iter)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation trmf - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data


def recoveryTRMF(data, lags=[], K=-1, lambda_f=1.0, lambda_x=1.0, lambda_w=1.0, eta=1.0, alpha=1000.0, max_iter=100):
    """Temporal Regularized Matrix Factorization : https://github.com/SemenovAlex/trmf

    Parameters
    ----------
    data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).
    lags : array-like, optional
        Set of lag indices to use in model.
    K : int, optional
        Length of latent embedding dimension
    lambda_f : float, optional
        Regularization parameter used for matrix F.
    lambda_x : float, optional
        Regularization parameter used for matrix X.
    lambda_w : float, optional
        Regularization parameter used for matrix W.
    alpha : float, optional
        Regularization parameter used for make the sum of lag coefficient close to 1.
        That helps to avoid big deviations when forecasting.
    eta : float, optional
        Regularization parameter used for X when undercovering autoregressive dependencies.
    max_iter : int, optional
        Number of iterations of updating matrices F, X and W.

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values recovered.
    """
    if not lags:
        lags = list(range(1, 11))

    if K == -1:
        n = data.shape[0]
        K = n + 1

    print("(IMPUTATION) TRMF: Matrix Shape: (", data.shape[0], ", ", data.shape[1], ") for lags ", lags, ", K ", K,
          ", lambda_f ", lambda_f, " lambda_x", lambda_x, ", lambda_w ", lambda_w, ", eta ", eta, " alpha", alpha,
          ", and max_iter ", max_iter, ")...")

    incomp_data = np.copy(data)  # Copy data to avoid modifying original

    model = trmf(lags, K, lambda_f, lambda_x, lambda_w, alpha, eta, max_iter, F_step=1e-6, X_step=1e-6, W_step=1e-6)
    model.fit(incomp_data)
    data_imputed = model.impute_missings()
    data_imputed = np.array(data_imputed)

    return data_imputed


"""
Temporal Regularized Matrix Factorization
"""

# Author: Alexander Semenov <alexander.s.semenov@yandex.ru>
# https://github.com/SemenovAlex/trmf

class trmf:
    """Temporal Regularized Matrix Factorization.

    Parameters
    ----------
    lags : array-like, shape (n_lags,)
        Set of lag indices to use in model.
    K : int
        Length of latent embedding dimension
    lambda_f : float
        Regularization parameter used for matrix F.
    lambda_x : float
        Regularization parameter used for matrix X.
    lambda_w : float
        Regularization parameter used for matrix W.
    alpha : float
        Regularization parameter used for make the sum of lag coefficient close to 1.
        That helps to avoid big deviations when forecasting.
    eta : float
        Regularization parameter used for X when undercovering autoregressive dependencies.
    max_iter : int
        Number of iterations of updating matrices F, X and W.
    F_step : float
        Step of gradient descent when updating matrix F.
    X_step : float
        Step of gradient descent when updating matrix X.
    W_step : float
        Step of gradient descent when updating matrix W.

    Attributes
    ----------
    F : ndarray, shape (n_timeseries, K)
        Latent embedding of timeseries.
    X : ndarray, shape (K, n_timepoints)
        Latent embedding of timepoints.
    W : ndarray, shape (K, n_lags)
        Matrix of autoregressive coefficients.
    """
    def __init__(self, lags, K, lambda_f, lambda_x, lambda_w, alpha, eta, max_iter,
                 F_step=0.00001, X_step=0.00001, W_step=0.00001):
        self.lags = lags
        self.L = len(lags)
        self.K = K
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.lambda_w = lambda_w
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.F_step = F_step
        self.X_step = X_step
        self.W_step = W_step

        self.W = None
        self.F = None
        self.X = None

    def fit(self, train, resume=False):
        """Fit the TRMF model according to the given training data.

        Model fits through sequential updating three matrices:
            -   matrix self.F;
            -   matrix self.X;
            -   matrix self.W.

        Each matrix updated with gradient descent.

        Parameters
        ----------
        train : ndarray, shape (n_timeseries, n_timepoints)
            Training data.
        resume : bool
            Used to continue fitting.

        Returns
        -------
        self : object
            Returns self.
        """
        if not resume:
            self.Y = train
            mask = np.array((~np.isnan(self.Y)).astype(int))
            self.mask = mask
            self.Y[self.mask == 0] = 0.
            assert not np.isnan(self.Y).any(), "Input contains NaN"
            assert not np.isinf(self.Y).any(), "Input contains Inf"
            self.N, self.T = self.Y.shape
            self.W = np.random.randn(self.K, self.L) / self.L
            self.F = np.random.randn(self.N, self.K)
            self.X = np.random.randn(self.K, self.T)

        for _ in range(self.max_iter):
            self._update_F(step=self.F_step)
            self._update_X(step=self.X_step)
            self._update_W(step=self.W_step)

    def predict(self, h):
        """Predict each of timeseries h timepoints ahead.

        Model evaluates matrix X with the help of matrix W,
        then it evaluates prediction by multiplying it by F.

        Parameters
        ----------
        h : int
            Number of timepoints to forecast.

        Returns
        -------
        preds : ndarray, shape (n_timeseries, T)
            Predictions.
        """
        X_preds = self._predict_X(h)
        return np.dot(self.F, X_preds)

    def _predict_X(self, h):
        """Predict X h timepoints ahead.

        Evaluates matrix X with the help of matrix W.

        Parameters
        ----------
        h : int
            Number of timepoints to forecast.

        Returns
        -------
        X_preds : ndarray, shape (self.K, h)
            Predictions of timepoints latent embeddings.
        """
        X_preds = np.zeros((self.K, h))
        X_adjusted = np.hstack([self.X, X_preds])
        for t in range(self.T, self.T + h):
            for l in range(self.L):
                lag = self.lags[l]
                X_adjusted[:, t] += X_adjusted[:, t - lag] * self.W[:, l]
        return X_adjusted[:, self.T:]

    def impute_missings(self):
        """Impute each missing element in timeseries.

        Model uses matrix X and F to get all missing elements.

        Parameters
        ----------

        Returns
        -------
        data : ndarray, shape (n_timeseries, T)
            Predictions.
        """
        data = self.Y
        data[self.mask == 0] = np.dot(self.F, self.X)[self.mask == 0]

        return data

    def _update_F(self, step, n_iter=1):
        """Gradient descent of matrix F.

        n_iter steps of gradient descent of matrix F.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.
        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """
        for _ in range(n_iter):
            self.F -= step * self._grad_F()

    def _update_X(self, step, n_iter=1):
        """Gradient descent of matrix X.

        n_iter steps of gradient descent of matrix X.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.
        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """
        for _ in range(n_iter):
            self.X -= step * self._grad_X()

    def _update_W(self, step, n_iter=1):
        """Gradient descent of matrix W.

        n_iter steps of gradient descent of matrix W.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.
        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.W -= step * self._grad_W()

    def _grad_F(self):
        """Gradient of matrix F.

        Evaluating gradient of matrix F.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """
        return - 2 * np.dot((self.Y - np.dot(self.F, self.X)) * self.mask, self.X.T) + 2 * self.lambda_f * self.F

    def _grad_X(self):
        """Gradient of matrix X.

        Evaluating gradient of matrix X.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (np.roll(self.X, -lag, axis=1) - X_l) * W_l
            z_2[:, -lag:] = 0.

        grad_T_x = z_1 + z_2
        return - 2 * np.dot(self.F.T, self.mask * (
                    self.Y - np.dot(self.F, self.X))) + self.lambda_x * grad_T_x + self.eta * self.X

    def _grad_W(self):
        """Gradient of matrix W.

        Evaluating gradient of matrix W.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        grad = np.zeros((self.K, self.L))
        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (z_1 * np.roll(self.X, lag, axis=1)).sum(axis=1)
            grad[:, l] = z_2
        return grad + self.W * 2 * self.lambda_w / self.lambda_x - \
            self.alpha * 2 * (1 - self.W.sum(axis=1)).repeat(self.L).reshape(self.W.shape)