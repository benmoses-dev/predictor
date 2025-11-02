from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge, TheilSenRegressor
import numpy as np
import numpy.fft as fft
from scipy.stats import norm
from scipy.signal import detrend


class Predictor:
    """
    A hybrid forecasting model that combines polynomial regression and
    Bayesian ridge regression for predicting the next value in a series,
    and combines simple frequency analysis and robust linear regression for
    predicting multiple future values from a univariate time series.

    Use 'predict_next_value' for a single prediction and
    'generate_next_points' for multiple value extrapolation.

    Note: This is currently a proof of concept. I have added comments to
    explain the process throughout - any ideas or alterations, feel free
    to get in touch!
    """

    def __init__(
        self,
        data,
        alpha=0.3,
        poly_deg=2,
        window_size=3,
        c=2.0,
        ci=0.95
    ):
        """
        Initialise parameters using the given time series.

        Use a smaller polynomial degree (1 or 2) unless there is a lot of data.

        Parameters:
            data (list): Input time series data.
            alpha (float): Smoothing factor.
            poly_deg (int): Degree of the polynomial used for regression.
            window_size (int): Rolling window size for slope estimation.
            c (float): Constant controlling adaptive weight scaling.
            ci (float): Confidence interval level for Bayesian prediction.
        """
        self.alpha = alpha
        self.poly_deg = poly_deg
        self.window_size = window_size
        self.c = c
        self.ci = ci

        # Derived data for numpy vector arithmetic.
        self.data = np.array(data, dtype=float)
        self.n = len(self.data)
        self.x = np.arange(self.n).reshape(-1, 1)

        # Get the overall linear gradient and intercept for the time series.
        # Use Theil Sen to handle major outliers.
        theil_sen_regression = TheilSenRegressor().fit(self.x, self.data)
        self.gradient = float(theil_sen_regression.coef_[0])

        # Get the standard deviation for the time series to model the noise.
        # Todo: Should this use the whole dataset?
        self.noise_std = np.std(self.data[-self.window_size:])

        # Detrend and centre the data before performing frequency analysis.
        detrended = detrend(self.data)
        centred = detrended - np.mean(detrended)

        # Get the frequency spectrum and dominant freq from the FFT.
        transform = fft.rfft(centred)
        freqs = fft.rfftfreq(self.n)
        peak_idx = np.argmax(np.abs(transform))
        peak_ampl = freqs[peak_idx]

        """
        Store the peak amplitude and the dominant period from the spectrum.

        We only have the real-valued amplitude as we discounted
        the complex conjugate, so we need to multiply by 2 and then
        normalise for the number of samples.
        """
        self.amplitude = 2 * np.abs(transform[peak_idx]) / self.n
        self.period = 1 / peak_ampl if peak_ampl != 0 else None

    def smoothed_series(self):
        """
        Reduce short time volatility
        by weighting the previous value into the series at each step.
        """
        smoothed = [self.data[0]]
        for r in self.data[1:]:
            smoothed.append(self.alpha * r + (1 - self.alpha) * smoothed[-1])
        return np.array(smoothed)

    def poly_regression(self):
        """
        Fit a polynomial to the data and extrapolate the next value.
        Returns:
            next_value (float): Extrapolated next value.
        """
        if self.n <= self.poly_deg:
            return float(self.data[-1])
        # Todo: Use the polynomial API.
        coefficients = np.polyfit(
            np.arange(self.n),
            self.data,
            deg=self.poly_deg
        )
        poly_func = np.poly1d(coefficients)
        return float(poly_func(self.n))

    def linear_extrapolation(self):
        """
        Perform simple linear extrapolation using the last two data points.
        Returns:
            diff_last (float): Last slope estimate.
            next_value (float): Linearly extrapolated next value.
        """
        if self.n < 2:
            return 0.0, float(self.data[-1])
        diff_last = self.data[-1] - self.data[-2]
        return diff_last, self.data[-1] + diff_last

    def rolling_slope(self):
        """
        Compute a rolling slope over the last `window_size` points of the
        smoothed data to estimate the short-term trend.

        Returns None if the number of samples is less than the window size.
        """
        if self.n >= self.window_size:
            smoothed = self.smoothed_series()
            window_diff = smoothed[-1] - smoothed[-self.window_size]
            slope_roll = window_diff / (self.window_size - 1)
            return smoothed[-1] + slope_roll
        return None

    def adaptive_weights(self):
        """
        Dynamically adjust the weighting between local and global models
        based on recent volatility (standard deviation).
        """
        if self.n >= self.window_size:
            recent_data = self.data[-self.window_size:]
        else:
            recent_data = self.data
        recent_sigma = np.std(recent_data)
        weight_local = recent_sigma / (recent_sigma + self.c)
        weight_regression = 1 - weight_local
        return weight_local, weight_regression

    def deterministic_pred(self):
        """
        Combine multiple deterministic predictors using adaptive weighting:
        - Polynomial regression (global trend)
        - Linear extrapolation (local slope)
        - Rolling slope (short-term trend)
        """
        weight_local, weight_reg = self.adaptive_weights()
        weight_last = weight_local / 2
        weight_roll = weight_local - weight_last
        print("Weights:")
        print(weight_local, weight_last, weight_roll, weight_reg)
        # Get the predictions.
        pred_reg = self.poly_regression()
        slope_last, pred_last = self.linear_extrapolation()
        pred_roll = self.rolling_slope()
        if pred_roll is None:
            pred_roll = slope_last
        # Adjust the predictions using the weightings.
        adj_reg = pred_reg * weight_reg
        adj_last = pred_last * weight_last
        adj_roll = pred_roll * weight_roll
        pred_d = adj_reg + adj_last + adj_roll
        return float(pred_d)

    def bayesian_pred(self):
        """
        Fit a Bayesian Ridge regression model, including a confidence interval
        """
        poly = PolynomialFeatures(degree=self.poly_deg, include_bias=False)
        poly.fit(self.x)
        x_poly = poly.transform(self.x)
        model = BayesianRidge(compute_score=True, fit_intercept=True)
        model.fit(x_poly, self.data)
        # Predict the next point and compute confidence interval using the ppf
        x_next = np.array([[self.n]])
        x_next_poly = poly.transform(x_next)
        mean_pred, std_pred = model.predict(x_next_poly, return_std=True)
        z = norm.ppf(0.5 + self.ci / 2)  # Handy api (thanks python)
        lower = mean_pred - z * std_pred
        upper = mean_pred + z * std_pred
        return {
            "mean": float(mean_pred),
            "std": float(std_pred),
            "lower": float(lower),
            "upper": float(upper)
        }

    def predict_next_value(self):
        """
        Return both deterministic and Bayesian predictions for the next value.
        """
        return {
            "deterministic": self.deterministic_pred(),
            "bayesian": self.bayesian_pred()
        }

    def generate_next_points(self, steps=5):
        """
        Generate several future points using a heuristic model that
        combines trend, periodic oscillation, and random noise.

        Parameters:
            steps (int): Number of points to generate.

        Returns:
            list of floats: Simulated future data points.
        """
        new_points = []
        print('Gradient: ' + str(self.gradient))
        print('Period: ' + str(self.period))
        print('Amplitude: ' + str(self.amplitude))
        print('Std: ' + str(self.noise_std))
        for i in range(1, steps + 1):
            t = self.n + i - 1
            trend_part = self.data[-1] + self.gradient * i
            if self.period is not None:
                freq = t / self.period
                osc_part = self.amplitude * np.sin(2 * np.pi * freq)
            else:
                osc_part = 0.0
            noise_part = np.random.normal(scale=self.noise_std)
            new_point = trend_part + osc_part + noise_part
            new_points.append(new_point)
        return new_points
