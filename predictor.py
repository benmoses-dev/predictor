from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge
import numpy as np
import numpy.fft as fft
from scipy.stats import norm
from scipy.signal import detrend


class Predictor:
    def __init__(
        self,
        data,
        alpha=0.3,
        poly_deg=2,
        window_size=3,
        c=2.0,
        ci=0.95
    ):
        self.data = data
        self.alpha = alpha
        self.poly_deg = poly_deg
        self.window_size = window_size
        self.c = c
        self.ci = ci
        self.data = np.array(data, dtype=float)
        self.n = len(self.data)
        self.x = np.arange(self.n).reshape(-1, 1)
        self.gradient = (self.data[-1] - self.data[0]) / (self.n - 1)
        self.noise_std = np.std(self.data[-self.window_size:])
        detrended = detrend(self.data)
        centred = detrended - np.mean(detrended)
        transform = fft.rfft(centred)
        freqs = fft.rfftfreq(self.n)
        peak_idx = np.argmax(np.abs(transform))
        self.amplitude = 2 * np.abs(transform[peak_idx]) / self.n
        peak_ampl = freqs[peak_idx]
        self.period = 1 / peak_ampl if peak_ampl != 0 else None

    def exponential_smoothing(self):
        smoothed = [self.data[0]]
        for r in self.data[1:]:
            smoothed.append(self.alpha * r + (1 - self.alpha) * smoothed[-1])
        return np.array(smoothed)

    def poly_regression(self):
        if self.n <= self.poly_deg:
            return float(self.data[-1])
        coefficients = np.polyfit(
            np.arange(self.n),
            self.data,
            deg=self.poly_deg
        )
        poly_func = np.poly1d(coefficients)
        return float(poly_func(self.n))

    def linear_extrapolation(self):
        if self.n < 2:
            return 0.0, float(self.data[-1])
        slope_last = self.data[-1] - self.data[-2]
        return slope_last, self.data[-1] + slope_last

    def rolling_slope(self, smoothed, slope_last):
        if self.n >= self.window_size:
            window_diff = smoothed[-1] - smoothed[-self.window_size]
            slope_roll = window_diff / (self.window_size - 1)
        else:
            slope_roll = slope_last
        return smoothed[-1] + slope_roll

    def adaptive_weights(self):
        if self.n >= self.window_size:
            recent_data = self.data[-self.window_size:]
        else:
            recent_data = self.data
        recent_sigma = np.std(recent_data)
        weight_local = recent_sigma / (recent_sigma + self.c)
        weight_regression = 1 - weight_local
        return weight_local, weight_regression

    def deterministic_pred(self):
        weight_local, weight_reg = self.adaptive_weights()
        weight_last = weight_local / 2
        weight_roll = weight_local / 2
        smoothed = self.exponential_smoothing()
        pred_reg = self.poly_regression()
        adj_reg = pred_reg * weight_reg
        slope_last, pred_last = self.linear_extrapolation()
        adj_last = pred_last * weight_last
        pred_roll = self.rolling_slope(smoothed, slope_last)
        adj_roll = pred_roll * weight_roll
        pred_d = adj_reg + adj_last + adj_roll
        return float(pred_d)

    def bayesian_pred(self):
        poly = PolynomialFeatures(degree=self.poly_deg, include_bias=False)
        x_poly = poly.fit_transform(self.x)
        model = BayesianRidge(compute_score=True)
        model.fit(x_poly, self.data)
        x_next = np.array([[self.n]])
        x_next_poly = poly.transform(x_next)
        mean_pred, std_pred = model.predict(x_next_poly, return_std=True)
        z = norm.ppf(0.5 + self.ci / 2)
        lower = mean_pred - z * std_pred
        upper = mean_pred + z * std_pred
        return {
            "mean": float(mean_pred),
            "std": float(std_pred),
            "lower": float(lower),
            "upper": float(upper)
        }

    def predict_next_value(self):
        return {
            "deterministic": self.deterministic_pred(),
            "bayesian": self.bayesian_pred()
        }

    def generate_next_points(self, steps=5):
        new_points = []
        print('Gradient: ' + str(self.gradient))
        print('Period: ' + str(self.period))
        print('Amplitude: ' + str(self.amplitude))
        for i in range(1, steps + 1):
            t = self.n + i - 1
            trend_part = self.data[-1] + self.gradient * i
            if self.period is not None:
                osc_part = self.amplitude * np.sin(2 * np.pi * t / self.period)
            else:
                osc_part = 0.0
            noise_part = np.random.normal(scale=self.noise_std)
            new_point = trend_part + osc_part + noise_part
            new_points.append(new_point)
        return new_points
