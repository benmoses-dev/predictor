import numpy as np
import matplotlib.pyplot as plt
from predictor import Predictor

np.random.seed(42)
n_points = 200
x_full = np.arange(n_points)
m = 0.2
trend = m * x_full
a = 5
period = 14
oscillation = a * np.sin(2 * np.pi * x_full / period)
noise = np.random.normal(scale=3.0, size=n_points)
y_full = trend + oscillation + noise
last = y_full[-1]
y = y_full[:-1]
x = np.arange(len(y))
next = len(y)

predictor = Predictor(data=y, alpha=0.3, poly_deg=2,
                      window_size=5, c=2.0, ci=0.95)

prediction = predictor.predict_next_value()
deterministic_value = prediction['deterministic']
bayesian_mean = prediction['bayesian']['mean']
bayesian_lower = prediction['bayesian']['lower']
bayesian_upper = prediction['bayesian']['upper']

plt.figure(figsize=(18, 10))

plt.plot(x, y, label='Original Data', marker='o',
         markersize=4, linestyle='-', linewidth=1, color='blue')

plt.scatter(next, deterministic_value, color='green',
            s=30, label='Deterministic Prediction')
plt.plot([next - 1, next], [y[-1], deterministic_value],
         color='green', linestyle='--', linewidth=1)

plt.scatter(next, bayesian_mean, color='red',
            s=30, label='Bayesian Prediction (mean)')
plt.plot([next - 1, next], [y[-1], bayesian_mean],
         color='red', linestyle='--', linewidth=1)
plt.vlines(next, bayesian_lower, bayesian_upper, color='red',
           linestyle='-', linewidth=1, label='95% Credible Interval')

plt.scatter(next, last, color='black', s=50,
            marker='x', label='Actual Next Value')


plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Deterministic vs Bayesian Prediction')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plot_filename = 'next_forecast_plot.png'
plt.savefig(plot_filename, dpi=300)

print('Plot saved as ' + plot_filename)
print('Deterministic next value = ' + str(deterministic_value))
print('Bayesian next value = ' + str(bayesian_mean))
print('Upper bound = ' + str(bayesian_upper))
print('Lower bound = ' + str(bayesian_lower))
print('Actual next value = ' + str(last))
