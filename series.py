import numpy as np
import matplotlib.pyplot as plt
from predictor import Predictor

np.random.seed(42)
n_points = 365
x = np.arange(n_points)
m = 0.3
trend = m * x
a = 5
period = 60
oscillation = a * np.sin(2 * np.pi * x / period)
noise = np.random.normal(scale=3.0, size=n_points)
y = trend + oscillation + noise
n_predictions = 50

print('Actual Gradient: ' + str(m))
print('Actual Period: ' + str(period))
print('Actual Amplitude: ' + str(a))

predictor = Predictor(data=y, alpha=0.3, poly_deg=2,
                      window_size=5, c=2.0, ci=0.95)
next_data = predictor.generate_next_points(n_predictions)

x_next = np.arange(n_points, n_points + n_predictions)
fft_pred_vals = np.array(next_data)

plt.figure(figsize=(18, 10))
plt.plot(x,
         y,
         label='Original Data',
         marker='o',
         markersize=2,
         linestyle='-',
         color='blue',
         linewidth=1,
         )
det_label = 'Deterministic Predictions Using Fourier Transform'
plt.plot(x_next,
         fft_pred_vals,
         label=det_label,
         marker='o',
         markersize=2,
         linestyle='-',
         color='green',
         linewidth=1
         )
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Next ' + str(n_predictions) + ' Points Forecast')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plot_filename = 'series_forecast_plot.png'
plt.savefig(plot_filename, dpi=300)

print('Plot saved as ' + plot_filename)
print('Deterministic next ' + str(n_predictions) + ' points:')
print(fft_pred_vals)
