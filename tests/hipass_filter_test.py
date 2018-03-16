from dataset_util import highpass_filtering
import numpy as np
import matplotlib.pyplot as plt

fs = 1000
T = 2
nsamples = T * fs
t = np.linspace(0, T, nsamples, endpoint=False)
a = 0.02
f0 = 40.0
x = 0.1 * np.sin(2 * np.pi * f0 * t)
# x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
# x += a * np.cos(2 * np.pi * f0 * t + .11)
x += 0.03 * np.cos(2 * np.pi * t)

plt.figure(2)
plt.clf()
plt.plot(t, x, label='Noisy signal')

lowcut = 1.5
y = highpass_filtering(x, lowcut, fs)
plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
plt.xlabel('time (seconds)')
plt.hlines([-a, a], 0, T, linestyles='--')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

plt.show()