from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np


def get_1d_frequency(time_duration, x, y, percentage=0.5, plot=False):
    """
    Args:
        time_duration: the time duration of the signal
        y: is a N * 1 tensor / array

    Returns:

    """
    # N = Total number of samples = Sample rate x time duration
    # sample_freq = 1 / Sample Rate
    N = y.shape[0]
    sample_freq = time_duration / N
    yf = fft(y.flatten())

    # we plot the half of range since the frequency plot is symmetric
    xf = fftfreq(N, sample_freq)[:N // 2]
    yf_half = 2.0 / N * np.abs(yf[:N // 2])

    # acquire some statistical values
    max_freq = xf[np.argmax(yf_half)]
    xf_range = xf[yf_half > (1 - percentage) * np.max(yf_half)]
    yf_range = np.ones_like(xf_range) * (1 - percentage) * np.max(yf_half)
    r_min_freq = np.min(xf_range)
    r_max_freq = np.max(xf_range)
    omega = max_freq * 2 * np.pi
    # print("=" * 47, " Fast Fourier Transform ", "=" * 47)
    # print("the maximum frequency is around {}".format(max_freq))
    # print("the maximum omega is around {}".format(omega))

    freq_range = [r_min_freq, r_max_freq]
    omega_range = [r_min_freq * np.pi * 2, r_max_freq * np.pi * 2]
    # print("the frequency range is between {} and {}".format(freq_range[0], freq_range[1]))
    # print("the omega range is between {} and {}".format(omega_range[0], omega_range[1]))
    # print("-" * 120)

    if plot:
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(x, y)
        axes[1].plot(xf, yf_half)
        axes[1].plot(xf_range, yf_range, 'r--')
        axes[1].grid()
        plt.show()

    return max_freq, omega, freq_range, omega_range
