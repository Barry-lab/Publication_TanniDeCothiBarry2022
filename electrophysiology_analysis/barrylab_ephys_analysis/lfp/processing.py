
from scipy.signal import butter, filtfilt, hilbert
from scipy.fftpack import next_fast_len


def hilbert_fast(signal_in):
    """Uses scipy.signal.hilbert with correct number of Fourier components for speedup

    for amplitude envelope use numpy.abs(analytic_signal)
    for instantaneous phase use numpy.unwrap(np.angle(analytic_signal))
    for instantaneous frequency use (numpy.diff(numpy.unwrap(np.angle(analytic_signal)) / (2.0*np.pi) * fs)

    :param numpy.ndarray signal_in: shape (N,)
    :return: analytical_signal of signal_in
    """
    return hilbert(signal_in, next_fast_len(len(signal_in)))[:len(signal_in)]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * float(fs)
    low = lowcut / nyq
    high = highcut / nyq
    ba = butter(order, [low, high], btype='band')
    return ba[0], ba[1]


def bandpass_filter(
        signal_in, sampling_rate=30000.0, highpass_frequency=300.0, lowpass_frequency=6000.0, filter_order=4):
    """Filters signal in specified frequency ranges using digital butter non-causal IIR filter.

    :param numpy.ndarray signal_in: shape (N,)
    :param int sampling_rate:
    :param float highpass_frequency:
    :param float lowpass_frequency:
    :param int filter_order:
    :return: signal in filtered form
    :rtype: numpy.ndarray
    """
    if signal_in.ndim > 1:
        raise Exception('signal_in must have shape (N,), but has shape ' + str(signal_in.shape))

    b, a = butter_bandpass(float(highpass_frequency), float(lowpass_frequency),
                           sampling_rate, order=filter_order)
    signal_out = filtfilt(b, a, signal_in)

    return signal_out
