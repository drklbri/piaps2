import numpy as np
import matplotlib.pyplot as plot
from scipy.fftpack import fft, ifft
from scipy.signal import butter, lfilter

# Функция для генерации гармонического сигнала
def generate_harmonic_signal(frequency, duration, sampling_rate):
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal

# Функция для генерации меандра
def generate_square_wave(frequency, duration, sampling_rate):
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = 0.5 * (1 + np.sign(np.sin(2 * np.pi * frequency * t)))
    return t, signal

# Функция для выполнения амплитудной модуляции
def amplitude_modulation(carrier_frequency, modulating_frequency, duration, sampling_rate):
    t, carrier_signal = generate_harmonic_signal(carrier_frequency, duration, sampling_rate)
    _, modulating_signal = generate_square_wave(modulating_frequency, duration, sampling_rate)
    modulated_signal = (1 + 0.5 * modulating_signal) * carrier_signal
    return t, modulated_signal

# Функция для выполнения частотной модуляции
def frequency_modulation(carrier_frequency, modulating_frequency, duration, sampling_rate):
    t, carrier_signal = generate_harmonic_signal(carrier_frequency, duration, sampling_rate)
    _, modulating_signal = generate_square_wave(modulating_frequency, duration, sampling_rate)
    modulated_signal = np.cos(2 * np.pi * carrier_frequency * t + 2 * np.pi * 0.1 * np.cumsum(modulating_signal) / sampling_rate)
    return t, modulated_signal

# Функция для выполнения фазовой модуляции
def phase_modulation(carrier_frequency, modulating_frequency, duration, sampling_rate):
    t, carrier_signal = generate_harmonic_signal(carrier_frequency, duration, sampling_rate)
    _, modulating_signal = generate_square_wave(modulating_frequency, duration, sampling_rate)
    modulated_signal = np.cos(2 * np.pi * carrier_frequency * t + 2 * np.pi * 0.5 * modulating_signal)
    return t, modulated_signal

# Функция для выполнения фильтрации сигнала
def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# Параметры сигнала
carrier_frequency = 10  # Частота несущего сигнала
modulating_frequency = 1  # Частота модулирующего сигнала
duration = 2  # Длительность сигнала в секундах
sampling_rate = 1000  # Частота дискретизации

# Амплитудная модуляция
t_am, modulated_signal_am = amplitude_modulation(carrier_frequency, modulating_frequency, duration, sampling_rate)

# Частотная модуляция
t_fm, modulated_signal_fm = frequency_modulation(carrier_frequency, modulating_frequency, duration, sampling_rate)

# Фазовая модуляция
t_pm, modulated_signal_pm = phase_modulation(carrier_frequency, modulating_frequency, duration, sampling_rate)

plot.figure(figsize=(15, 10))

plot.subplot(3, 2, 1)
plot.plot(t_am, modulated_signal_am, label='Amplitude Modulation')
plot.title('Amplitude Modulation')
plot.legend()

plot.subplot(3, 2, 3)
plot.plot(t_fm, modulated_signal_fm, label='Frequency Modulation')
plot.title('Frequency Modulation')
plot.legend()

plot.subplot(3, 2, 5)
plot.plot(t_pm, modulated_signal_pm, label='Phase Modulation')
plot.title('Phase Modulation')
plot.legend()

# Выполнение FFT и отсечение низких и высоких частот для амплитудной модуляции
fft_am = fft(modulated_signal_am)
cutoff_frequency_am = 20  # Задаем частоту отсечения
fft_am[(np.abs(fft_am) > cutoff_frequency_am)] = 0
filtered_signal_am = ifft(fft_am).real

# Фильтрация сигнала для приближения к исходному модулирующему сигналу
filtered_signal_am = butter_lowpass_filter(filtered_signal_am, modulating_frequency, sampling_rate)

# Визуализация сигналов амплитудной модуляции и фильтрованного сигнала
plot.subplot(3, 2, 2)
plot.plot(t_am, modulated_signal_am, label='Amplitude Modulation')
plot.title('Amplitude Modulation')
plot.legend()

plot.subplot(3, 2, 4)
plot.plot(t_am, filtered_signal_am, label='Filtered Signal')
plot.title('Filtered Signal (After FFT)')
plot.legend()

plot.tight_layout()
plot.show()