import numpy as np
from matplotlib import pyplot


def read_wav(path: str) -> (list, int):
    from scipy.io import wavfile

    standard_deviation_table = {
        'int32': 2147483648,
        'int16': 32768,
        'float64': 1
    }

    framerate_in_samples_per_sec, data = wavfile.read(path)

    data = data / standard_deviation_table[str(data.dtype)]

    wave_info = f"wav file {path} framerate {framerate_in_samples_per_sec}, data type {data.dtype} channels {data.shape}"
    print(wave_info)
    return data, framerate_in_samples_per_sec


def plot_wave(data: np.array, figure_number: int, title: str, line_format='g-'):
    pyplot.figure(figure_number)
    pyplot.title(title)
    pyplot.stem(data, linefmt=line_format, markerfmt=' ')
    pyplot.show()
    # pyplot.savefig(f"{title}.png")

    print(f"{title} saved")


def noise(t: int) -> float:
    # Parametro 9 : f_1 = 2.4kHz f_2 = 2.7kHz A_r = 60Db deltaW =0.05pi
    A = 0.01
    f_1 = 2.4e3
    f_2 = 2.7e3
    return A*np.cos(2*np.pi*f_1*t) + A*np.cos(2*np.pi*f_2*t)


def make_noise(size_sample: int, frame_rate: int) -> np.array:
    period_rate = 1/frame_rate
    return np.array([noise(i*period_rate) for i in range(size_sample)])


# calcula a norma de um sinal
def norm(signal: np.array) -> np.array:
    return np.array([np.sqrt(value.real**2 + value.imag**2) for value in signal])


# calcula a fase de um sinal
def phase(signal: np.array) -> np.array:
    return np.array([np.arctan(value.imag / value.real) for value in signal])


song, frame_rate = read_wav('trabalho_final.wav')
noise_sample = make_noise(song.shape[0], frame_rate)


figure_number = 1

# plot_wave(song[:, 1], figure_number, 'song', 'b-')
# figure_number += 1

# plot_wave(noise_sample[0:100], figure_number, '100 samples of noise')
# figure_number += 1


noised_song = song[:, 1] + noise_sample

# plot_wave(noised_song, figure_number, 'noised song', 'b-')
# figure_number += 1


sp = np.fft.fft(noised_song)

norm_sp = norm(sp)
freq = np.fft.fftfreq(noised_song.shape[-1])
pyplot.plot(freq, norm_sp.real, freq, norm_sp.imag)
pyplot.title('Minha voz com ruido na frequencia')
pyplot.show()
figure_number += 1

# phase_sp = phase_sp(phase_sp)
