import numpy as np
from matplotlib import pyplot
import os

FIGURE_NUMBER = 1  # Cada plot possui um numero que identifica a "folha em que o grafico he desenhado,


def read_wav(path: str) -> (list, int):
    from scipy.io import wavfile

    standard_deviation_table = {
        'int32': 2147483648,
        'int16': 32768,
        'float64': 1
    }

    frame_rate_in_samples_per_sec, data = wavfile.read(path)

    data = data / standard_deviation_table[str(data.dtype)]

    wave_info = f"wav file {path} framerate {frame_rate_in_samples_per_sec}, data type {data.dtype} channels {data.shape}"
    print(wave_info)
    return data, frame_rate_in_samples_per_sec


def plot_wave(y: np.array, figure_number: int, title: str, line_format='g-', x: np.array = np.array([])):
    pyplot.figure(figure_number)
    pyplot.title(title)
    plot_dir = 'plots'
    if x.size:
        pyplot.plot(x, y)
    else:
        pyplot.stem(y, linefmt=line_format, markerfmt=' ')

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    pyplot.savefig(os.path.join(plot_dir, f'{title}.png'))
    print(f"{title} saved")


def noise(t: float) -> float:
    # Parametro 9 : f_1 = 2.4kHz f_2 = 2.7kHz A_r = 60Db deltaW =0.05pi
    A = 0.005  # esse parâmetro é a minha escolha.
    f_1 = 2.4e3
    f_2 = 2.7e3
    return A * np.cos(2 * np.pi * f_1 * t) + A * np.cos(2 * np.pi * f_2 * t)


# cria o ruido a partir  do numero total de amostras e do espacamento entre as frequencias na coleta
def make_noise(size_sample: int, sample_rate: int) -> np.array:
    period_rate = 1 / sample_rate
    return np.array([noise(i * period_rate) for i in range(size_sample)])


# [0, 1, 2, ... , N/2, -N/2, (N/2-N), ((N/2 +1) -N), ((N/2 +2) -N), ... , (N -1) - N ]
# onde N e o numero de amostras
def make_frequency_values_in_rads(sample_size: int, sample_rate: int) -> np.array:
    rads = sample_rate / sample_size
    return np.array(
        [rads * n if n < sample_size // 2 else rads * (n - sample_size) for n in range(sample_size)])


# calcula a norma de um sinal
def norm(signal: np.array) -> np.array:
    return np.array([np.sqrt(value.real ** 2 + value.imag ** 2) for value in signal])


# calcula a fase de um sinal
def phase(signal: np.array) -> np.array:
    return np.array([np.arctan(value.imag / value.real) for value in signal])


def main():
    song, sample_rate = read_wav('trabalho_final.wav')
    noise_sample = make_noise(song.shape[0], sample_rate)

    second_channel: np.ndarray = song[:, 1]

    figure_number = 1

    time_in_seconds = np.arange(second_channel.size) * 1 / sample_rate

    plot_wave(second_channel, figure_number, 'song', 'b-', x=time_in_seconds)
    figure_number += 1

    plot_wave(noise_sample[0:1000], figure_number, '1000 samples of noise', x=time_in_seconds[0:1000])
    figure_number += 1

    noised_song = second_channel + noise_sample

    plot_wave(noised_song, figure_number, 'noised song', 'b-', x=time_in_seconds)
    figure_number += 1

    noised_song_in_frequency_domain = np.fft.fft(noised_song)
    norm_ns = norm(noised_song_in_frequency_domain)

    freq = make_frequency_values_in_rads(noised_song.size, sample_rate)

    plot_wave(norm_ns, figure_number, 'Modulo da Minha voz com ruido na frequencia', x=freq)
    figure_number += 1

    noise_in_frequency = np.fft.fft(noise_sample)
    norm_noise = norm(noise_in_frequency)

    plot_wave(norm_noise, figure_number, 'ruido na frequencia', x=freq)
    figure_number += 1


if __name__ == '__main__':
    main()
