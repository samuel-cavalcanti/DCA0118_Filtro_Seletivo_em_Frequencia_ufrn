import numpy as np
from matplotlib import pyplot
import os
from scipy.io import wavfile


def read_wav(path: str) -> (list, int):
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


def write_wav(file_name: str, sample_rate, sinal: np.array):
    audio_dir = 'audios'
    if not os.path.isdir(audio_dir):
        os.mkdir(audio_dir)

    wavfile.write(os.path.join(audio_dir, f'{file_name}.wav'), rate=sample_rate, data=sinal)
    print(f'Saved  {file_name} audio')


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

    print(f"save plot {title} saved")


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
    song, sample_rate_in_hertz = read_wav('audios/song.wav')
    noise_sample = make_noise(song.shape[0], sample_rate_in_hertz)

    second_channel: np.ndarray = song[:, 1]
    noised_song = second_channel + noise_sample

    noised_song_in_frequency_domain = np.fft.fft(noised_song)

    w = make_frequency_values_in_rads(noised_song.size, sample_rate_in_hertz)

    filtered_signal = noised_song_in_frequency_domain.copy()

    f_1_in_hertz = 2.4e3
    f_2_in_hertz = 2.7e3

    '''
    1 Hertz = 1rad/2pi
    0.05pi rad  = 0.025 Hertz
    '''
    delta_omega_in_hertz = 0.025

    mask_f_1 = (f_1_in_hertz - delta_omega_in_hertz < np.abs(w)) & (
            np.abs(w) < f_1_in_hertz + delta_omega_in_hertz)

    mask_f_2 = (f_2_in_hertz - delta_omega_in_hertz < np.abs(w)) & (
            np.abs(w) < f_2_in_hertz + delta_omega_in_hertz)

    '''
    os vetor mask_f1, he vetor que contem os indicies que satisfazem as
    condicoes:   2.4kHz - 0.025Hz  < |w| < 0.24kHz + 0.025Hz
    
    e o vetor mask_f2, he o vetor que contem os indicies qie satisfazem
    as condicoes: 2.7kHz - 0.025Hz  < |w| < 0.27kHz + 0.025Hz
    
    onde:
     0.025 e o delta omega
     f_1 = 2.4kHz
     f_2 = 2.7Hz
    '''

    filtered_signal[mask_f_1] = 0  # dessa forma eu 0 a amplitude do ruido f_1

    filtered_signal[mask_f_2] = 0  # dessa forma eu 0 a amplitude do ruido f_2

    filtered_signal_frequency = filtered_signal.copy()

    filtered_signal = np.fft.ifft(filtered_signal)

    signals_to_save = [
        {'array': noised_song, 'name': 'noise_song', 'rate': sample_rate_in_hertz},
        {'array': noise_sample, 'name': 'noise.wav', 'rate': sample_rate_in_hertz},
        {'array': filtered_signal.real, 'name': 'filtered signal', 'rate': sample_rate_in_hertz},
    ]

    signals_to_plot = [
        {'signal': second_channel, 'name': 'minha voz ',
         'x': np.arange(second_channel.size) * 1 / sample_rate_in_hertz},

        {'signal': noised_song, 'name': 'minha voz com ruido',
         'x': np.arange(second_channel.size) * 1 / sample_rate_in_hertz},

        {'signal': noise_sample[:1000], 'name': 'ruido no tempo',
         'x': np.arange(1000) * 1 / sample_rate_in_hertz},

        {'signal': norm(filtered_signal_frequency), 'name': 'minha voz depois do filtro',
         'x': make_frequency_values_in_rads(second_channel.size, sample_rate_in_hertz)},

        {'signal': norm(noised_song_in_frequency_domain), 'name': 'minha voz ates do filtro',
         'x': make_frequency_values_in_rads(second_channel.size, sample_rate_in_hertz)},
    ]

    save_signals_as_wav(signals_to_save)

    plot_signals(signals_to_plot)


def save_signals_as_wav(signals: [dict]):
    for signal in signals:
        write_wav(signal['name'], signal['rate'], signal['array'])


def plot_signals(signals: [dict]):
    figure_number = 1

    for signal in signals:
        plot_wave(signal['signal'], figure_number, signal['name'], x=signal['x'])
        figure_number += 1


if __name__ == '__main__':
    main()
