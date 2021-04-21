import numpy as np
from matplotlib import pyplot
import os
from scipy.io import wavfile


def read_wav(path: str) -> (np.ndarray, int):
    standard_deviation_table = {
        'int32': 2147483648,
        'int16': 32768,
        'float64': 1
    }
    '''
     Olhando na referencia do wav, existem essas maneiras de armazenar um audio: int32, int16, float64,
     nos inteiros 32bits, os valores variam de -2147483648 ate 2147483648
     nos inteiros de 16 bits, os valores variam de -32768, 32768
     nos numeros ponto fluatante de 64 bits, os valores ja estao entre -1 ate 1
     
     
     pode existir outras maneiras de se armazar o audio entao caso ocorra por favor, adicionar na tabela
     o formato e sua precisao, nao adicionei todos pois nao testei todos, 
     
     https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
     
     para o processamento de sinais realizados nesse trabalho, foi coveniente adotar a variacao de -1 ate 1
     '''

    frame_rate_in_samples_per_sec, data = wavfile.read(path)

    data = data / standard_deviation_table[str(data.dtype)]

    wave_info = f"wav file {path} framerate {frame_rate_in_samples_per_sec}, data type {data.dtype} channels {data.shape}"

    print(wave_info)

    return data, frame_rate_in_samples_per_sec


def write_wav(file_name: str, sample_rate, sinal: np.array):
    audio_dir = 'audios'
    if not os.path.isdir(audio_dir):
        os.mkdir(audio_dir)

    if file_name[-4:] != '.wav':
        file_name += '.wav'

    wavfile.write(os.path.join(audio_dir, f'{file_name}'), rate=sample_rate, data=sinal)
    print(f'Saved {file_name} audio')


def plot_wave(y: np.ndarray, figure_number: int, title: str, line_format='g-', x: np.ndarray = np.array([])):
    pyplot.figure(figure_number)
    pyplot.title(title)
    plot_dir = 'plots'
    if x.size:
        pyplot.plot(x, y, line_format)
    else:
        pyplot.stem(y, linefmt=line_format, markerfmt=' ')

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    pyplot.savefig(os.path.join(plot_dir, f'{title}.png'))
    # pyplot.show()
    print(f"save plot {title} saved")


def sinc(n):
    return np.sin(n) / (np.pi * n)


def make_low_pass_kaiser_filter(frequency_in_hertz: float, m_kaiser: int, beta_kaiser: float, sample_rate: int):
    kaiser = np.kaiser(m_kaiser, beta_kaiser)
    x = np.arange(m_kaiser)

    '''
    o vetor x e um vetor de indices de amostra de 0 ate o M de kaiser
    para transformar esses indices em valores analogicos temos que multiplicar pelo intervalo de amostragem
    que he o inverso da frequencia de amostragem
    '''

    w_c = 2 * np.pi * frequency_in_hertz / sample_rate

    sample_sinc = [w_c / np.pi if value - m_kaiser / 2 == 0 else sinc(w_c * (value - (m_kaiser / 2))) for value in x]

    sample_sinc = np.array(sample_sinc)

    low_pass_filter = sample_sinc * kaiser

    low_pass_filter /= np.sum(low_pass_filter)  # normalizando o filtro ,para ficar com valores 0 -1

    return low_pass_filter


def make_high_pass_kaiser_filter(frequency_in_hertz: float, m_kaiser: int, beta_kaiser: float, sample_rate: int):
    kaiser = np.kaiser(m_kaiser, beta_kaiser)
    x = np.arange(m_kaiser) + 1e-10  # [0,1,2,3,4, ..., 145]

    '''
     o vetor x e um vetor de indices de amostra de 0 ate o M de kaiser
     para transformar esses indices em valores analogicos temos que multiplicar pelo intervalo de amostragem
     que he o inverso da frequencia de amostragem
     '''

    w_c = 2 * np.pi * frequency_in_hertz / sample_rate

    sample_sinc = [w_c / np.pi if value - m_kaiser / 2 == 0 else sinc(w_c * (value - (m_kaiser / 2))) for value in x]

    sample_sinc = np.array(sample_sinc)

    high_pass_filter = sample_sinc * kaiser

    high_pass_filter /= np.sum(high_pass_filter)  # normalizando o filtro de 0-1

    high_pass_filter = -high_pass_filter

    high_pass_filter[m_kaiser // 2] += 1  # somando um impulso unitario

    return high_pass_filter


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
def make_frequency_values_in_hertz(sample_size: int, sample_rate: int) -> np.array:
    rads = sample_rate / sample_size
    return np.array(
        [rads * n if n < sample_size // 2 else rads * (n - sample_size) for n in range(sample_size)])


# calcula a norma de um sinal
def norm(signal: np.ndarray) -> np.ndarray:
    return np.array([np.sqrt(value.real ** 2 + value.imag ** 2) for value in signal])


# calcula a fase de um sinal
def phase(signal: np.ndarray) -> np.ndarray:
    return np.array([np.arctan(value.imag / value.real) for value in signal])


# atrasa o sinal na metade da sua amostra
def delay(signal: np.ndarray):
    half_array = signal.size // 2
    temp = signal[:half_array].copy()

    signal[:half_array] = signal[half_array:]
    signal[half_array:] = temp


def main():
    song, sample_rate_in_hertz = read_wav('audios/song.wav')
    noise_sample = make_noise(song.shape[0], sample_rate_in_hertz)

    second_channel: np.ndarray = song[:, 1]
    noised_song: np.ndarray = second_channel + noise_sample

    noised_song_in_frequency_domain = np.fft.fft(noised_song)

    filtered_signal = noised_song_in_frequency_domain.copy()

    f_1_in_hertz = 2.4e3
    f_2_in_hertz = 2.7e3
    beta_kaiser = 5.65326  # 0.1102 * (a - 8.7), onde a = 60
    m_kaiser = 146  # aproximadadmente (a - 8)/(2.285*delta_Omega, onde a = 60 e delta_Omega =0.025

    '''
    1 Hertz = 1rad/2pi
    0.05pi rad  = 0.025 Hertz
    '''
    delta_omega_in_hertz = 0.025

    w = make_frequency_values_in_hertz(filtered_signal.size, sample_rate_in_hertz)

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

    filtered_signal[mask_f_1] = 0  # dessa forma eu zero a amplitude do ruido f_1

    filtered_signal[mask_f_2] = 0  # dessa forma eu zero a amplitude do ruido f_2

    filtered_signal_frequency = filtered_signal.copy()

    filtered_signal: np.ndarray = np.fft.ifft(filtered_signal)  # transformada inversa de fourier de tempo discreto

    '''
    Filtrando usando um filtro de kaiser
    '''

    '''
    um parametro dado pelo professor foi o DeltaW, que e igual a 0.05 pi rad
    para converter para  Hz tenhos que dividir o valor por 2pi
    porntando o deltaW/2pi = 0.025 Hz
    para converter o delta para a nossa banda ([-22.05kHz,22.05kHz]) temos que multiplicar o deltaW pela
    frequencia de coleta, ou seja 0.025* 44.1kHz
    logo a nossa banda de transicao he 0.025 * 44.1kHz = 1102.5
    '''

    transition_band_in_hertz = 1102.5 / 2

    w_c_1 = f_1_in_hertz - transition_band_in_hertz / 2

    w_c_2 = f_2_in_hertz + transition_band_in_hertz / 2

    low_pass_filter = make_low_pass_kaiser_filter(frequency_in_hertz=w_c_1,
                                                  m_kaiser=m_kaiser,
                                                  beta_kaiser=beta_kaiser,
                                                  sample_rate=sample_rate_in_hertz)

    high_pass_filter = make_high_pass_kaiser_filter(frequency_in_hertz=w_c_2,
                                                    m_kaiser=m_kaiser,
                                                    beta_kaiser=beta_kaiser,
                                                    sample_rate=sample_rate_in_hertz)

    band_reject = low_pass_filter + high_pass_filter

    filtered_by_kaiser_filter = np.convolve(noised_song, band_reject, mode='same')

    norm_low_pass_filter = norm(np.fft.fft(low_pass_filter))
    norm_high_pass_filter = norm(np.fft.fft(high_pass_filter))
    norm_band_reject = norm(np.fft.fft(band_reject))

    norm_filtered_by_kaiser_filter = norm(np.fft.fft(filtered_by_kaiser_filter))

    frequency_values_kaiser = make_frequency_values_in_hertz(m_kaiser, sample_rate_in_hertz)

    '''
    np.fft.fft(low_pass_filter) retorna a transformada de fourier do sinal
    norm(sinal) retorna o modulo do sinal ou seja, retorna |x|,
    lembrando que um sinal complexo pode ser representado pelo seu modulo fase
    
    portando norm(np.fft.fft(sinal)), retorna o modolo da resposta em frequencia do sinal,
    pego o modulo da resposta em frequencia para plotar o grafico
    '''

    norm_filtered_signal_frequency = norm(filtered_signal_frequency)
    norm_noised_song_in_frequency_domain = norm(noised_song_in_frequency_domain)

    '''
    a transformada rapida de fourier retorna um sinal peridico onde
    ele comeca se repetir no meio do vetor, ou seja se o vetor tem tamanho N
    no indicie N/2 volta a possuir valores próximos a indice 0
    entao e interessante "atrasar" o sinal em N/2, para o sinal ficar simetrico em relacao a origem,
    he isso que que a funcao delay faz, ela atrasa um sinal em N/2    
    '''
    delay(w)
    delay(frequency_values_kaiser)

    delay(norm_low_pass_filter)
    delay(norm_high_pass_filter)
    delay(norm_band_reject)
    delay(norm_filtered_signal_frequency)
    delay(norm_noised_song_in_frequency_domain)
    delay(norm_filtered_by_kaiser_filter)

    '''
    lista de sinais que salvo em formato de audio, para salvar um audio
    e necessario fornecer um nome do arquivo, o sample rate em hertz e
    o sinal com APENAS valores reais, afinal a parte imaginaria so existe
    na sua cabeca.
    
    todos os arquivos de auido seram salvos na pasta audios, caso nao tenha
    o script cria a pasta.    
    '''
    signals_to_save = [
        {'array': noised_song, 'name': 'noise_song', 'rate': sample_rate_in_hertz},
        {'array': noise_sample, 'name': 'noise', 'rate': sample_rate_in_hertz},
        {'array': filtered_signal.real, 'name': 'filtered signal if and else', 'rate': sample_rate_in_hertz},
        {'array': filtered_by_kaiser_filter, 'name': 'filtered by kaiser', 'rate': sample_rate_in_hertz},
    ]

    '''
    lista de sinais que visualizo o grafico e salvo em formado png,
    para plotar o sinal, eu preciso de nome que sera o titulo da figura
    o eixo x que pode ser em segundos ou em hertz
    exite um outro atributo opcional  que he o line_format,
    que pode modificar a cor e o tipo de traco, 
    ex: -b (linha azul), ob (bolinha azul), -r (linha vermelha)
    
    eu adotei azul para tempo e verde para frequencia,
    por padrao caso nao coloque nada sera linha verde
    '''

    '''
    vetor que representa o  eixo x em segundos da amostra coletada pelo meu microfone
    1 Hz = 1/segundos
    logo  y segundos = 1/x Hz
    '''
    time_in_seconds = np.arange(second_channel.size) * 1 / sample_rate_in_hertz

    signals_to_plot = [
        {'signal': second_channel, 'name': 'minha voz ',
         'line_format': '-b',
         'x': time_in_seconds},

        {'signal': noised_song, 'name': 'minha voz com ruido',
         'line_format': '-b',
         'x': time_in_seconds},

        {'signal': filtered_by_kaiser_filter,
         'name': 'minha voz filtrada pelo rejeita-banda de kaiser',
         'line_format': '-b',
         'x': time_in_seconds},

        {'signal': noise_sample[:1000], 'name': 'ruido no tempo',
         'line_format': '-b',
         'x': time_in_seconds[:1000]},

        {'signal': second_channel - filtered_by_kaiser_filter,
         'name': 'diferenca da minha voz com o resultado do filtro kaiser',
         'line_format': '-b',
         'x': time_in_seconds},

        {'signal': second_channel - filtered_signal.real,
         'name': 'diferenca da minha voz com o resultado do filtro if e else',
         'line_format': '-b',
         'x': time_in_seconds},

        {'signal': norm_filtered_signal_frequency, 'name': 'minha voz depois do filtro if e else',
         'x': w},

        {'signal': norm_filtered_by_kaiser_filter,
         'name': 'resposta em frequencia da minha voz depois do filtro kaiser ',
         'x': w},

        {'signal': norm_band_reject,
         'name': 'resposta em frequencia do filtro kaiser ',
         'x': frequency_values_kaiser},

        {'signal': norm_noised_song_in_frequency_domain, 'name': 'resposta em frequencia da minha voz com ruido',
         'x': w},

        {'signal': norm_high_pass_filter,
         'name': 'resposta em frequencia do filtro passa alta kaiser',
         'x': frequency_values_kaiser,
         'line_format': '-g'},

        {'signal': norm_low_pass_filter,
         'name': 'resposta em frequencia do filtro passa baixa kaiser',
         'x': frequency_values_kaiser},

        {'signal': norm_band_reject,
         'name': 'resposta em frequencia do filtro kaiser',
         'x': frequency_values_kaiser},
    ]

    save_signals_as_wav(signals_to_save)

    plot_signals(signals_to_plot)


def save_signals_as_wav(signals: [dict]):
    for signal in signals:
        write_wav(signal['name'], signal['rate'], signal['array'])


def plot_signals(signals: [dict], show_plot: bool = False):
    figure_number = 1

    pyplot.rcParams["figure.figsize"] = (20, 5)

    for signal in signals:
        plot_wave(signal['signal'], figure_number, signal['name'], line_format=signal.get('line_format', '-g'),
                  x=signal['x'])
        figure_number += 1

    if show_plot:
        pyplot.show()


def test_filter():
    sample_rate = 44100

    m = 148
    a = 60
    beta = 0.1102 * (a - 8.7)  # 5.65326

    f_1_in_hertz = 2.4e3
    f_2_in_hertz = 2.7e3

    low_pass_filter = make_low_pass_kaiser_filter(f_1_in_hertz, m, beta, sample_rate)

    high_pass_filter = make_high_pass_kaiser_filter(f_2_in_hertz, m, beta, sample_rate)

    h_filter = low_pass_filter + high_pass_filter

    w = make_frequency_values_in_hertz(m, sample_rate)

    norm_low_pass_filter = norm(np.fft.fft(low_pass_filter))
    norm_high_pass_filter = norm(np.fft.fft(high_pass_filter))
    norm_h_filter = norm(np.fft.fft(h_filter))

    delay(w)
    delay(norm_low_pass_filter)
    delay(norm_high_pass_filter)
    delay(norm_h_filter)

    signals_to_plot = [
        {'signal': norm_low_pass_filter, 'name': 'filtro passa baixa com $w_{c_1}$ = 2.4 kHz',
         'x': w},

        {'signal': 20 * np.log10(norm_high_pass_filter), 'name': 'filtro passa alta com $w_{c_2} $= 2.7  kHz',
         'x': w},

        {'signal': norm_h_filter, 'name': 'Filtro final',
         'x': w},
    ]

    plot_signals(signals_to_plot)


if __name__ == '__main__':
    # test_filter()
    main()
