# Projeto final de Processamento Digital de Sinais

__Informações dadas pelo professor:__ [Projeto de Filtragem FIR](Projeto_Filtragem_FIR_2.pdf)

__Parâmetro escolhido__: 9

## Dependências

Para instalar as dependências, use o comando:

```shell
pip install -r requirements.txt
```

Basicamente tenha instalado os pacotes:
- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/stable/)


## informações sobre a solução

Basicamente o único arquivo usado para o projeto foi [trabalho_final.py](trabalho_final.py)  
ao executar o script, vai re-gravar os arquivos de audio e os graficos feitos

```shell
python3 [trabalho_final.

# Saida esperada

#wav file audios/song.wav framerate 44100, data type float64 channels (176400, 2)
#Saved noise_song.wav audio
#Saved noise.wav audio
#Saved filtered signal if and else.wav audio
#Saved filtered by kaiser.wav audio
#save plot minha voz  saved
#save plot minha voz com ruido saved
#save plot minha voz filtrada pelo rejeita-banda de kaiser saved
#save plot ruido no tempo saved
#save plot diferenca da minha voz com o resultado do filtro kaiser saved
#save plot diferenca da minha voz com o resultado do filtro if e else saved
#save plot minha voz depois do filtro if e else saved
#save plot resposta em frequencia da minha voz depois do filtro kaiser  saved
#save plot resposta em frequencia do filtro kaiser  saved
#save plot resposta em frequencia da minha voz com ruido saved
#save plot resposta em frequencia do filtro passa alta kaiser saved
#save plot resposta em frequencia do filtro passa baixa kaiser saved
#save plot resposta em frequencia do filtro kaiser saved
```