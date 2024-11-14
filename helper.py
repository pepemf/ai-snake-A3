import matplotlib.pyplot as plt
from IPython import display

# Ativa o modo interativo do Matplotlib
plt.ion()

def plot(scores, mean_scores):
    """
    Plota as pontuações e as pontuações médias ao longo das gerações.

    Args:
        scores (list): Lista de pontuações.
        mean_scores (list): Lista de pontuações médias.
    """
    # Limpa a saída anterior e exibe o gráfico atual
    display.clear_output(wait=True)
    display.display(plt.gcf())
    
    # Limpa a figura atual
    plt.clf()
    
    # Define o título e os rótulos dos eixos
    plt.title('Treinamento do Agente no Jogo da Cobrinha')
    plt.xlabel('Número de Gerações')
    plt.ylabel('Pontuação')
    
    # Plota as pontuações e as pontuações médias
    plt.plot(scores, label='Pontuação', color='blue')
    plt.plot(mean_scores, label='Pontuação Média', color='orange')
    
    # Define o limite mínimo do eixo y
    plt.ylim(ymin=0)
    
    # Adiciona texto com a pontuação atual no gráfico
    plt.text(len(scores)-1, scores[-1], str(scores[-1]), color='blue')
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]), color='orange')
    
    # Adiciona uma grade ao gráfico
    plt.grid(True)
    
    # Adiciona uma legenda
    plt.legend()
    
    # Exibe o gráfico sem bloquear a execução do código
    plt.show(block=False)
    
    # Pausa para atualizar o gráfico
    plt.pause(.1)