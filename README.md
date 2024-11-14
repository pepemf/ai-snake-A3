# Jogo da Cobrinha com Aprendizado por Reforço

Este projeto implementa um agente inteligente que joga o clássico jogo da cobrinha utilizando técnicas de Aprendizado por Reforço (Reinforcement Learning). O objetivo é treinar o agente para maximizar sua pontuação no jogo, aprendendo a evitar colisões e a capturar a fruta.

## Estrutura do Projeto

- `game.py`: Contém a lógica do jogo da cobrinha.
- `model.py`: Define a estrutura da rede neural e o treinador.
- `helper.py`: Contém funções auxiliares para plotar os resultados.
- `agent.py`: Implementa o agente que joga o jogo utilizando Q-Learning.
- `requirements.txt`: Lista de dependências do projeto.

## Dependências

Para instalar as dependências do projeto, execute:

```sh
pip install -r 

requirements.txt


```

## Como Executar

1. **Clone o repositório**:
   ```sh
   git clone https://github.com/pepemf/ai-snake-A3
   cd seu-repositorio
   ```

2. **Ative seu ambiente virtual** (se estiver usando um):
   - No Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - No macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

3. **Instale as dependências**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Execute o treinamento do agente**:
   ```sh
   python agent.py
   ```

## Descrição do Projeto

### Base de Dados Utilizada

Para este projeto, não utilizamos uma base de dados tradicional. Em vez disso, o ambiente do jogo da cobrinha serve como a fonte de dados. As interações do agente com o ambiente geram os dados necessários para o treinamento. Cada estado do jogo, ação tomada, recompensa recebida e próximo estado formam uma transição que é armazenada na memória do agente.

### Abordagem de IA

A abordagem utilizada é o Aprendizado por Reforço, especificamente com a técnica de Q-Learning. O Q-Learning é um método de aprendizado por reforço que busca aprender uma política ótima para um agente, maximizando a recompensa acumulada ao longo do tempo.

- **Modelo de Rede Neural**: Utilizamos uma rede neural simples com duas camadas lineares para aproximar a função Q. A rede neural recebe o estado atual do jogo como entrada e retorna os valores Q para cada ação possível.
- **Treinamento**: O agente é treinado utilizando a técnica de Q-Learning com experiência de replay. As transições são armazenadas em uma memória de replay e amostras aleatórias são utilizadas para treinar a rede neural.
- **Exploração vs. Exploração**: Utilizamos uma estratégia epsilon-greedy para balancear a exploração de novas ações e a exploração das ações já conhecidas.

### Resultados

Os resultados do treinamento do agente são apresentados em gráficos que mostram a pontuação ao longo do tempo e a média das pontuações. O agente é capaz de aprender a jogar o jogo da cobrinha de forma eficiente, maximizando sua pontuação e evitando colisões.

### Implementação

A implementação foi realizada em Python utilizando as bibliotecas Pygame para o ambiente do jogo e PyTorch para a construção e treinamento da rede neural. O código foi organizado em diferentes módulos para facilitar a manutenção e a compreensão.

### Conclusão

Este projeto demonstra a aplicação prática de técnicas de Aprendizado por Reforço em um problema clássico de jogos. O agente treinado é capaz de aprender a jogar o jogo da cobrinha de forma eficiente, mostrando a eficácia do Q-Learning e das redes neurais na solução de problemas de tomada de decisão sequencial.

## Referências

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
- Pygame Documentation: https://www.pygame.org/docs/
- PyTorch Documentation: https://pytorch.org/docs/


