import random
import torch
import numpy as np
from collections import deque
from game import Cobrinha_Game, Direction, Point
from helper import plot
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    """
    Agente que joga o jogo da cobrinha usando Q-Learning.
    """
    
    def __init__(self):
        """
        Inicializa o agente com os parâmetros necessários.
        """
        self.n_games = 0
        self.epsilon = 0 # Aleatoriedade
        self.gamma = 0.9 # Taxa de desconto
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.last_distance = None
        self.last_direction = None
    
    def get_state(self, game):
        """
        Obtém o estado atual do jogo.
        
        Args:
            game (Cobrinha_Game): Instância do jogo.
        
        Returns:
            np.array: Estado atual do jogo.
        """
        head = game.cobrinha[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Perigo em frente
            (dir_r and game.valida_colisao(point_r)) or 
            (dir_l and game.valida_colisao(point_l)) or 
            (dir_u and game.valida_colisao(point_u)) or 
            (dir_d and game.valida_colisao(point_d)),

            # Perigo à direita
            (dir_u and game.valida_colisao(point_r)) or 
            (dir_d and game.valida_colisao(point_l)) or 
            (dir_l and game.valida_colisao(point_u)) or 
            (dir_r and game.valida_colisao(point_d)),

            # Perigo à esquerda
            (dir_d and game.valida_colisao(point_r)) or 
            (dir_u and game.valida_colisao(point_l)) or 
            (dir_r and game.valida_colisao(point_u)) or 
            (dir_l and game.valida_colisao(point_d)),
            
            # Direção do movimento
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Localização da comida
            game.fruta.x < game.cabeca.x,  # comida à esquerda
            game.fruta.x > game.cabeca.x,  # comida à direita
            game.fruta.y < game.cabeca.y,  # comida acima
            game.fruta.y > game.cabeca.y  # comida abaixo
            ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, game_over):
        """
        Armazena uma transição na memória.
        
        Args:
            state (array): Estado atual.
            action (array): Ação tomada.
            reward (float): Recompensa recebida.
            next_state (array): Próximo estado.
            game_over (bool): Indicador de fim de jogo.
        """
        self.memory.append((state, action, reward, next_state, game_over))
    
    def train_long_memory(self):
        """
        Treina a memória de longo prazo usando amostras da memória.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        """
        Treina a memória de curto prazo com a transição atual.
        
        Args:
            state (array): Estado atual.
            action (array): Ação tomada.
            reward (float): Recompensa recebida.
            next_state (array): Próximo estado.
            game_over (bool): Indicador de fim de jogo.
        """
        self.trainer.train_step(state, action, reward, next_state, game_over)
    
    def get_action(self, state):
        """
        Obtém a ação a ser tomada com base no estado atual.
        
        Args:
            state (array): Estado atual.
        
        Returns:
            list: Ação a ser tomada [frente, esquerda, direita].
        """
        # Tradeoff exploração / exploração
        self.epsilon = 80 - self.n_games
        final_move = [0, 0 ,0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Cobrinha_Game()
    while True:
        # Obtém o estado antigo
        state_old = agent.get_state(game)
        
        # Obtém a ação
        final_move = agent.get_action(state_old)
        
        # Executa a ação e obtém o novo estado
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Penaliza se o número de iterações exceder o limite baseado no comprimento da cobra
        if game.frame_iteration > 100 * len(game.cobrinha):
            reward -= 1
        
        # Penaliza movimentos repetitivos
        if agent.last_direction == game.direction:
            reward -= 0.1
        agent.last_direction = game.direction
        
        # Penaliza a falta de progresso
        distance = np.linalg.norm(np.array([game.cabeca.x, game.cabeca.y]) - np.array([game.fruta.x, game.fruta.y]))
        if agent.last_distance is not None and distance >= agent.last_distance:
            reward -= 0.5
        agent.last_distance = distance
        
        # Treina a memória de curto prazo
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # Lembra a transição
        agent.remember(state_old, final_move, reward, state_new, game_over)
        
        if game_over:
            # Treina a memória de longo prazo
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print(f'Game: {agent.n_games}\nScore: {score}\nRecord: {record}')
            
            plot_scores.append(score)
            total_score += score
            mean_scores = total_score / agent.n_games
            plot_mean_scores.append(mean_scores)
            plot(plot_scores, plot_mean_scores)
            

if __name__ == "__main__":
    train()
