import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    """
    Rede Neural Linear para Q-Learning.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Inicializa a rede neural com camadas lineares.
        
        Args:
            input_size (int): Tamanho da camada de entrada.
            hidden_size (int): Tamanho da camada oculta.
            output_size (int): Tamanho da camada de saída.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Passa os dados pela rede neural.
        
        Args:
            x (Tensor): Dados de entrada.
        
        Returns:
            Tensor: Saída da rede neural.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name="model.pth"):
        """
        Salva o estado do modelo em um arquivo.
        
        Args:
            file_name (str): Nome do arquivo para salvar o modelo.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        
class QTrainer:
    """
    Treinador para a rede neural Q-Learning.
    """
    
    def __init__(self, model, lr, gamma):
        """
        Inicializa o treinador com o modelo, taxa de aprendizado e fator de desconto.
        
        Args:
            model (nn.Module): Modelo da rede neural.
            lr (float): Taxa de aprendizado.
            gamma (float): Fator de desconto.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, state, action, reward, next_state, game_over):
        """
        Executa um passo de treinamento.
        
        Args:
            state (array): Estado atual.
            action (array): Ação tomada.
            reward (float): Recompensa recebida.
            next_state (array): Próximo estado.
            game_over (bool): Indicador de fim de jogo.
        """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # Adiciona uma dimensão extra se necessário
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)  
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )
            
        # Predição dos valores Q com o estado atual
        pred = self.model(state)
        
        # Clona a predição para modificar os valores Q
        targ = pred.clone()
        
        for idx in range(len(game_over)):
            qnew = reward[idx]
            if not game_over[idx]:
                qnew = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            targ[idx][torch.argmax(action).item()] = qnew
        
        # r + y * max(próximo valor Q predito)
        self.optimizer.zero_grad()
        loss = self.criterion(targ, pred)
        loss.backward()
        
        self.optimizer.step()