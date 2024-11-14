import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# Cores
BRANCO = (255, 255, 255)
COR_FRUTA = (200, 0, 0)
COR_COBRA = (0, 255, 0)  # Verde
COR_COBRA_BORDA = (0, 200, 0)  # Verde escuro
FUNDO = (0, 0, 0)  # Preto
CINZA = (40, 40, 40)  # Cor da grade

BLOCK_SIZE = 20
SPEED = 20

class Cobrinha_Game:
    """
    Classe principal do jogo da cobrinha.
    """
    
    def __init__(self, w=720, h=480):
        """
        Inicializa o jogo com a largura e altura especificadas.
        
        Args:
            w (int): Largura da janela do jogo.
            h (int): Altura da janela do jogo.
        """
        self.w = w
        self.h = h
        # Inicia Janela
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Cobrinha')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        """
        Reinicia o estado do jogo.
        """
        # Inicialização do jogo
        self.direction = Direction.RIGHT
        
        self.cabeca = Point(self.w/2, self.h/2)
        self.cobrinha = [self.cabeca, 
                        Point(self.cabeca.x-BLOCK_SIZE, self.cabeca.y),
                        Point(self.cabeca.x-(2*BLOCK_SIZE), self.cabeca.y)]
        
        self.pontuacao = 0
        self.fruta = None
        self._gerar_fruta()
        self.frame_iteration = 0
        
    def _gerar_fruta(self):
        """
        Gera uma nova fruta em uma posição aleatória.
        """
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.fruta = Point(x, y)
        if self.fruta in self.cobrinha:
            self._gerar_fruta()
        
    def play_step(self, action):
        """
        Executa um passo do jogo baseado na ação fornecida.
        
        Args:
            action (list): Ação a ser executada [frente, esquerda, direita].
        
        Returns:
            tuple: Recompensa, estado de game over e pontuação.
        """
        self.frame_iteration +=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
        # Movimentação
        self._mover(action) # atualiza cabeca
        self.cobrinha.insert(0, self.cabeca)
        
        # Valida o Game Over
        recompensa = 0
        game_over = False
        if self.valida_colisao():
            game_over = True
            recompensa = -10
            return recompensa, game_over, self.pontuacao
            
        if self.frame_iteration > 50*len(self.cobrinha):
            game_over = True
            recompensa = -30
            return recompensa, game_over, self.pontuacao
        
        # Coloca nova fruta ou apenas move
        if self.cabeca == self.fruta:
            self.pontuacao += 1
            recompensa = 10
            self._gerar_fruta()
        else:
            self.cobrinha.pop()
        
        # Atualiza a interface e o relógio
        self._update_ui()
        self.clock.tick(SPEED)
        
        return recompensa, game_over, self.pontuacao
    
    def valida_colisao(self, pt=None):
        """
        Verifica se houve colisão.
        
        Args:
            pt (Point): Ponto a ser verificado. Se None, verifica a cabeça da cobrinha.
        
        Returns:
            bool: True se houve colisão, False caso contrário.
        """
        if pt is None:
            pt = self.cabeca

        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        if pt in self.cobrinha[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        """
        Atualiza a interface do usuário.
        """
        self.display.fill(FUNDO)
        
        # Desenha a grade
        self._draw_grid()
        
        for pt in self.cobrinha:
            pygame.draw.rect(self.display, COR_COBRA, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), border_radius=5)
            pygame.draw.rect(self.display, COR_COBRA_BORDA, pygame.Rect(pt.x+4, pt.y+4, 12, 12), border_radius=5)
            
        pygame.draw.rect(self.display, COR_FRUTA, pygame.Rect(self.fruta.x, self.fruta.y, BLOCK_SIZE, BLOCK_SIZE), border_radius=5)
        
        text = font.render("Pontuação: " + str(self.pontuacao), True, BRANCO)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _draw_grid(self):
        """
        Desenha a grade no fundo da tela.
        """
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, CINZA, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, CINZA, (0, y), (self.w, y))
        
    def _mover(self, action):
        """
        Move a cobrinha baseado na ação fornecida.
        
        Args:
            action (list): Ação a ser executada [frente, esquerda, direita].
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # Sem mudança
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx  + 1) % 4
            new_dir = clock_wise[next_idx] # Vira à direita
        else:
            next_idx = (idx  - 1) % 4
            new_dir = clock_wise[next_idx] # Vira à esquerda
        
        self.direction = new_dir
        
        x = self.cabeca.x
        y = self.cabeca.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.cabeca = Point(x, y)