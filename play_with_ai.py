import numpy as np
import torch
import torch.nn as nn
import os

# Gomoku Environment
class GomokuEnv:
    def __init__(self, size=8):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def step(self, action):
        if self.done:
            raise Exception("Game is over")
        row, col = divmod(action, self.size)
        if self.board[row, col] != 0:
            raise Exception("Invalid move")
        self.board[row, col] = self.current_player
        if self._check_winner(row, col):
            self.done = True
            self.winner = self.current_player
            return self.board.copy(), 1, True, f'player {self.current_player} wins'
        if np.all(self.board != 0):
            self.done = True
            return self.board.copy(), 0, True, 'draw'
        self.current_player *= -1
        return self.board.copy(), 0, False, ''

    def _check_winner(self, row, col):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        player = self.board[row, col]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + dr * i, col + dc * i
                if not (0 <= r < self.size and 0 <= c < self.size) or self.board[r, c] != player:
                    break
                count += 1
            for i in range(1, 5):
                r, c = row - dr * i, col - dc * i
                if not (0 <= r < self.size and 0 <= c < self.size) or self.board[r, c] != player:
                    break
                count += 1
            if count >= 5:
                return True
        return False

    def render(self):
        print('   ' + ' '.join(str(i) for i in range(self.size)))
        print('  +' + '-' * (self.size * 2 - 1) + '+')
        for i in range(self.size):
            print(f'{i} |' + ' '.join(['X' if x == 1 else 'O' if x == -1 else '.' for x in self.board[i]]) + '|')

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, model_path='gomoku_dqn.pth'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cpu")
        self.model = self._build_model().to(self.device)
        self.load_model(model_path)

    def _build_model(self):
        model = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.state_size[1] * self.state_size[2], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def load_model(self, model_path):
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        else:
            raise FileNotFoundError(f"No pre-trained model found at {model_path}. Please train the model first.")

    def get_state(self, board, current_player):
        player_pieces = (board == current_player).astype(np.float32)
        opponent_pieces = (board == -current_player).astype(np.float32)
        return np.stack([player_pieces, opponent_pieces], axis=0)

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor).cpu().numpy()[0]
        # Flatten the occupancy check to match act_values' 1D shape
        occupied = (state[0] + state[1]).flatten() != 0
        act_values[occupied] = float('-inf')  # Mask occupied positions
        return np.argmax(act_values)

# Human vs AI Game
def play_human_vs_ai(model_path='gomoku_dqn.pth'):
    env = GomokuEnv(size=8)
    try:
        agent = DQNAgent(state_size=(2, 8, 8), action_size=64, model_path=model_path)
    except FileNotFoundError as e:
        print(e)
        return

    state = env.reset()
    print("Welcome to Gomoku! You are 'O' (-1), AI is 'X' (1)")
    print("Enter moves as 'row col' (e.g., '3 4')")
    env.render()

    while not env.done:
        if env.current_player == 1:  # AI's turn
            print("\nAI's turn...")
            state_channels = agent.get_state(state, 1)
            action = agent.act(state_channels)
            state, _, done, info = env.step(action)
            row, col = divmod(action, env.size)
            print(f"AI placed 'X' at ({row}, {col})")
        else:  # Human's turn
            print("\nYour turn!")
            while True:
                try:
                    move = input("Enter row and column (e.g., '3 4'): ").strip().split()
                    if len(move) != 2:
                        raise ValueError("Please enter two numbers separated by a space")
                    row, col = map(int, move)
                    if not (0 <= row < env.size and 0 <= col < env.size):
                        raise ValueError("Move out of bounds")
                    action = row * env.size + col
                    state, _, done, info = env.step(action)
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}. Try again.")
                except Exception as e:
                    print(f"Error: {e}. Try again.")
        
        env.render()
        if done:
            if env.winner == 1:
                print("AI wins!")
            elif env.winner == -1:
                print("You win!")
            else:
                print("It's a draw!")

if __name__ == "__main__":
    model_path = 'gomoku_dqn3.pth'
    play_human_vs_ai(model_path)