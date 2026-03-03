import numpy as np
import random

class GoGame:
    def __init__(self, size=5):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1
        self.pass_count = 0
        self.is_game_over_flag = False

    def copy(self):
        g = GoGame(self.size)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.pass_count = self.pass_count
        g.is_game_over_flag = self.is_game_over_flag
        return g

    def get_valid_moves(self):
        moves = []
        for i in range(self.size * self.size):
            r, c = divmod(i, self.size)
            if self.board[r][c] == 0:
                # Basic check: could add suicide/ko here for more accuracy
                moves.append(i)
        return moves

    def play(self, move):
        if move is None:
            self.pass_count += 1
            if self.pass_count >= 2:
                self.is_game_over_flag = True
            self.current_player *= -1
            return True

        r, c = divmod(move, self.size)
        if self.board[r][c] != 0:
            return False

        self.board[r][c] = self.current_player
        self.current_player *= -1
        self.pass_count = 0
        return True

    def pass_move(self):
        self.play(None)

    def is_game_over(self):
        return self.is_game_over_flag or not (self.board == 0).any()

    def get_winner(self):
        score = np.sum(self.board)
        if score > 0:
            return 1
        elif score < 0:
            return -1
        return 0

    def get_state(self):
        black = (self.board == 1).astype(np.float32)
        white = (self.board == -1).astype(np.float32)
        player = np.full_like(black, self.current_player, dtype=np.float32)
        return np.stack([black, white, player])

class MCTS:
    def __init__(self, simulations=100):
        self.simulations = simulations

    def search(self, root_game):
        visit_counts = {}
        for _ in range(self.simulations):
            game = root_game.copy()
            path = []
            while not game.is_game_over():
                moves = game.get_valid_moves()
                if not moves:
                    game.pass_move()
                    continue
                move = random.choice(moves)
                path.append((game.current_player, move))
                game.play(move)

            winner = game.get_winner()
            for player, move in path:
                if move not in visit_counts:
                    visit_counts[move] = 0
                if player == winner:
                    visit_counts[move] += 1

        total = sum(visit_counts.values()) + 1e-6
        policy = np.zeros(root_game.size * root_game.size)
        for m, v in visit_counts.items():
            policy[m] = v / total

        return policy
