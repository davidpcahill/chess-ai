import chess

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []

    def reset(self):
        self.board.reset()
        self.move_history = []
        return self.get_state()

    def step(self, action):
        move = chess.Move.from_uci(action)
        if move in self.board.legal_moves:
            san_move = self.board.san(move)  # Get SAN before pushing the move
            self.board.push(move)
            self.move_history.append(san_move)
            done = self.board.is_game_over()
            reward = self.get_reward()
            return self.get_state(), reward, done, {}
        else:
            print(f"Illegal move attempted: {action}")
            print(f"Current board state: {self.board.fen()}")
            print(f"Legal moves: {[move.uci() for move in self.board.legal_moves]}")
            return self.get_state(), -1, True, {}

    def get_state(self):
        state = []
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                state.extend([0] * 12)
            else:
                state.extend([1 if piece.piece_type == pt and piece.color == color else 0
                              for color in [chess.WHITE, chess.BLACK]
                              for pt in range(1, 7)])
        state.append(1 if self.board.turn == chess.WHITE else 0)  # Add current player to state
        return state

    def get_reward(self):
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or \
             self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return 0
        else:
            return 0

    def render(self):
        print(self.board)

    def get_legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]

    def get_result(self):
        if self.board.is_checkmate():
            return "1-0" if self.board.turn == chess.BLACK else "0-1"
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or \
             self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return "1/2-1/2"
        else:
            return None

    def get_move_history(self):
        return " ".join(self.move_history)