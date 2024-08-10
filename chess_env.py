import chess
import chess.pgn
import socket
from datetime import datetime

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []
        self.move_count = 0

    def reset(self):
        self.board.reset()
        self.move_history = []
        self.move_count = 0
        return self.get_state()

    def step(self, action):
        move = chess.Move.from_uci(action)
        if move in self.board.legal_moves:
            san_move = self.board.san(move)
            self.board.push(move)
            self.move_history.append(san_move)
            self.move_count += 1
            done = self.board.is_game_over()
            reward = self.get_reward()
            return self.get_state(), reward, done, {}
        else:
            print(f"Illegal move attempted: {action}")
            print(f"Current board state: {self.board.fen()}")
            print(f"Legal moves: {[move.uci() for move in self.board.legal_moves]}")
            return self.get_state(), -1, False, {"illegal_move": True}

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
        
        # Piece values
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King's value doesn't change
        }
        
        # Calculate material balance
        material_balance = sum(len(self.board.pieces(piece_type, chess.WHITE)) * value
                            for piece_type, value in piece_values.items()) - \
                        sum(len(self.board.pieces(piece_type, chess.BLACK)) * value
                            for piece_type, value in piece_values.items())
        
        # Reward for pawn promotion
        promotion_reward = 0
        last_move = self.board.move_stack[-1] if self.board.move_stack else None
        if last_move and last_move.promotion:
            promotion_reward = piece_values[last_move.promotion] - piece_values[chess.PAWN]
        
        # Penalize for putting pieces in danger
        danger_penalty = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                if self.board.is_attacked_by(not piece.color, square):
                    danger_penalty -= piece_values.get(piece.piece_type, 0) * 0.1
        
        # Combine rewards
        reward = (material_balance + promotion_reward + danger_penalty)
        
        return reward if self.board.turn == chess.WHITE else -reward
    
    def render(self):
        print(self.board)

    def get_legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]

    def board_to_state(self):
        state = []
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                state.extend([0] * 12)
            else:
                state.extend([1 if piece.piece_type == pt and piece.color == color else 0
                            for color in [chess.WHITE, chess.BLACK]
                            for pt in range(1, 7)])
        state.append(1 if self.board.turn == chess.WHITE else 0)
        return state

    def get_pgn(self, white_agent, black_agent, episode=None):
        game = chess.pgn.Game()
        game.headers["Event"] = "AI Training Game"
        game.headers["Site"] = socket.gethostname()
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(episode) if episode is not None else "?"
        game.headers["White"] = f"AI (epsilon: {white_agent.epsilon:.4f}, model: {white_agent.model_file})"
        game.headers["Black"] = f"AI (epsilon: {black_agent.epsilon:.4f}, model: {black_agent.model_file})"
        game.headers["Result"] = self.get_result() or "*"
        game.headers["PlyCount"] = str(len(self.board.move_stack))

        node = game
        for move in self.board.move_stack:
            node = node.add_variation(move)

        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        pgn_string = game.accept(exporter)
        return pgn_string
    
    def is_game_over(self):
        return self.board.is_game_over(claim_draw=True)

    def get_result(self):
        if self.board.is_checkmate():
            return "1-0" if self.board.turn == chess.BLACK else "0-1"
        elif self.board.is_stalemate():
            return "1/2-1/2 (Stalemate)"
        elif self.board.is_insufficient_material():
            return "1/2-1/2 (Insufficient Material)"
        elif self.board.is_seventyfive_moves():
            return "1/2-1/2 (75-move rule)"
        elif self.board.is_fivefold_repetition():
            return "1/2-1/2 (Fivefold Repetition)"
        elif self.board.can_claim_draw():
            return "1/2-1/2 (Draw Claim)"
        elif self.board.is_game_over():
            return "1/2-1/2 (Other Draw)"
        return None

    def get_move_history(self):
        return " ".join(self.move_history)