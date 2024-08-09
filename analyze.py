import chess
import chess.pgn
import json
import argparse
import os
from collections import defaultdict
import io

def analyze_games(games_dir, num_games=None):
    stats = defaultdict(int)
    move_frequencies = defaultdict(int)
    game_lengths = []
    promotions = defaultdict(int)

    for i, pgn_file in enumerate(os.listdir(games_dir)):
        if num_games and i >= num_games:
            break
        
        with open(os.path.join(games_dir, pgn_file)) as f:
            pgn_content = f.read()
            
        pgn_io = io.StringIO(pgn_content)
        game = chess.pgn.read_game(pgn_io)
        
        if game is None:
            continue
            
        result = game.headers["Result"]
        stats[result] += 1
        
        board = game.board()
        moves = list(game.mainline_moves())
        game_lengths.append(len(moves))
        
        for move in moves:
            move_frequencies[move.uci()] += 1
            if board.piece_at(move.from_square) == chess.PAWN and chess.square_rank(move.to_square) in [0, 7]:
                promotions[board.turn] += 1
            board.push(move)

    stats["total_games"] = sum(stats.values())
    stats["avg_game_length"] = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    stats["most_common_moves"] = sorted(move_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
    stats["white_promotions"] = promotions[chess.WHITE]
    stats["black_promotions"] = promotions[chess.BLACK]

    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze chess games")
    parser.add_argument("--games_dir", required=True, help="Directory containing PGN files")
    parser.add_argument("--num_games", type=int, help="Number of games to analyze (default: all)")
    parser.add_argument("--output", default="stats.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    stats = analyze_games(args.games_dir, args.num_games)
    
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Analysis complete. Results written to {args.output}")