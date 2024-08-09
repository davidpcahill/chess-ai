import chess.pgn
import json
import argparse
from collections import defaultdict

def analyze_games(games_dir, num_games=None):
    stats = defaultdict(int)
    move_frequencies = defaultdict(int)
    game_lengths = []

    for i, pgn_file in enumerate(os.listdir(games_dir)):
        if num_games and i >= num_games:
            break
        
        with open(os.path.join(games_dir, pgn_file)) as f:
            game = chess.pgn.read_game(f)
            
        result = game.headers["Result"]
        stats[result] += 1
        
        board = game.board()
        for move in game.mainline_moves():
            move_frequencies[move.uci()] += 1
            board.push(move)
        
        game_lengths.append(len(list(game.mainline_moves())))

    stats["total_games"] = sum(stats.values())
    stats["avg_game_length"] = sum(game_lengths) / len(game_lengths)
    stats["most_common_moves"] = sorted(move_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]

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