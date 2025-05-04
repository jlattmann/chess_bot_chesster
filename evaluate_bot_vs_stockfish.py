import chess
import chess.engine
import random
import time
import statistics
import os
import matplotlib.pyplot as plt
from minimax_bot import get_best_move

# === Config ===
STOCKFISH_PATH = "C:\\Projects\\chess_bot\\stockfish\\stockfish-windows-x86-64-avx2.exe"
ELO_LEVELS = ["BadBot", "WeakBot", 1350]
GAMES_PER_LEVEL = 2
DEPTH = 3
MAX_MOVES = 400

results_per_elo = {}

# === Full-Strength Stockfish for Accuracy Reference ===
reference_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# === BadBot: Random Capturer (~400 Elo) ===
def very_bad_bot_move(board):
    capture_moves = [m for m in board.legal_moves if board.is_capture(m)]
    if capture_moves:
        return random.choice(capture_moves)
    return random.choice(list(board.legal_moves))

# === WeakBot: Material Greedy (~600-800 Elo) ===
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9
}

def weak_bot_move(board):
    best_score = -float('inf')
    best_move = None
    for move in board.legal_moves:
        board.push(move)
        score = 0
        for square, piece in board.piece_map().items():
            value = PIECE_VALUES.get(piece.piece_type, 0)
            score += value if piece.color == board.turn else -value
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move if best_move else random.choice(list(board.legal_moves))

# === Run Evaluation Per Elo Level ===
for elo in ELO_LEVELS:
    print(f"\n🔍 Evaluating bot vs {elo}")

    if elo not in ["BadBot", "WeakBot"]:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    else:
        engine = None

    evals = []
    accuracies = []
    wins = 0
    losses = 0
    draws = 0

    for game_num in range(1, GAMES_PER_LEVEL + 1):
        board = chess.Board()
        bot_color = random.choice([chess.WHITE, chess.BLACK])
        move_count = 0
        accuracy_log = []

        if board.turn != bot_color:
            if elo == "BadBot":
                board.push(very_bad_bot_move(board))
            elif elo == "WeakBot":
                board.push(weak_bot_move(board))
            else:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)

        while not board.is_game_over() and move_count < MAX_MOVES:
            if board.turn == bot_color:
                move = get_best_move(board, depth=DEPTH, track_accuracy=True, accuracy_log=accuracy_log, reference_engine=reference_engine)
            else:
                if elo == "BadBot":
                    move = very_bad_bot_move(board)
                elif elo == "WeakBot":
                    move = weak_bot_move(board)
                else:
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    move = result.move

            if move:
                board.push(move)

            move_count += 1
            time.sleep(0.01)

        result = board.result(claim_draw=True)

        if result == "1-0":
            if bot_color == chess.WHITE:
                wins += 1
            else:
                losses += 1
        elif result == "0-1":
            if bot_color == chess.BLACK:
                wins += 1
            else:
                losses += 1
        elif result == "1/2-1/2":
            draws += 1

        # === Evaluation ===
        if elo in ["BadBot", "WeakBot"]:
            score = 0
            for square, piece in board.piece_map().items():
                value = PIECE_VALUES.get(piece.piece_type, 0)
                score += value if piece.color == bot_color else -value
            normalized = (score + 39) / 78
            avg_eval = max(0.0, min(1.0, normalized))
        else:
            tmp_board = chess.Board()
            sf_evals = []
            for move in board.move_stack:
                if tmp_board.turn == bot_color:
                    try:
                        info = engine.analyse(tmp_board, chess.engine.Limit(time=0.1))
                        cp = info["score"].pov(bot_color).score(mate_score=10000)
                        cp = max(min(cp, 1000), -1000)
                        normalized = (cp / 1000 + 1) / 2
                        sf_evals.append(normalized)
                    except:
                        pass
                tmp_board.push(move)
            avg_eval = sum(sf_evals) / len(sf_evals) if sf_evals else 0.5

        evals.append(avg_eval)
        accuracy_percent = 100 * sum(accuracy_log) / len(accuracy_log) if accuracy_log else 0
        accuracies.append(accuracy_percent)

        print(f"  Game {game_num}/{GAMES_PER_LEVEL} → Result: {result} | Avg Eval: {avg_eval:.2f} | Accuracy: {accuracy_percent:.1f}%")

    if engine:
        engine.quit()

    results_per_elo[elo] = {
        "evals": evals,
        "accuracies": accuracies,
        "wins": wins,
        "losses": losses,
        "draws": draws
    }

# === Cleanup Reference Engine ===
reference_engine.quit()

# === Results Summary ===
print("\n📊 Evaluation Summary:")
for elo, result_data in results_per_elo.items():
    label = elo
    mean_eval = statistics.mean(result_data["evals"])
    mean_acc = statistics.mean(result_data["accuracies"])
    print(f"  {label}: Avg Eval = {mean_eval:.3f} | Accuracy = {mean_acc:.1f}% | W:{result_data['wins']} D:{result_data['draws']} L:{result_data['losses']}")

# === Plotting ===
plt.figure(figsize=(14, 6))

# Plot Eval
plt.subplot(1, 2, 1)
for elo, result_data in results_per_elo.items():
    plt.plot(result_data["evals"], marker='o', label=f"{elo} Eval")
plt.axhline(0.5, color='gray', linestyle='--', label='Neutral')
plt.ylim(0, 1)
plt.title("Normalized Avg Eval")
plt.xlabel("Game")
plt.ylabel("Eval")
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
for elo, result_data in results_per_elo.items():
    plt.plot(result_data["accuracies"], marker='x', label=f"{elo} Accuracy")
plt.ylim(0, 100)
plt.title("Bot Accuracy vs Stockfish")
plt.xlabel("Game")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
