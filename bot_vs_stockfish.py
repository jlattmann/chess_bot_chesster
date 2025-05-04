import os
import json
import time
import random
import chess
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

from minimax_bot import (
    get_best_move,
    learn_from_game,
    track_position,
    stockfish_engine
)

# === Settings ===
MAX_GAMES = 50
DATA_FILE = "game_logs.json"

# === Piece Values for Material Plot ===
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

def get_material_count(board, bot_color):
    white_material = 0
    black_material = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white_material += value
        else:
            black_material += value
    bot_material = white_material if bot_color == chess.WHITE else black_material
    stockfish_material = black_material if bot_color == chess.WHITE else white_material
    return bot_material, stockfish_material

# === Moving Average Function ===
def moving_average(data, window_size=10):
    if not data:
        return []
    return [
        sum(data[max(0, i - window_size + 1):i + 1]) / len(data[max(0, i - window_size + 1):i + 1])
        for i in range(len(data))
    ]

# === Plot Setup ===
plt.ion()
fig = plt.figure(1, figsize=(16, 10))
fig.subplots_adjust(bottom=0.1)

stop = False
num_games = 0
logs = []
loss_log = []
eval_log = []
material_log = []
eval_gain_log = []

# === Load Existing Logs ===
existing_logs = []
if os.path.exists(DATA_FILE):
    try:
        with open(DATA_FILE, 'r') as f:
            existing_logs = json.load(f)
            loss_log = [log['loss'] for log in existing_logs if log['loss'] is not None]
            eval_log = [log['avg_eval'] for log in existing_logs]
    except json.JSONDecodeError:
        print("⚠️ Failed to load game logs. Starting fresh.")
        existing_logs = []

# === Stop Button Callback ===
def stop_callback(event):
    global stop
    stop = True

stop_button_ax = plt.axes([0.81, 0.01, 0.1, 0.05])
stop_button = widgets.Button(stop_button_ax, 'Stop')
stop_button.on_clicked(stop_callback)

# === Main Training Loop ===
while not stop and num_games < MAX_GAMES:
    board = chess.Board()
    position_history = []
    move_count = 0
    bot_color = random.choice([chess.WHITE, chess.BLACK])
    game_num = len(existing_logs) + num_games + 1

    print(f"\n=== Game {game_num} | Bot is {'White' if bot_color == chess.WHITE else 'Black'} ===")

    if board.turn != bot_color:
        result = stockfish_engine.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
        position_history = track_position(board, position_history)

    while not board.is_game_over() and move_count < 150:
        if board.turn == bot_color:
            move = get_best_move(board, depth=3)
            if move is None:
                print("No legal moves. Game ends.")
                break
            board.push(move)
            position_history = track_position(board, position_history)
        else:
            result = stockfish_engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)
            position_history = track_position(board, position_history)

        time.sleep(0.05)
        move_count += 1

    result = board.result()
    print("Game result:", result)

    bot_mat, sf_mat = get_material_count(board, bot_color)
    material_log.append((bot_mat, sf_mat))

    if position_history:
        loss = learn_from_game(position_history, board, bot_color)
        if loss is not None:
            print(f"✅ Trained. Loss: {loss:.4f}")
        else:
            print("⚠️ Model skipped training.")
            loss = None
    else:
        print("⚠️ No positions collected.")
        loss = None

    sf_evals = []
    tmp_board = chess.Board()
    for move in board.move_stack:
        if tmp_board.turn == bot_color:
            try:
                info = stockfish_engine.analyse(tmp_board, chess.engine.Limit(time=0.1))
                cp = info["score"].pov(bot_color).score(mate_score=10000)
                cp = max(min(cp, 1000), -1000)
                normalized = (cp / 1000 + 1) / 2
                sf_evals.append(normalized)
            except Exception as e:
                print(f"Stockfish eval error: {e}")
        tmp_board.push(move)

    avg_eval = sum(sf_evals) / len(sf_evals) if sf_evals else 0.5
    eval_log.append(avg_eval)

    # Eval gain
    early = sf_evals[:len(sf_evals)//3]
    late = sf_evals[-len(sf_evals)//3:]
    eval_gain = (sum(late)/len(late)) - (sum(early)/len(early)) if early and late else 0
    eval_gain_log.append(eval_gain)

    if result == "1-0":
        outcome = "Win" if bot_color == chess.WHITE else "Loss"
    elif result == "0-1":
        outcome = "Win" if bot_color == chess.BLACK else "Loss"
    else:
        outcome = "Draw"

    print(f"🏁 Outcome: {outcome} | Eval: {avg_eval:.2f} | Loss: {loss:.4f}" if loss is not None else "No training this game.")

    game_data = {
        "game": game_num,
        "result": outcome,
        "bot_color": "White" if bot_color == chess.WHITE else "Black",
        "move_count": move_count,
        "avg_eval": avg_eval,
        "loss": loss
    }

    logs.append(game_data)
    loss_log.append(loss)
    num_games += 1

    with open(DATA_FILE, 'w') as f:
        json.dump(existing_logs + logs, f, indent=2)

    full_logs = existing_logs + logs
    outcomes = [g["result"] for g in full_logs]
    all_losses = [g["loss"] for g in full_logs if g["loss"] is not None]
    all_evals = [g["avg_eval"] for g in full_logs]

    window_size = 10
    moving_avg = moving_average(all_evals, window_size)
    global_avg = sum(all_evals) / len(all_evals) if all_evals else 0.5
    eval_gain_avg = moving_average(eval_gain_log, window_size)

    plt.figure(1)
    plt.clf()

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.bar(['Win', 'Loss', 'Draw'],
            [outcomes.count("Win"), outcomes.count("Loss"), outcomes.count("Draw")],
            color=['green', 'red', 'gray'])
    ax1.set_title("Game Outcomes")

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(all_losses, marker='o')
    ax2.set_title("Training Loss")
    ax2.set_xlabel("Game #")
    ax2.set_ylabel("Loss")

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(all_evals, marker='x', label="Avg Eval")
    ax3.plot(moving_avg, linestyle='--', label="10-Game Moving Avg")
    ax3.axhline(global_avg, linestyle=':', color='purple', label=f"Global Avg = {global_avg:.2f}")
    ax3.set_title("Stockfish Eval")
    ax3.set_ylim(0, 1)
    ax3.legend()

    ax4 = fig.add_subplot(3, 2, 4)
    bot_vals = [b for b, _ in material_log]
    sf_vals = [s for _, s in material_log]
    ax4.plot(bot_vals, label='Bot Material', color='blue')
    ax4.plot(sf_vals, label='Stockfish Material', color='orange')
    ax4.set_title("Bot vs Stockfish Material")
    ax4.set_xlabel("Game #")
    ax4.set_ylabel("Piece Value")
    ax4.legend()

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(eval_gain_log, label="Eval Gain (Raw)", alpha=0.4, linestyle='--', marker='x')
    ax5.plot(eval_gain_avg, label="10-Game Moving Avg", color='orange')
    ax5.axhline(0, linestyle=':', color='gray', label="Neutral")
    ax5.set_title("Eval Improvement During Games")
    ax5.set_xlabel("Game #")
    ax5.set_ylabel("Eval Gain (End - Start)")
    ax5.legend()

    plt.draw()
    plt.pause(0.1)
    time.sleep(0.5)

# === Cleanup ===
stockfish_engine.quit()
print("\n✅ Training session complete. Log saved to 'game_logs.json'.")