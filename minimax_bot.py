import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import chess.engine
import atexit
import os

# === Paths & Model Config ===
MODEL_FILE = "model.pth"
STOCKFISH_PATH = "C:\\Projects\\chess_bot\\stockfish\\stockfish-windows-x86-64-avx2.exe"

# === Neural Network Definition ===
class ChessEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = ChessEvalModel()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.MSELoss()

if os.path.exists(MODEL_FILE):
    try:
        model.load_state_dict(torch.load(MODEL_FILE))
        print("✅ Model loaded.")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
else:
    print("ℹ️ No existing model found. Training from scratch.")

# === Stockfish Engine Setup ===
stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
stockfish_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": 1350})
atexit.register(lambda: stockfish_engine.quit())

# === Board Encoding ===
def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        tensor[idx][row][col] = 1
    return torch.tensor(tensor).unsqueeze(0)  # shape: (1, 12, 8, 8)

# === Evaluation Function ===
def evaluate_board(board, perspective_color=chess.WHITE):
    if board.is_checkmate():
        return float('-inf') if board.turn == perspective_color else float('inf')
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    bot_material = 0
    opponent_material = 0
    for square, piece in board.piece_map().items():
        value = piece_values.get(piece.piece_type, 0)
        if piece.color == perspective_color:
            bot_material += value
        else:
            opponent_material += value

    material_score = bot_material - opponent_material
    normalized_material = material_score / 39.0

    with torch.no_grad():
        tensor = board_to_tensor(board)
        eval_score = model(tensor).item()
        scaled_eval = eval_score if board.turn == perspective_color else -eval_score

    return 0.7 * scaled_eval + 0.3 * normalized_material

# === Minimax with Alpha-Beta Pruning ===
def minimax(board, depth, alpha, beta, is_maximizing, perspective_color):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board, perspective_color)

    if is_maximizing:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, perspective_color)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, perspective_color)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# === Bot Move Selection ===
def get_best_move(board, depth=4, track_accuracy=False, accuracy_log=None, reference_engine=None):
    if board.is_game_over():
        return None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    perspective_color = board.turn
    best_move = None
    best_score = float('-inf')

    for move in legal_moves:
        board.push(move)
        score = minimax(board, depth - 1, float('-inf'), float('inf'), False, perspective_color)
        board.pop()

        if best_move is None or score > best_score:
            best_score = score
            best_move = move

    if track_accuracy and reference_engine and accuracy_log is not None:
        try:
            info = reference_engine.play(board, chess.engine.Limit(depth=10))
            accuracy_log.append(1 if best_move == info.move else 0)
        except:
            accuracy_log.append(0)

    return best_move

# === Learning From Game ===
def learn_from_game(position_tensors, board, perspective_color):
    if not position_tensors:
        return None

    tmp_board = chess.Board()
    targets = []

    for move, tensor in zip(board.move_stack, position_tensors):
        if tmp_board.turn == perspective_color:
            try:
                info = stockfish_engine.analyse(tmp_board, chess.engine.Limit(depth=10))
                cp = info["score"].pov(perspective_color).score(mate_score=10000)
                cp = max(min(cp, 1000), -1000)
                targets.append([cp / 1000])
            except Exception as e:
                print(f"[Stockfish Error] {e}")
                return None
        tmp_board.push(move)

    if not targets:
        print("⚠️ No usable targets.")
        return None

    model.train()
    optimizer.zero_grad()
    inputs = torch.cat(position_tensors[:len(targets)])
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    predictions = model(inputs)
    loss = loss_fn(predictions, targets_tensor)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"[Learned] {len(targets)} positions | Loss: {loss.item():.4f}")
    return loss.item()

# === Position Tracker ===
def track_position(board, history):
    history.append(board_to_tensor(board))
    return history