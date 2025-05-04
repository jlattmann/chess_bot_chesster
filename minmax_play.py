import os
import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# === Paths & Model Config ===
MODEL_FILE = "model.pth"

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

# === Board Encoding ===
def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        tensor[idx][row][col] = 1
    return torch.tensor(tensor).unsqueeze(0)

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
def get_best_move(board, depth=4):
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

    return best_move

# === Position Tracker ===
def track_position(board, history):
    history.append(board_to_tensor(board))
    return history