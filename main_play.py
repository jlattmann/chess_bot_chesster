import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pygame
import chess
import random
from minmax_play import get_best_move, track_position

# === Config ===
BOT_SEARCH_DEPTH = 3

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 512, 512
SQUARE_SIZE = WIDTH // 8
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Learning Chess Bot")

# Initialize chess board
board = chess.Board()
position_history = []

# Randomize bot color
bot_color = random.choice([chess.WHITE, chess.BLACK])
human_color = not bot_color
flip = (human_color == chess.BLACK)

print(f"🤖 Bot is playing as {'White' if bot_color == chess.WHITE else 'Black'}")
print(f"🧑 Human is playing as {'White' if human_color == chess.WHITE else 'Black'}")

# Load promotion piece images
promotion_options = {
    'q': pygame.image.load("pieces/white/Q.png"),
    'r': pygame.image.load("pieces/white/R.png"),
    'b': pygame.image.load("pieces/white/B.png"),
    'n': pygame.image.load("pieces/white/N.png")
}
for key in promotion_options:
    promotion_options[key] = pygame.transform.scale(promotion_options[key], (SQUARE_SIZE, SQUARE_SIZE))

# Draw the board and pieces
def draw_board():
    colors = [pygame.Color("burlywood1"), pygame.Color("saddlebrown")]
    for row in range(8):
        for col in range(8):
            draw_row = row if not flip else 7 - row
            draw_col = col if not flip else 7 - col
            square = chess.square(col, 7 - row)

            color = colors[(row + col) % 2]
            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(draw_col * SQUARE_SIZE, draw_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            )

            piece = board.piece_at(square)
            if piece:
                folder = "white" if piece.color == chess.WHITE else "black"
                symbol = piece.symbol()
                img = pygame.image.load(f"pieces/{folder}/{symbol}.png")
                img = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
                screen.blit(img, pygame.Rect(draw_col * SQUARE_SIZE, draw_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Get clicked square
def get_square_under_mouse(pos):
    x, y = pos
    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE
    if flip:
        col = 7 - col
        row = 7 - row
    return chess.square(col, 7 - row)

# Promotion GUI
def choose_promotion_gui():
    options = ['q', 'r', 'b', 'n']
    option_rects = []
    for i, key in enumerate(options):
        rect = pygame.Rect(i * SQUARE_SIZE, HEIGHT // 2 - SQUARE_SIZE // 2, SQUARE_SIZE, SQUARE_SIZE)
        screen.blit(promotion_options[key], rect)
        option_rects.append((rect, key))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                for rect, key in option_rects:
                    if rect.collidepoint(event.pos):
                        return {
                            'q': chess.QUEEN,
                            'r': chess.ROOK,
                            'b': chess.BISHOP,
                            'n': chess.KNIGHT
                        }[key]

# Main loop
running = True
selected_square = None

# Bot plays first if it's White
if board.turn == bot_color:
    print("🤖 Bot (White) thinking...")
    bot_move = get_best_move(board, depth=BOT_SEARCH_DEPTH)
    if bot_move:
        board.push(bot_move)
        position_history = track_position(board, position_history)

while running:
    draw_board()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and not board.is_game_over():
            if board.turn != human_color:
                continue  # ignore clicks when it's not your turn

            square = get_square_under_mouse(pygame.mouse.get_pos())

            if selected_square is None:
                if board.piece_at(square) and board.piece_at(square).color == human_color:
                    selected_square = square
            else:
                move = chess.Move(selected_square, square)

                # Handle pawn promotion
                if board.piece_at(selected_square).piece_type == chess.PAWN and \
                   ((board.turn == chess.WHITE and chess.square_rank(square) == 7) or \
                    (board.turn == chess.BLACK and chess.square_rank(square) == 0)):
                    promotion_piece = choose_promotion_gui()
                    move = chess.Move(selected_square, square, promotion=promotion_piece)

                if move in board.legal_moves:
                    board.push(move)
                    selected_square = None

                    # Bot responds if game not over
                    if not board.is_game_over() and board.turn == bot_color:
                        print("🤖 Bot thinking...")
                        bot_move = get_best_move(board, depth=BOT_SEARCH_DEPTH)
                        if bot_move:
                            board.push(bot_move)
                            position_history = track_position(board, position_history)

                    if board.is_game_over():
                        print("Game over:", board.result())
                        learn_from_game(position_history, board, bot_color)
                        running = False
                else:
                    selected_square = None

# Final check if game ends outside loop
if board.is_game_over():
    print("Game over:", board.result())
    learn_from_game(position_history, board, bot_color)

pygame.quit()
