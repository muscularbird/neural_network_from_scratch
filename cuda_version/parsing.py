from utils import PIECES, POSITIONS
import sys

def parse_line(line):
    chess_board = []
    position_status = [0, 0, 0]
    splited_line_space = line.split(' ')
    chess_board_lines = splited_line_space[0].split('/')
    for chart in chess_board_lines:
        for char in chart:
            if char.isdigit():
                for _ in range(int(char)):
                    chess_board.append(PIECES[""])
            elif char in PIECES:
                chess_board.append(PIECES[char])
            elif char == ' ':
                break
    if len(chess_board) != 64:
        print("Error: Invalid chess board configuration.")
        sys.exit(84)
    position_status = [1 if status in line else 0 for status in POSITIONS]  
    return chess_board, position_status