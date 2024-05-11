import random
import math
from datetime import datetime
import enum
from typing import List, Tuple


@enum.unique
class State(enum.Enum):
    HIDDEN: int = 0
    REVEALED: int = 1
    FLAGGED: int = 2


class Map:
    def __init__(self):
        self.val: int = 0
        self.state: State = State.HIDDEN


class MinesweeperMap:
    def __init__(self, size: int):
        self.size: int = size
        self.init_map()
        self.turns: int = 0
        self.flags: int = 0
        self.lives: int = 3
        self.num_mines: int = math.ceil(0.15 * (self.size ** 2))
        if self.num_mines < 4:
            self.lives = max(1, self.num_mines - 1) 

    def is_flagged(self, x: int, y: int) -> bool:
        return self.map[x][y].state == State.FLAGGED

    def is_revealed(self, x: int, y: int) -> bool:
        return self.map[x][y].state == State.REVEALED

    def is_hidden(self, x: int, y: int) -> bool:
        return self.map[x][y].state == State.HIDDEN

    def number_of_lives(self) -> int:
        return self.lives

    def number_of_turns_taken(self) -> int:
        return self.turns
    
    def number_of_flags(self) -> int:
        return self.flags

    def number_of_mines(self) -> int:
        return self.num_mines

    def change_number_of_lives(self, lives: int):
        self.lives = lives

    def init_map(self):
        self.map: List[List[Map]] = [[Map() for _ in range(self.size)] for _ in range(self.size)]

    def generate_bombs(self, x: int, y: int):
        seed = random.randint(-3000000,3000000)
        random.seed(seed)

        for _ in range(self.num_mines):
            i: int = random.randrange(0, self.size)
            j: int = random.randrange(0, self.size)
            seed = random.randint(-5000000,5000000)
            random.seed(seed)

            while self.map[i][j].val == -1 or (x == i and y == j):
                i = random.randrange(0, self.size)
                j = random.randrange(0, self.size)
                seed = random.randint(-7000000,7000000)
                random.seed(seed)

            self.map[i][j].val = -1

    def generate_hints(self):
        i, j = 0, 0
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i][j].val != -1:
                    self.map[i][j].val += self.nearby_bombs(i, j)

    def nearby_bombs(self, i: int, j: int) -> int:
        num_bombs: int = 0
        if i > 0:
            num_bombs += 1 if self.map[i - 1][j].val == -1 else 0

        if i < self.size - 1:
            num_bombs += 1 if self.map[i + 1][j].val == -1 else 0

        if j > 0:
            num_bombs += 1 if self.map[i][j - 1].val == -1 else 0

        if j < self.size - 1:
            num_bombs += 1 if self.map[i][j + 1].val == -1 else 0

        if i > 0 and j > 0:
            num_bombs += 1 if self.map[i - 1][j - 1].val == -1 else 0

        if i > 0 and j < self.size - 1:
            num_bombs += 1 if self.map[i - 1][j + 1].val == -1 else 0

        if i < self.size - 1 and j > 0:
            num_bombs += 1 if self.map[i + 1][j - 1].val == -1 else 0

        if i < self.size - 1 and j < self.size - 1:
            num_bombs += 1 if self.map[i + 1][j + 1].val == -1 else 0

        return num_bombs

    def validate_input(self, x: int, y: int) -> bool:
        return x >= 1 and x <= self.size and y >= 1 and y <= self.size

    def validate_mode(self, m: str) -> bool:
        return m == "r" or m == "f" or m == "q"

    def validate_move(self, move: str) -> Tuple[bool, str, int, int]:
        move_list: List[str] = move.split()
        if len(move_list) == 3:
            m = move_list[0]
            if self.validate_mode(move_list[0]):
                try:
                    x: int = int(move_list[1])
                    y: int = int(move_list[2])
                    if self.validate_input(x, y):
                        return (False, m, (x - 1), (y - 1))
                except:
                    return (True, "", -1, -1)
        elif len(move_list) == 1 and move_list[0] == "q":
            return (False, "q", -1, -1)
        return (True, "", -1, -1)

    def get_stats_str(self) -> str:
        stats_str: str = ""
        stats_str += "Lives remaining: " + str(self.lives) + "\t"
        stats_str += "Turns taken: " + str(self.turns) + "\t"
        stats_str += "Flagged places: " + str(self.flags) + "\t"
        stats_str += "Number of Mines: " + str(self.num_mines)
        stats_str += "\n\n"
        return stats_str

    def get_map_str(self) -> str:
        map_str: str = ""
        i: int
        for i in range(self.size):
            j: int
            for j in range(self.size):
                if self.map[i][j].state == State.FLAGGED:
                    map_str += "[|>]\t"
                elif self.map[i][j].state == State.REVEALED:
                    map_str += ((" " + str(self.map[i][j].val)) if self.map[i][j].val != -1 else "[X] ") + "\t"
                else:
                    map_str += "[ ] \t"
            map_str += "\n"
        return map_str

    def map_revealed(self) -> str:
        map_revealed_str: str = ""
        i: int
        for i in range(self.size):
            j: int
            for j in range(self.size):
                map_revealed_str += (str(self.map[i][j].val) if self.map[i][j].val != -1 else "X") + " "
            map_revealed_str += "\n"
        return map_revealed_str

    def accept_input(self) -> Tuple[str, int, int]:
        move: str = input(":")
        err: bool
        m: str
        x: int
        y: int
        err, m, x, y = self.validate_move(move)
        while err:
            print("Please enter valid input\n")
            move = input(":")
            err, m, x, y = self.validate_move(move)
        print()
        return m, x, y

    def reveal(self, x: int, y: int, num_revealed = 0) -> Tuple[int, int]:
        already_revealed: bool
        if self.map[x][y].state != State.REVEALED:
            already_revealed = False

            if self.map[x][y].state == State.FLAGGED:
                self.flags -= 1

            self.map[x][y].state = State.REVEALED

            if self.map[x][y].val == 0:
                if x < self.size - 1 and self.map[x + 1][y].state != State.REVEALED:
                    _, num_revealed = self.reveal(x + 1, y, num_revealed)

                if x > 0 and self.map[x - 1][y].state != State.REVEALED:
                    _, num_revealed = self.reveal(x - 1, y, num_revealed)

                if y > 0 and self.map[x][y - 1].state != State.REVEALED:
                    _, num_revealed = self.reveal(x, y - 1, num_revealed)

                if y < self.size - 1 and self.map[x][y + 1].state != State.REVEALED:
                    _, num_revealed = self.reveal(x, y + 1, num_revealed)

                if x > 0 and y > 0 and self.map[x - 1][y - 1].state != State.REVEALED:
                    _, num_revealed = self.reveal(x - 1, y - 1, num_revealed)

                if x > 0 and y < self.size - 1 and self.map[x - 1][y + 1].state != State.REVEALED:
                    _, num_revealed = self.reveal(x - 1, y + 1, num_revealed)

                if x < self.size - 1 and y > 0 and self.map[x + 1][y - 1].state != State.REVEALED:
                    _, num_revealed = self.reveal(x + 1, y - 1, num_revealed)

                if x < self.size - 1 and y < self.size - 1 and self.map[x + 1][y + 1].state != State.REVEALED:
                    _, num_revealed = self.reveal(x + 1, y + 1, num_revealed)

        return self.map[x][y].val, num_revealed + (0 if already_revealed else 1)

    def flag(self, x: int, y: int):
        if self.map[x][y].state == State.REVEALED:
            print("Invalid location: Already revealed")
        elif self.map[x][y].state == State.FLAGGED:
            self.map[x][y].state = State.HIDDEN
            self.flags -= 1
        else:
            self.map[x][y].state = State.FLAGGED
            self.flags += 1

    def get_play_str(self) -> str:
        play_str: str = ""
        play_str += self.get_stats_str()
        play_str += self.get_map_str()
        return play_str 

    def generate_map(self, x: int, y: int):
        self.generate_bombs(x, y)
        self.generate_hints()

    def save_map(self, out = print):
        out(str(self.size) + "\n" + self.get_play_str() + self.map_revealed())

    def export_map(self, out = print):
        self.generate_map(self.size//2, self.size//2)
        _map = str(self.size) + " " + str(self.num_mines) + "\n" + self.map_revealed()
        #out(_map)
        return _map

    def play(self, out = print) -> int:
        result: int = 0
        out(self.get_play_str())        
        m: str
        x: int
        y: int
        m, x, y = self.accept_input()

        if m != "q":
            while m != "r":
                self.flag(x, y)
                out(self.get_play_str())        
                m, x, y = self.accept_input()

            self.generate_map(x, y)
            remaining_loc: int = (self.size ** 2) - self.num_mines

            while m != "q":
                if m == "r":
                    val: int
                    locs_revealed: int
                    val, locs_revealed = self.reveal(x, y)
                    if val == -1:
                        if self.lives > 0:
                            self.lives -= 1
                            out("1 life lost!")
                            out("\n")
                        else:
                            result = -1
                    else:
                        remaining_loc -= locs_revealed
                        if remaining_loc == 0:
                            result = 1
                else:
                    self.flag(x, y)
                self.turns += 1
                out(self.get_play_str())
                out("\n")
                if result != 0:
                    break
                m, x, y = self.accept_input()

        if m == "q":
            result = 0
            out("Be back soon!")
        else:
            out("Congrats!" if result == 1 else "Better luck next time!")
        out("\n")

        return result

