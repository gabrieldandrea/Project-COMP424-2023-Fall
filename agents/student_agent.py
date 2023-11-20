# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        root = Node(chess_board, my_pos, adv_pos, max_step, 2)

        # Perform MCTS iterations
        for _ in range(250):
            node = root
            while not node.is_terminal():
                if not len(node.children) > 10:
                    node = node.expand()
                    break
                else:
                    node = node.select_child()

            result = node.get_score()
            node.backpropagate(result)

        # Select the best move based on visits
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.my_pos, self.dir_map[best_child.get_wall_direction()]


class Node:
    def __init__(self, chess_board, my_pos, adv_pos, max_steps, exploration_constant):
        self.parent = None
        self.children = []
        self.board_size = len(chess_board)
        self.s = 0
        self.visits = 0
        self.board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_steps = max_steps
        self.exploration_constant = exploration_constant  # Constant for UCB1 exploration
        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        exploitation = self.s / self.visits
        exploration = self.exploration_constant * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self):
        return max(self.children, key=lambda child: child.ucb1())

    def expand(self):
        possible_moves = self.get_possible_moves(True, self.my_pos, self.max_steps)
        for move in possible_moves:
            new_board = deepcopy(self.board)
            child = Node(new_board, move, self.adv_pos, self.max_steps, self.exploration_constant)
            child.parent = self
            self.children.append(child)
        return self.select_child()

    def get_score(self):
        my_available_moves = len(self.get_possible_moves(True, self.my_pos, self.max_steps))
        adv_available_moves = len(self.get_possible_moves(False, self.adv_pos, self.max_steps))

        return my_available_moves - adv_available_moves

    def get_possible_moves(self, me, cur_pos, distance):
        if distance == 0:
            return set()

        valid_moves = set()
        other = self.adv_pos if me else self.my_pos
        # BFS
        state_queue = [(cur_pos, 0)]
        visited = {tuple(cur_pos)}

        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_steps:
                break
            for dir, move in enumerate(self.moves):
                if self.board[r, c, dir]:
                    continue

                next_pos = np.add(cur_pos, move)
                if np.array_equal(next_pos, other) or tuple(next_pos) in visited:
                    continue

                # check if box in
                count = 0
                for dir in self.board[next_pos[0], next_pos[1]]:
                    if dir:
                        count += 1

                if count == 3:
                    continue

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
                # valid_moves.add(tuple(next_pos))

        sorted_moves = sorted(visited, key=lambda move: abs(move[0] - other[0]) + abs(move[1] - other[1]), reverse=True)
        return sorted_moves

    def check_valid_step(self, me, start_pos, end_pos):
        # Endpoint already has barrier or is border
        r, c = end_pos

        if np.array_equal(start_pos, end_pos):
            return True

        # Get position of the adversary
        adv_pos = self.adv_pos if me else self.my_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_steps:
                break
            for dir, move in enumerate(self.moves):
                if self.board[r, c, dir]:
                    continue

                next_pos = np.add(cur_pos, move)
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    def is_within_board(self, pos):
        return 0 <= pos[0] < self.board_size and 0 <= pos[1] < self.board_size

    def backpropagate(self, result):
        self.visits += 1
        self.s += result
        if self.parent:
            self.parent.backpropagate(result)

    def get_wall_direction(self):
        y_diff = self.adv_pos[0] - self.my_pos[0]
        x_diff = self.adv_pos[1] - self.my_pos[1]

        available_positions = [not self.board[self.my_pos[0], self.my_pos[1], i] for i in range(4)]

        if abs(x_diff) > abs(y_diff):
            # Place wall horizontally
            if x_diff > 0:
                if available_positions[1]:
                    return "r"
                elif available_positions[2]:
                    return "d"
                elif available_positions[3]:
                    return "l"
                else:
                    return "u"
            else:
                if available_positions[3]:
                    return "l"
                elif available_positions[0]:
                    return "u"
                elif available_positions[1]:
                    return "r"
                else:
                    return "d"
        else:
            # Place wall vertically
            if y_diff < 0:
                if available_positions[0]:
                    return "u"
                elif available_positions[1]:
                    return "r"
                elif available_positions[2]:
                    return "d"
                else:
                    return "l"
            else:
                if available_positions[2]:
                    return "d"
                elif available_positions[3]:
                    return "l"
                elif available_positions[0]:
                    return "u"
                else:
                    return "r"

    def is_terminal(self):
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(self.moves[1:3]):
                    if self.board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(self.my_pos))
        p1_r = find(tuple(self.adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie

        return True
