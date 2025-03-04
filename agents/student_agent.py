# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import heapq


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
        for _ in range(200):
            node = root
            while not node.is_terminal():
                if not len(node.children) > 15:
                    node = node.expand()
                    break
                else:
                    node = node.select_child()

                result = node.get_score()
                node.backpropagate(result)

        best_child = max(root.children, key=lambda child: child.ucb())
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
        self.exploration_constant = exploration_constant
        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.num_map = {
            0 :"u",
            1 : "r",
            2 : "d",
            3 : "l"
        }

    def ucb(self):
        '''
        Upper Confidence bound formula taken from the slides
        '''
        if self.visits == 0:
            return float('inf')
        exploitation = self.s / self.visits
        exploration = self.exploration_constant * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self):
        return max(self.children, key=lambda child: child.ucb())

    def expand(self):
        possible_moves = self.get_possible_moves(True, self.my_pos, self.max_steps, True)
        for move in possible_moves:
            new_board = deepcopy(self.board)
            child = Node(new_board, move, self.adv_pos, self.max_steps, self.exploration_constant)
            child.parent = self
            self.children.append(child)
        return self.select_child()

    def get_score(self):
        '''
        The heuristic we are using is the number of available moves based on player positions.
        This function counts available moves of both players and returns the difference
        '''
        my_available_moves = len(self.get_possible_moves(True, self.my_pos, self.max_steps, True))
        adv_available_moves = len(self.get_possible_moves(False, self.adv_pos, self.max_steps, True))

        return my_available_moves - adv_available_moves

    def get_possible_moves(self, me, cur_pos, distance, collision):
        '''
        Function used to count the number of available moves.
        BFS approach taken from world.py
        cur_pos is the position of the player we want to consider
        The me parameter is used to determine whether we are counting moves for our agent or the opponent.
        The collision paramter is used to determine whether we want to consider the other player as a blocker
        Distance is the max_steps
        '''

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
                if (collision and np.array_equal(next_pos, other)) or tuple(next_pos) in visited:
                    continue

                valid_moves.add(tuple(next_pos))

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        sorted_moves = sorted(valid_moves, key=lambda move: abs(move[0] - other[0]) + abs(move[1] - other[1]), reverse=True)
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

    def set_barrier(self, place, r, c, dir):
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}

        # Set the barrier to True
        self.board[r, c, dir] = place
        # Set the opposite barrier to True
        move = self.moves[dir]

        self.board[r + move[0], c + move[1], opposites[dir]] = place

    def get_wall_direction(self):
        '''
        This function determines the best direction to place a wall
        It works by iterating over the available directions, placing the wall in a temporary state
        Then checking how many moves are allowed for both players
        The difference is added to a min heap.
        The direction with the smallest value is taken, since this means
        '''

        r, c = self.my_pos[0], self.my_pos[1]

        available_directions = [not self.board[r, c, i] for i in range(4)]

        move_heap = []

        for i in range(len(available_directions)):
            if not available_directions[i]:
                continue

            # place wall
            self.set_barrier(True, r, c, i)

            # check moves
            my_available_moves = len(self.get_possible_moves(True, self.my_pos, self.max_steps, False))
            adv_available_moves = len(self.get_possible_moves(False, self.adv_pos, self.max_steps, False))

            # remove wall
            self.set_barrier(False, r, c, i)

            # add move [moves, dir]
            heapq.heappush(move_heap, [(adv_available_moves - my_available_moves), i])

        # Check for ties
        print(move_heap)
        best_moves = heapq.nsmallest(2, move_heap)
        print(best_moves)
        if len(best_moves) > 1 and best_moves[0][0] == best_moves[1][0]:
            # There is a tie, check the adv facing direction without a barrier
            y_diff = self.adv_pos[0] - self.my_pos[0]
            x_diff = self.adv_pos[1] - self.my_pos[1]

            if abs(x_diff) > abs(y_diff):
                # Place wall horizontally
                if x_diff < 0:
                    if available_directions[1]:
                        return "r"
                    elif available_directions[2]:
                        return "d"
                    elif available_directions[3]:
                        return "l"
                    else:
                        return "u"
                else:
                    if available_directions[3]:
                        return "l"
                    elif available_directions[0]:
                        return "u"
                    elif available_directions[1]:
                        return "r"
                    else:
                        return "d"
            else:
                # Place wall vertically
                if y_diff < 0:
                    if available_directions[0]:
                        return "u"
                    elif available_directions[1]:
                        return "r"
                    elif available_directions[2]:
                        return "d"
                    else:
                        return "l"
                else:
                    if available_directions[2]:
                        return "d"
                    elif available_directions[3]:
                        return "l"
                    elif available_directions[0]:
                        return "u"
                    else:
                        return "r"

        return self.num_map[heapq.heappop(move_heap)[1]]


    def is_terminal(self):
        '''
        This function checks if the board is in a terminal state'''
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
        if p0_r == p1_r:
            return False

        return True
