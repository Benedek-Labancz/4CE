import torch
import numpy as np
import math
from copy import deepcopy

from config import device


class Game:
    O, E, X, M = -1, 0, 1, 2
    str_symbols = {O: "O", E: "_", X: "X", M: "M"}
    def __init__(self, num_dimensions, mask_middle):
        self.num_dimensions = num_dimensions
        self.mask_middle = mask_middle
        self.num_actions = 3**num_dimensions
        self.state = self.init_state()
        self.score = self.init_score()
        self.player = self.X # X starts the game by convention

        self.scoring_positions = None # need to be overwritten by the child class

    '''Parent class settings.'''
    def change_encoding(cls, encoding):
        '''
        Change the encoding of the game symbols.
        encoding (tuple): (O, E, X, M)
        '''
        cls.O, cls.E, cls.X, cls.M = encoding

    '''Printing utilities.'''
    def print_score(self, score):
        '''
        Prints game score in the terminal.
        '''
        print(f"[X - {score[self.X]}\tO - {score[self.O]}]")

    '''Game utilities.'''
    def init_state(self):
        '''
        Initialize game state, i.e. a blank board.
        '''
        state = torch.ones(self.num_dimensions * (3,)).to(device) * self.E
        if self.mask_middle:
            state[self.num_dimensions * (1,)] = self.M
        return state

    def init_score(self):
        return {
            self.X: 0,
            self.O: 0
        }

    def switch_turn(self):
        if self.player == self.X:
            self.player = self.O
        else:
            self.player = self.X

    def perform_action(self, state, action, player):
        '''
        Write the player's symbol to the tile defined by action.
        '''
        state[action] = player
        return state

    def valid_action(self, state, action):
        '''
        Check if the chosen action in a state is valid.
        '''
        if type(action) != tuple:
            return False
        if len(action) != self.num_dimensions:
            return False
        for i in range(len(action)):
            try:
                if action[i] < 0 or action[i] > 2:
                    return False
            except:
                return False
        try:
            if state[action] != self.E:
                return False
        except:
            return False
        return True

    def evaluate_action(self, state, action):
        '''
        Count the number of points scored by an action.

        Args:
            state: the state following the action
            action: the action taken by the actor

        I have chosen to include the state following the action in order to not rely on
        having to pass or calculate the player.
        '''
        if self.scoring_positions is None:
            raise Exception("No scoring positions are calculated.")

        def scoring_position(position, clean_state, action):
            if action in position:
                for coord in position:
                    if clean_state[coord] != 1:
                        return False
                return True
            else:
                return False

        player = state[action]
        clean_state = (state == player)
        score = 0
        for position in self.scoring_positions:
            score += int(scoring_position(position, clean_state, action))
        return score

    def terminal_state(self, state):
        '''
        Returns True if game state is terminal,
        i.e. there are no empty tile left.
        '''
        return (state == self.E).sum() == 0

    def determine_winner(self):
        if self.score[self.X] == self.score[self.O]:
            None
        elif self.score[self.X] > self.score[self.O]:
            return self.X
        else:
            return self.O

    def reset(self):
        self.state = self.init_state()
        self.score = self.init_score()
        self.player = self.X

    '''AI Training utilities.'''
    def get_action_mask(self, state):
        """
        Mask all invalid states. 
        Return a binary list indicating valid moves.
        """
        mask = (state == self.E)
        return mask

    def mask_actions(self, action_values, action_mask):
        inf_tensor = -torch.inf * torch.ones_like(action_values)
        action_values = torch.where(action_mask == True, action_values, inf_tensor)
        return action_values

    def fp(self, state, current_player=None):
        '''
        Convert a state to first-person.
        A first-person state's symbols are overwritten to X for the current player
        and O for the opponent.

        Specifying current player is possible and is useful when we only want
        to simulate the environment and not step it actually.
        The default value of current_player is self.player
        '''
        def write_symbols(state, coords, symbol):
            for coord in coords:
                state[tuple(coord)] = symbol
            return state

        if current_player == None: # If current player is not specified
            current_player = self.player

        me_coords = np.transpose((np.array(state) == current_player).nonzero())
        other_player = self.O if current_player == self.X else self.X
        you_coords = np.transpose((np.array(state) == other_player).nonzero())
        blank_state = self.init_state()
        fp_state = write_symbols(blank_state, me_coords, self.X)
        fp_state = write_symbols(fp_state, you_coords, self.O)
        return fp_state



class _2CE(Game):
    def __init__(self):
        super().__init__(num_dimensions=2, mask_middle=False)
        self.scoring_positions = self.calculate_scoring_positions()

    '''Printing utilities.'''
    def print_state(self, state):
        '''
        Prints a state in the terminal.
        Terminal has to be cleared separately for elegance (see example usage).
        '''
        state_str = ""
        x, y = 0, 0
        for i in range(9):
            x = i % 3
            y = math.floor(i / 3)
            tile = self.str_symbols[state[x, y].item()]
            state_str += ' ' + tile
            if x == 2: state_str += '\n'
        print(state_str)

    '''Game utilities.'''
    def step(self, action):
        '''
        Computes the next state of the environment along with the reward
        given the current state of the game and an action.

        The function is designed to look similar to the OpenAI Gym API,
        hence the form of the return is `new_state, reward, done, info`,
        where done indicates whether new_state is terminal, and info is ignored for now.
        '''
        # validate
        if not self.valid_action(self.state, action):
            raise Exception("Invalid action encountered")

        # stepping the env
        new_state = self.perform_action(deepcopy(self.state), action, self.player)
        self.state = new_state

        # bookkeeping
        points_scored = self.evaluate_action(new_state, action)
        self.score[self.player] += points_scored

        reward = points_scored

        done = self.terminal_state(new_state)
        info = '' # TODO

        # prepare next step
        self.switch_turn()

        return new_state, reward, done, info

    def calculate_scoring_positions(self):
        '''
        Calculate all the possible scoring positions in the form of {(x, y), (x, y), (x, y)}
        '''
        def desc(x, y):
            '''
            Descartes product of two tuples.
            e.g.
            x = (0)
            y = (0, 1, 2)
            return [(0, 0), (0, 1), (0, 2)]
            '''  
            fin = []
            for i in range(len(x)):
                for j in range(len(y)):
                    fin.append((x[i], y[j]))
            return fin

        horizontal_xy = [tuple(desc((0, 1, 2), (i,))) for i in [0, 1, 2]]
        vertical_xy = [tuple(desc((i,), (0, 1, 2))) for i in [0, 1, 2]]
        diagonal_xy = [((0, 0), (1, 1), (2, 2)), ((0, 2), (1, 1), (2, 0))]
        xy_options = horizontal_xy + vertical_xy + diagonal_xy # all the possible lineups in 2 dimensions

        return xy_options

    '''AI Training utilities.'''
    def action_to_coords(self, flat_action):
        '''
        Convert a flattened action into a regular action.
        Actions are represented as a single intiger chosen by the agent.
        We need to convert this intiger index into (x, y) coordinates.
        '''
        if flat_action > self.num_actions - 1 or flat_action < 0:
            raise Exception("Invalid action encountered when converting to coords.")
        i = flat_action
        x = i % 3
        y = math.floor(i / 3)
        return (x, y)
    
    def coords_to_action(self, coords):
        '''
        Convert a (x, y) action into a flat action.
        The inverse function of action_to_coords.
        '''
        x, y = coords
        action = 3 * y + x
        return action

    def flat(self, state):
        '''
        Flatten 4-dimensional state.
        (3, 3) -> (9)
        '''
        state = torch.transpose(state, 0, 1)
        state = state.reshape(-1)
        return state
    
    def reshape_state(self, state):
        '''
        Inverse function of flatten_state.
        '''
        state = state.reshape(3, 3)
        state = torch.transpose(state, 1, 0)
        return state



class _4CE:
    def __init__(self):
        self.state = self.init_state()
        self.player = X
        self.score = {
            X: 0,
            O: 0
        }
        self.action_spec = { # the action specification of the game
            "shape": (3, 3, 3, 3),
            "n": 81 # number of all hypothetical actions (middle is always invalid)
        }

        self.scoring_positions = self.calculate_scoring_positions()

    def print_state(self, state):
        '''
        Prints a state in the terminal.
        Terminal has to be cleared separately for elegance (see example usage).
        '''
        state_str = ""
        K, L, x, y = 0, 0, 0, 0
        for i in range(81):
            K = math.floor(i / 3) % 3
            L = math.floor(i / 27)
            x = i % 3
            y = math.floor(i / 9) % 3
            tile = str_symbols[state[K, L, x, y].item()]
            state_str += ' ' + tile
            if x == 2: state_str += ' '
            if K == 2 and x == 2: state_str += '\n'
            if K == 2 and x == 2 and y == 2: state_str += '\n'
        print(state_str)

    def print_score(self, score):
        '''
        Prints game score in the terminal.
        '''
        print(f"[X - {score[X]}\tO - {score[O]}]")

    def step(self, action):
        '''
        Computes the next state of the environment along with the reward
        given the current state of the game and an action.

        The function is designed to look similar to the OpenAI Gym API,
        hence the form of the return is `new_state, reward, done, info`,
        where done indicates whether new_state is terminal, and info is ignored for now.
        Note that the action is not directly preformed, therefore the state of the game
        has to be overwritten manually (see example usage).
        '''
        # validate
        if not self.valid_action(self.state, action):
            raise Exception("Invalid action encountered")

        # stepping the env
        new_state = self.perform_action(deepcopy(self.state), action, self.player)

        # bookkeeping
        points_scored = self.evaluate_action(new_state, action)
        self.score[self.player] += points_scored

        reward = points_scored
        done = self.terminal_state(new_state)
        info = '' # TODO

        return new_state, reward, done, info

    def switch_turn(self):
        if self.player == X:
            self.player = O
        else:
            self.player = X

    def calculate_scoring_positions(self):
        '''
        Calculate all the possible scoring positions in the form of {(K, L, x, y), (K, L, x, y), (K, L, x, y)}
        '''
        '''
        vertical
        {(0, 0), (0, 1), (0, 2)}
        {(1, 0), (1, 1), (1, 2)}
        {(2, 0), (2, 1), (2, 2)}
        
        horizontal
        {(0, 0), (1, 0), (2, 0)}
        {(0, 1), (1, 1), (2, 1)}
        {(0, 2), (1, 2), (2, 2)}

        diagonal
        {(0, 0), (1, 1), (2, 2)}
        {(0, 2), (1, 1), (2, 0)}
        '''
        '''
        column (9) same x, y
        face-diag (8)
        body-diag (4)
        '''
        '''
        Total number of positions: 72 (2d) + 72 (columns) + 2x8x8(body and face diag) = 272
        '''
        def desc(x, y):
            '''
            Descartes product of two tuples.
            e.g.
            x = (0)
            y = (0, 1, 2)
            return [(0, 0), (0, 1), (0, 2)]
            '''  
            fin = []
            for i in range(len(x)):
                for j in range(len(y)):
                    fin.append((x[i], y[j]))
            return fin

        sc = [] # set of all possible scoring positions
        horizontal_xy = [tuple(desc((0, 1, 2), (i,))) for i in [0, 1, 2]]
        vertical_xy = [tuple(desc((i,), (0, 1, 2))) for i in [0, 1, 2]]
        diagonal_xy = [((0, 0), (1, 1), (2, 2)), ((0, 2), (1, 1), (2, 0))]
        xy_options = horizontal_xy + vertical_xy + diagonal_xy # all the possible lineups in 2 dimensions

        # 2 dimensions
        for K in [0, 1, 2]:
            for L in [0, 1, 2]:
                for xy_trio in xy_options:
                    sc.append([(K, L) + xy for xy in xy_trio])

        # face and body diagonals
        for kl_trio in xy_options:
            for xy_trio in xy_options:
                regular = [kl_trio[i] + xy_trio[i] for i in [0, 1, 2]]
                reverse = [kl_trio[i] + xy_trio[2 - i] for i in [0, 1, 2]]
                sc += [regular, reverse]
        
        #columns
        for kl_trio in xy_options:
            for x in [0, 1, 2]:
                for y in [0, 1, 2]:
                    sc.append([kl + (x, y) for kl in kl_trio])   
        return sc

    def init_state(self):
        '''
        Initialize game state, i.e. a blank board.
        '''
        state = torch.ones((3, 3, 3, 3)).to(device) * E
        state[1, 1, 1, 1] = M # middle tile
        return state

    def first_person_state(self, state):
        '''
        Convert a state to first-person.
        A first-person state's symbols are overwritten to X for the current player
        and O for the opponent.
        '''
        def write_symbols(state, coords, symbol):
            for coord in coords:
                state[tuple(coord)] = symbol
            return state

        me_coords = np.transpose((np.array(state) == self.player).nonzero())
        other_player = O if self.player == X else X
        you_coords = np.transpose((np.array(state) == other_player).nonzero())
        blank_state = self.init_state()
        fp_state = write_symbols(blank_state, me_coords, X)
        fp_state = write_symbols(fp_state, you_coords, O)
        return fp_state

    def perform_action(self, state, action, player):
        '''
        Write the player's symbol to the tile defined by action.
        '''
        state[action] = player
        return state

    def valid_action(self, state, action):
        '''
        CHeck if the chosen action in a state is valid.
        '''
        if type(action) != tuple:
            return False
        if len(action) != 4:
            return False
        for i in range(len(action)):
            try:
                if action[i] < 0 or action[i] > 2:
                    return False
            except:
                return False
        try:
            if state[action] != E:
                return False
        except:
            return False
        return True

    def get_action_mask(self, state):
        """
        Mask all invalid states. 
        Return a binary list indicating valid moves.
        """
        mask = (state == E)
        return mask
    
    def mask_actions(self, action_values, action_mask):
        inf_tensor = -torch.inf * torch.ones_like(action_values)
        action_values = torch.where(action_mask == True, action_values, inf_tensor)
        return action_values
    
    def action_to_coords(self, flat_action):
        '''
        Convert a flattened action into a regular action.
        Actions are represented as a single intiger chosen by the agent.
        We need to convert this intiger index into (K, L, x, y) coordinates.
        '''
        i = flat_action
        K = math.floor(i / 9) % 3
        L = math.floor(i / 27)
        x = int(i % 3)
        y = math.floor(i / 3) % 3
        return (K, L, x, y)

    def coords_to_action(self, coords):
        '''
        Convert a (K, L, x, y) action into a flat action.
        The inverse function of action_to_coords.
        '''
        K, L, x, y = coords
        action = K * 27 + L * 9 + x + y * 3
        return action

    def flatten_state(self, state):
        '''
        Flatten 4-dimensional state.
        (3, 3, 3, 3) -> (81)
        '''
        state = torch.transpose(state, 2, 3)
        state = torch.transpose(state, 0, 1)
        state = state.reshape(-1)
        return state
    
    def reshape_state(self, state):
        '''
        Inverse function of flatten_state.
        '''
        state = state.reshape(3, 3, 3, 3)
        state = torch.transpose(state, 1, 0)
        state = torch.transpose(state, 3, 2)
        return state
    
    def evaluate_action(self, state, action):
        '''
        Count the number of points scored by an action.

        Args:
            state: the state following the action
            action: the action taken by the actor

        I have chosen to include the state following the action in order to not rely on
        having to pass or calculate the player.
        '''

        def scoring_position(position, clean_state, action):
            if action in position:
                for coord in position:
                    if clean_state[coord] != 1:
                        return False
                return True
            else:
                return False

        player = state[action]
        clean_state = (state == player)
        score = 0
        for position in self.scoring_positions:
            score += int(scoring_position(position, clean_state, action))
        return score

    def terminal_state(self, state):
        '''
        Returns True if game state is terminal,
        i.e. there are no empty tile left.
        '''
        return (state == E).sum() == 0
    
    def reset(self):
        self.state = self.init_state()
        self.player = X
        self.score = {
            X: 0,
            O: 0
        }

class _3CE:
    def __init__(self):
        self.state = self.init_state()
        self.player = X
        self.score = {
            X: 0,
            O: 0
        }
        self.action_spec = { # the action specification of the game
            "shape": (3, 3, 3),
            "n": 27 # number of all hypothetical actions (middle is always invalid)
        }

        self.scoring_positions = self.calculate_scoring_positions()

    def print_state(self, state):
        '''
        Prints a state in the terminal.
        Terminal has to be cleared separately for elegance (see example usage).
        '''
        state_str = ""
        L, x, y = 0, 0, 0
        for i in range(27):
            L = math.floor(i / 9)
            x = i % 3
            y = math.floor(i / 3) % 3
            tile = str_symbols[state[L, x, y].item()]
            state_str += ' ' + tile
            if x == 2: state_str += '\n'
            if y == 2 and x == 2: state_str += '\n'
        print(state_str)

    def print_score(self, score):
        '''
        Prints game score in the terminal.
        '''
        print(f"[X - {score[X]}\tO - {score[O]}]")

    def step(self, action):
        '''
        Computes the next state of the environment along with the reward
        given the current state of the game and an action.

        The function is designed to look similar to the OpenAI Gym API,
        hence the form of the return is `new_state, reward, done, info`,
        where done indicates whether new_state is terminal, and info is ignored for now.
        Note that the action is not directly preformed, therefore the state of the game
        has to be overwritten manually (see example usage).
        '''
        # validate
        if not self.valid_action(self.state, action):
            raise Exception("Invalid action encountered")

        # stepping the env
        new_state = self.perform_action(deepcopy(self.state), action, self.player)

        # bookkeeping
        points_scored = self.evaluate_action(new_state, action)
        self.score[self.player] += points_scored

        reward = points_scored
        done = self.terminal_state(new_state)
        info = '' # TODO

        return new_state, reward, done, info

    def switch_turn(self):
        if self.player == X:
            self.player = O
        else:
            self.player = X

    def calculate_scoring_positions(self):
        '''
        Calculate all the possible scoring positions in the form of {(L, x, y), (L, x, y), (L, x, y)}
        '''
        def desc(x, y):
            '''
            Descartes product of two tuples.
            e.g.
            x = (0)
            y = (0, 1, 2)
            return [(0, 0), (0, 1), (0, 2)]
            '''  
            fin = []
            for i in range(len(x)):
                for j in range(len(y)):
                    fin.append((x[i], y[j]))
            return fin

        sc = [] # set of all possible scoring positions
        horizontal_xy = [tuple(desc((0, 1, 2), (i,))) for i in [0, 1, 2]]
        vertical_xy = [tuple(desc((i,), (0, 1, 2))) for i in [0, 1, 2]]
        diagonal_xy = [((0, 0), (1, 1), (2, 2)), ((0, 2), (1, 1), (2, 0))]
        xy_options = horizontal_xy + vertical_xy + diagonal_xy # all the possible lineups in 2 dimensions

        # 2 dimensions
        for L in [0, 1, 2]:
            for xy_trio in xy_options:
                sc.append([(L,) + xy for xy in xy_trio])

        # face and body diagonals
        for xy_trio in xy_options:
            regular = [(i,) + xy_trio[i] for i in [0, 1, 2]]
            reverse = [(i,) + xy_trio[2 - i] for i in [0, 1, 2]]
            sc += [regular, reverse]
        
        #columns
        for x in [0, 1, 2]:
            for y in [0, 1, 2]:
                sc.append([k + (x, y) for k in [0, 1, 2]])   
        return sc

    def init_state(self):
        '''
        Initialize game state, i.e. a blank board.
        '''
        state = torch.zeros((3, 3, 3)).to(device)
        return state

    def first_person_state(self, state):
        '''
        Convert a state to first-person.
        A first-person state's symbols are overwritten to X for the current player
        and O for the opponent.
        '''
        def write_symbols(state, coords, symbol):
            for coord in coords:
                state[tuple(coord)] = symbol
            return state

        me_coords = np.transpose((np.array(state) == self.player).nonzero())
        other_player = O if self.player == X else X
        you_coords = np.transpose((np.array(state) == other_player).nonzero())
        blank_state = self.init_state()
        fp_state = write_symbols(blank_state, me_coords, X)
        fp_state = write_symbols(fp_state, you_coords, O)
        return fp_state

    def perform_action(self, state, action, player):
        '''
        Write the player's symbol to the tile defined by action.
        '''
        state[action] = player
        return state

    def valid_action(self, state, action):
        '''
        CHeck if the chosen action in a state is valid.
        '''
        if type(action) != tuple:
            return False
        if len(action) != 3:
            return False
        for i in range(len(action)):
            try:
                if action[i] < 0 or action[i] > 2:
                    return False
            except:
                return False
        try:
            if state[action] != E:
                return False
        except:
            return False
        return True

    def get_action_mask(self, state):
        """
        Mask all invalid states. 
        Return a binary list indicating valid moves.
        """
        mask = (state == E)
        return mask
    
    def mask_actions(self, action_values, action_mask):
        inf_tensor = -torch.inf * torch.ones_like(action_values)
        action_values = torch.where(action_mask == True, action_values, inf_tensor)
        return action_values
    
    def action_to_coords(self, flat_action):
        '''
        Convert a flattened action into a regular action.
        Actions are represented as a single intiger chosen by the agent.
        We need to convert this intiger index into (L, x, y) coordinates.
        '''
        i = flat_action
        L = math.floor(i / 9)
        x = int(i % 3)
        y = math.floor(i / 3) % 3
        return (L, x, y)

    def coords_to_action(self, coords):
        '''
        Convert a (K, L, x, y) action into a flat action.
        The inverse function of action_to_coords.
        '''
        L, x, y = coords
        action = (9 * L) + (3 * y) + X
        return action

    def flatten_state(self, state):
        '''
        Flatten 4-dimensional state.
        (3, 3, 3) -> (27)
        '''
        state = torch.transpose(state, 0, 1)
        state = state.reshape(-1)
        return state
    
    def reshape_state(self, state):
        '''
        Inverse function of flatten_state.
        '''
        state = state.reshape(3, 3, 3)
        state = torch.transpose(state, 1, 0)
        return state
    
    def evaluate_action(self, state, action):
        '''
        Count the number of points scored by an action.

        Args:
            state: the state following the action
            action: the action taken by the actor

        I have chosen to include the state following the action in order to not rely on
        having to pass or calculate the player.
        '''

        def scoring_position(position, clean_state, action):
            if action in position:
                for coord in position:
                    if clean_state[coord] != 1:
                        return False
                return True
            else:
                return False

        player = state[action]
        clean_state = (state == player)
        score = 0
        for position in self.scoring_positions:
            score += int(scoring_position(position, clean_state, action))
        return score

    def terminal_state(self, state):
        '''
        Returns True if game state is terminal,
        i.e. there are no empty tile left.
        '''
        return (state == E).sum() == 0
    
    def reset(self):
        self.state = self.init_state()
        self.player = X
        self.score = {
            X: 0,
            O: 0
        }

