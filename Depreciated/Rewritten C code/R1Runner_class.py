from .liars_dice_h import LiarsDice_C
import numpy as np
from .recursive_solving_cc import normalize_beliefs_inplace
from subgame_solving_cc import build_solver

from game_tree import recursive_game_tree


class R1Runner:

    def __init__(self, game, params, net, seed, subgame_params):
        self.game = game
        self.net = net
        self.seed = seed
        self.random_action_prob = params.random_action_prob
        self.sample_leaf_ = params.sample_leaf_ # Boolean value
        self.subgame_params = subgame_params

        self.state = None
        self.beliefs = None
        self.num_iters = params.num_iters

        self.tree = recursive
    

    def step():
        self.state = self.game.get_initial_state()
        self.beliefs = ([1/self.game.num_hands]*self.game.num_hands, [1/self.game.num_hands] * self.game.num_hands)

        while not self.game.is_terminal(self.state):
            solver = build_solver(self.game, self.state, self.beliefs, self.subgame_params, self.net)
            
            act_iter = np.random.randint(num_iters)
            for i in range(act_iter):
                solver.step(i % 2)
            self.sample_state()
            for i in range(act_iter, num_iter):
                solver.step(i % 2)
            
            solver.update_value_network()

    def sample_state(solver):
        if self.sample_leaf_:
            return self.sample_state_to_leaf(solver)
        else:
            return self.sample_state_single(solver)

    def sample_state_single():
        
        br_sampler = np.random.randint(2)
        eps = np.random.uniform()
        if state.player_id == br_sampler and eps < self.random_action_prob:
            action_begin, action_end = self.game.get_bid_range()
            action = np.random.randint(action_begin, action_end)
        else:
            beliefs = self.beliefs[state.player_id]
            hand = np.random.choice(len(beliefs), 1, p=beliefs)
            policy = solver.get_sampling_strategy()[0][hand]
            action = np.random.choice(len(policy), 1, p=policy)
        
        policy = solver.get_belief_propagation_strategy()
        for hand in range(self.game.num_hands()):
            self.beliefs[self.state.player_id][hand] *= policy[hand][action]
        normalize_beliefs_inplace(self.beliefs[state.player_id])
        self.state = self.game.act(self.state, action)

    def sample_state_to_leaf(solver):
        tree = solver.get_tree()
        path = []

        node_id = 0
        br_sampler = np.random.randint(2)
        strategy = solver.get_sampling_strategy()
        sampling_beliefs = self.beliefs

        while tree[node_id].num_children():
            eps = np.random.uniform()
            state = tree[node_id].state
            action_begin, action_end = self.game.get_bid_range(state)
            if state.player_id == br_sampler and eps < self.random_action_prob:
                action = np.random.randint(action_begin, action_end)
            else:
                beliefs = sampling_beliefs[state.player_id]
                hand = np.random.choice(len(beliefs), 1, p=beliefs)
                policy = strategy[node_id][hand]
                action = np.random.choice(len(policy), 1, p=policy)
                assert action >= action_begin and action < action_end
            
            policy = strategy[node_id]
            for hand in range(game.num_hands()):
                sampling_beliefs[state.player_id][hand] *= policy[hand][action]
            
            normalize_beliefs_inplace(sampling_beliefs[state.player_id])
            path.append((node_id, action))
            node_id = tree[node_id].children_begin + action - action_begin
        
        for node_id, action in path:
            action_begin = self.game.get_bid_range(state)[0]
            policy = solver.get_belief_propagation_strategy()[node_id]

            for hand in range(self.game.num_hands()):
                self.beliefs[self.state.player_id][hand] = policy[hand][action]
            
            normalize_beliefs_inplace(self.beliefs[state.player_id])
            child_node_id = tree[node_id].children_begin + action - action_begin
            self.state = tree[child_node_id].state