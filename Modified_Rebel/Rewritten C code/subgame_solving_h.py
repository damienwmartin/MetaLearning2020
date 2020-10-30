import pyspiel

class ISubgameSolver()

    def __init__(self):


    @virtual
    def get_values(self, player_id):
        raise NotImplementedError

    @virtual
    def print_strategy(self, path):
        raise NotImplementedError

    @virtual
    def step(self, traverser):
        raise NotImplementedError

    @virtual
    def multistep(self):
        raise NotImplementedError

    @virtual
    def get_strategy(self):
        raise NotImplementedError

    @virtual
    def get_sampling_strategy(self):
        raise NotImplementedError

    @virtual
    def get_belief_propogation_strategy(self):
        raise NotImplementedError

    @virtual
    def update_value_network(self):
        raise NotImplementedError

    @virtual
    def get_tree(self):
        raise NotImplementedError



def get_query(game_name, traverser, state, reaches1, reaches2):
    pass

def get_uniform_strategy(game, tree):
    pass