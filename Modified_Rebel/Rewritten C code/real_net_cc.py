from .net_interface_h import IValueNet
import torch

class ZeroOutputNet(IValueNet):

    def __init__(self, output_size, verbose)
        self.output_size = output_size
        self.verbose = verbose
    
    def compute_values(self, query):
        num_queries = query.size(0)
        if self.verbose:
            print("Called ZeroOutputNet.handle_nn_query() with num_queries=%d", num_queries)
        return torch.zeros(num_queries, self.output_size)
    
    def add_training_example(self, query, values):
        if self.verbose:
            print("Called ZeroOutputNet.handle_nn_query() with num_queries=%d", query.size(0))

class TorchScriptNet(IValueNet):

    def __init__(self, path, device):
        # Have to figure out what torch::jit:module is, and will complete soon

class OracleNetSolver(IValueNet):

    def __init__(self, game, subgame_solving_params):
        self.game = game
        self.params = subgame_solving_params

    def compute_values(self, queries):
        num_queries = queries.size(0)
        values = []
        for query_id in range(num_queries):
            row = queries[query_id]
            row_values = compute_values_pr(row)
            values.append(row_values)
        return torch.stack(values, 0)
    
    def compute_values_pr(self, queries):
        traverser, state, beliefs1, beliefs2 = deserialize_query(self.game, query)
        beliefs = (beliefs1, beliefs2)
        solver = build_solver(self.game, state, beliefs, self.params)
        solver.multistep()
        return solver.get_hand_values(traverser)

    def add_training_example(self, queries, values):
        raise NotImplementedError("Operation not supported in OracleNet")


def create_zero_net(output_size, verbose):
    pass
# What is std::make_shared<T>