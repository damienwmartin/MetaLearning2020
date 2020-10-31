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