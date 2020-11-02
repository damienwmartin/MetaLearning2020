import torch

class IValueNet:

    @virtual
    def __init__():
        raise NotImplementedError

    @virtual
    def compute_values(queries)
        raise NotImplementedError

    @virtual
    def add_training_example(queries, values):
        raise NotImplementedError