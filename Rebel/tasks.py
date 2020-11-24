from .Rebel_Alg import ReBeL
from .games import CoinGame
import torch
import torch.nn as nn


def build_value_net(game)
    """
    Given the name of a game, builds an appropriate neural net that approximates the value of public belief states.
    """
    if game == 'CoinGame':
        return Net2()
    elif game =='liarsdice':
        return Net2()
    else:
        raise Exception("Game's value_net is not supported yet.")
    




def build_mlp(
    *,
    n_in,
    n_hidden,
    n_layers,
    out_size=None,
    act=None,
    use_layer_norm=False,
    dropout=0,
):
"""
    Builds the body of the value net
"""

    if act is None:
        act = GELU()
    build_norm_layer = (
        lambda: nn.LayerNorm(n_hidden) if use_layer_norm else nn.Sequential()
    )
    build_dropout_layer = (
        lambda: nn.Dropout(dropout) if dropout > 0 else nn.Sequential()
    )

    last_size = n_in
    vals_net = []
    for _ in range(n_layers):
        vals_net.extend(
            [
                nn.Linear(last_size, n_hidden),
                build_norm_layer(),
                act,
                build_dropout_layer(),
            ]
        )
        last_size = n_hidden
    if out_size is not None:
        vals_net.append(nn.Linear(last_size, out_size))
    return nn.Sequential(*vals_net)



class Net2(nn.Module):
    """
    The model for the value net

    Input_size: 1(which player) + 1(player is acting/not) + num_actions + 2*num_hands (infostate_beliefs)
    Output_size: num_hands (value of the PBS given each hand)
    """
    def __init__(
        self,
        *,
        n_in,
        n_out,
        n_hidden=256,
        use_layer_norm=False,
        dropout=0,
        n_layers=3,
    ):
        super().__init__()

        self.body = build_mlp(
            n_in=n_in,
            n_hidden=n_hidden,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.output = nn.Linear(
            n_hidden if n_layers > 0 else n_in, n_out
        )
        # Make initial predictions closer to 0.
        with torch.no_grad():
            self.output.weight.data *= 0.01
            self.output.bias *= 0.01

    def forward(self, packed_input: torch.Tensor):
        return self.output(self.body(packed_input))


class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)