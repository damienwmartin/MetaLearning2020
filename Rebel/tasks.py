#from full_rebel import ReBeL
from games.coin_game import CoinGame
from games.liars_dice import LiarsDice
from CFR import CFR
import numpy as np
import torch
import torch.nn as nn
from game_tree import recursive_game_tree, PBS


def build_value_net(game):
    """
    Given a game, builds an appropriate neural net that approximates the value of public belief states.
    """
    if isinstance(game, CoinGame):
        return Net2(n_in=8, n_out=2, n_hidden=4, n_layers=1)
    elif isinstance(game, LiarsDice):
        input_size = 2 + game.num_actions + 2*game.num_hands
        return Net2(n_in=input_size, n_out=game.num_hands, n_hidden=256, n_layers=3)
    elif isinstance(game, Othello):
        input_size = 1+ game.board_size ** 2
        return Net2(n_in = input_size, n_out=1, n_hidden=256, n_layers=3) 
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



# game = LiarsDice(num_dice=2, num_faces=3)
# beliefs = np.array([np.ones(game.num_hands) / game.num_hands for i in range(2)])
# pbs = PBS(('root', ), beliefs)
#
#
# """
# G = recursive_game_tree(PBS, game)
# G.build_full_coin_game()
# params = {'dcfr': False, 'linear_update': False, 'num_iters': 10000}
# agent = CFR(game, G, build_value_net(game), beliefs, params)
#
# agent.multistep()
#
# """
# # ReBeL(pbs, game, build_value_net(game))
