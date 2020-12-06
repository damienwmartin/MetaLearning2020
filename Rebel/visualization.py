import networkx as nx
import numpy as np
from games.liars_dice import LiarsDice
from games.coin_game import CoinGame
from full_rebel import GameTree, Policy, rebel
import matplotlib.pyplot as plt


def build_pos(game, root, pos={}, width_limits=(0,1)):
    """
    Given a networkx graph. builds the proper pos dictionary for the game tree
    """

    pos[root] = (0.5*(width_limits[0] + width_limits[1]), game.max_depth() - len(root))
    legal_moves = game.get_legal_moves(root)
    num_valid_actions = len(legal_moves)
    i = 0
    for action in legal_moves:
        start = width_limits[0] + i*(width_limits[1] - width_limits[0])/num_valid_actions
        build_pos(game, root + (action, ), pos=pos, width_limits=(start, start + (width_limits[1] - width_limits[0])/num_valid_actions))
        i += 1
    
    return pos


def view_full_game_tree(game, depth_limit=100000):
    """
    Takes in a game and visualizes the game tree
    """
    tree = GameTree(game, None, game.get_initial_beliefs())
    tree.build_depth_limited_subgame(depth_limit=depth_limit, for_solving=False)

    pos = build_pos(game, ('root', ))
    nx.draw(tree.tree, pos, with_labels=True)
    plt.savefig('example2.png')


def view_policy(policy, hand=None):
    """
    Allows you to view the strategy given a particular hand, or all at once
    """

    edge_labels = {}
    if hand is None:
        for edge_name in policy.tree.edges:
            edge_labels[edge_name] = policy.tree.nodes[edge_name[0]]['policy']
    else:
        for edge_name in policy.tree.edges:
            edge_labels[edge_name] = policy.tree.nodes[edge_name[0]]['policy'][hand]

    
    pos = build_pos(policy.game, ('root', ))
    nx.draw_networkx_edge_labels(policy.tree, pos, edge_labels)

    plt.savefig('example2.png')


game = LiarsDice(num_dice=1, num_faces=2)
view_full_game_tree(game)