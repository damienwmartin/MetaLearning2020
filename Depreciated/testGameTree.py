'''
Unit tests for game_tree.py
'''
import unittest
from games.coin_game import CoinGame
from game_tree import recursive_game_tree, PBS

class TestGameTree(unittest.TestCase):
    #TODO: Confirm expected representations
    def test_build_depth_limited_subgame(self):
        with self.subTest('depth 1 from root'):
            pbs = PBS([], [])
            game = CointGame()
            tree = recursive_game_tree(pbs, game)
            depth_limit = 1
            start_node = ('root')
            tree.build_depth_limited_subgame(depth_limit, node_id=start_node)
            #add assertions

        with self.subTest('depth limit larger than full game tree'):
            pass
        with self.subTest('depth 1 from middle node'):
            pass
        with self.subTest('depth 1 from terminal node'):
            pass

    def test_set_leaf_values(self):
        pass

    def test_compute_ev(self):
        pass

    def test_sample_leaf(self):
        pass

    def test_sample_child(self):
        pass

    def test_expand_node(self):
        pass

    def test_add_child(self):
        pass

    def test_transition(self):
        pass

    def test_len(self):
        pass

if __name__ == "__main__":

    unittest.main()
