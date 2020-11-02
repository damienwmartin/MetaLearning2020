import pyspiel



class TreeNode:
    
    def __init__(self, state, children_begin, children_end, parent, depth):
        self.state = state
        self.children_begin = children_begin
        self.children_end = children_end
        self.parent = parent
        self.depth = depth

    def num_children():
        return self.children_end - self.children_begin
    
    def get_children():
        return [children_begin + i for i in range(self.num_children())]
    


def unroll_tree(game_name, root=None, max_depth=None):
    """
    Builds a subtree of depth max_depth using BFS, where the root itself is at depth 0
    """


    game = pyspiel.load_game(game_name)

    if root is None:
        root = game.new_initial_state()
    
    if max_depth is None:
        max_depth = 1000000000 # How to get max depth of a game?

    nodes = [TreeNode(root, 0, 0, -1, 0)]
    while node_id < len(nodes) and nodes[node_id].depth < max_depth
        parent_node = nodes[node_id]
        parent_node.children_begin = len(nodes)
        parent_node.children_end = parent_node.children_begin + len(parent_node.legal_actions()) - 1
        for act in parent_node.legal_actions():
            new_state = root
            new_state.apply_action(act)
            nodes.append(TreeNode(new_state, 0, 0, node_id, parent.depth+1))
        node_id += 1