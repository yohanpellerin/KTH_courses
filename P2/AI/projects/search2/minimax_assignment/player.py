#!/usr/bin/env python3
import random
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        
        self.timer = time.time()
        self.moves = {0:"stay", 1:"stay", 2:"stay", 3:"stay", 4:"stay", 5:"stay", 6:"stay", 7:"stay", 8:"stay", 9:"stay"}


    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})


    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!
        
        # Initialise time
        self.timer = time.time()
        
        # Initialise max depth
        max_depth = 9
        value_list = []
        move_list = []
        
        self.moves = {0:"stay", 1:"stay", 2:"stay", 3:"stay", 4:"stay", 5:"stay", 6:"stay", 7:"stay", 8:"stay", 9:"stay"}
        for depth in range(2, max_depth + 1):
            # Call the Minimax function with Alpha-Bêta
            try:    
                value = self.my_minimax(initial_tree_node, depth)
                value_list.append(value)
                move_list.append(self.moves[0])
            except Exception as e:
                break
        return move_list[-1]


    def my_minimax(self, node, depth):
        # Check if time is over
        if time.time() - self.timer >= 0.055:
            raise Exception("Time is over")
        
        node.compute_and_get_children()
        
        # Initialise alpha and beta 
        alpha = -float('inf')
        beta = float('inf')
        
        children_sorted = []
        for child in node.children:
            if ACTION_TO_STR[child.move] == self.moves[0]:
                children_sorted.insert(0,child)
            else:
                children_sorted.append(child)
        
        best_score = -float('inf')
        for child in children_sorted:
            value = self.minimax(child, depth - 1, alpha, beta, False, 1)
            if value > best_score:
                best_score = value
                self.moves[0] = ACTION_TO_STR[child.move]
            alpha = max(alpha, value)
        return alpha
        
        
    def minimax(self, node, depth, alpha, beta, maximizing_player, reel_depth):
        if time.time() - self.timer >= 0.055:
            raise Exception("Time is over")
        
        node.compute_and_get_children()

        if (depth == 0  or len(node.state.get_fish_positions()) == 0 or 
            ((node.state.get_caught()[0] is not None or node.state.get_caught()[1] is not None) and len(node.state.get_fish_positions()) == 1) or 
            (node.state.get_caught()[0] is not None and node.state.get_caught()[1] is not None and len(node.state.get_fish_positions()) == 2)) :
            return self.heuristic(node, depth)
        
        children_sorted = []
        for child in node.children:
            if ACTION_TO_STR[child.move] == self.moves[reel_depth]:
                children_sorted.insert(0,child)
            else:
                children_sorted.append(child)
        
        if maximizing_player:
            max_eval = -float('inf')
            for child in children_sorted:
                eval = self.minimax(child, depth - 1, alpha, beta, False, reel_depth+1)
                if eval > max_eval:
                    max_eval = eval
                    self.moves[reel_depth] = ACTION_TO_STR[child.move]
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for child in children_sorted:
                eval = self.minimax(child, depth - 1, alpha, beta, True, reel_depth+1)
                if eval < min_eval:
                    min_eval = eval
                    self.moves[reel_depth] = ACTION_TO_STR[child.move]
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval


    def heuristic(self, node, depth):
        hook_positions = node.state.get_hook_positions()
        hook_position_0 = hook_positions[0]
        hook_position_1 = hook_positions[1]
        caught_0, caught_1 = node.state.get_caught()

        fish_positions = node.state.get_fish_positions()
        fish_scores = node.state.get_fish_scores()
        player_scores = node.state.player_scores

        score = player_scores[0] - player_scores[1]
        value = 0
        
        for fish_indice, fish_position in fish_positions.items():
            nb_move = self.nb_move(fish_position, hook_position_0, hook_position_1, fish_indice, caught_1)
            nb_move_1 = self.nb_move(fish_position, hook_position_1, hook_position_0, fish_indice, caught_0)

            if nb_move == 0:
                value += fish_scores[fish_indice]                 

            elif nb_move_1 == 0:
                value -= fish_scores[fish_indice]
                
            else:
                value -= fish_scores[fish_indice] / (1.5 * nb_move_1)
                value += fish_scores[fish_indice] / (1.5 * nb_move)

        return value + score + depth * 0.00001

    def nb_move(self, fish_pos, pos_1, pos_2, fish_id, caught_fish):
        return min(abs(fish_pos[0] - pos_1[0]), 20 - abs(fish_pos[0] - pos_1[0])) + abs(fish_pos[1] - pos_1[1])