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
        
        player_hook_pos = initial_tree_node.state.get_hook_positions()[0]
        print("Player hook pos :", player_hook_pos)
        
        # Initialise time
        timer = time.time()
        
        # Initialise Alpha and Bêta
        alpha = -float('inf')
        beta = float('inf')

        # Initialise max depth
        max_depth = 12
        value_list = []
        move_list = []
        
        order = ["stay", "up", "left", "right", "down"]
        for depth in range(2, max_depth + 1):
            print("\n\n\n")
            print("Depth :", depth)
            print("\n\n\n")
            # Call the Minimax function with Alpha-Bêta
            value, the_move = self.my_minimax(initial_tree_node, depth, timer, order)
            value_list.append(value)
            move_list.append(the_move)
            order = [the_move] + [x for x in order if x != the_move]
            # Check if time is over
            if time.time() - timer >= 0.065:
                break
        print("Max depth :", depth)
        print("Value list :", value_list)
        print("Move list :", move_list)
        print("Best Move :", move_list[-1] if value_list[-1] != -10000000000 else move_list[-2])
        return move_list[-1] if value_list[-1] != -10000000000 else move_list[-2]

    def my_minimax(self, node, depth, timer, order):
        # Check if time is over
        if time.time() - timer >= 0.065:
            return -10000000000, "stay"
        
        node.compute_and_get_children()
         
        # Check if there is only one child (no choice because we have caught a fish)
        if len(node.children)==1:
            return 0, "up" 
        
        alpha = -float('inf')
        beta = float('inf')
        
        children_sorted = []
        for move in order:
            for child in node.children:
                if self.get_move(node, child) == move:
                    children_sorted.append(child)
                    break
                
        for child in children_sorted:
            value = self.minimax(child, depth - 1, alpha, beta, False, timer)
            if value > alpha:
                alpha = value
                best_move = self.get_move(node, child)
        return alpha, best_move
        
    def minimax(self, node, depth, alpha, beta, maximizing_player, timer):
        node.compute_and_get_children()
        
        if depth == 0:
            return self.evaluate(node.state)
        if time.time() - timer >= 0.065:
            return -10000000000
        
        if maximizing_player:
            max_eval = -float('inf')
            for child in node.children:
                eval = self.minimax(child, depth - 1, alpha, beta, False, timer)
                if eval > max_eval:
                    max_eval = eval
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for child in node.children:
                eval = self.minimax(child, depth - 1, alpha, beta, True, timer)
                if eval < min_eval:
                    min_eval = eval
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval


    def get_move(self, node, child):
        if node.state.get_hook_positions()[0] == child.state.get_hook_positions()[0]:
            return "stay" 
        elif node.state.get_hook_positions()[0][0] > child.state.get_hook_positions()[0][0]:
            return "left"
        elif node.state.get_hook_positions()[0][0] < child.state.get_hook_positions()[0][0]:
            return "right"
        elif node.state.get_hook_positions()[0][1] < child.state.get_hook_positions()[0][1]:
            return "up"
        elif node.state.get_hook_positions()[0][1] > child.state.get_hook_positions()[0][1]:
            return "down"


    def evaluate(self, state):
        player_score, opponent_score = state.get_player_scores()
        print("Player score :", player_score)
        player_caught, opponent_caught = state.get_caught()
        fish_scores = state.get_fish_scores()
        player_hook_pos = state.get_hook_positions()[0]
        opponent_hook_pos = state.get_hook_positions()[1]
        fish_positions = state.get_fish_positions()

        fish_score_difference = (player_score - opponent_score)
        
        if player_caught is None:
            caught_by_player = 0
        else:
            caught_by_player = self.count_caught_fish(fish_scores, player_caught, player_hook_pos)
        if opponent_caught is None:
            caught_by_opponent = 0
        else:
            caught_by_opponent = self.count_caught_fish(fish_scores, opponent_caught, opponent_hook_pos)
            
        caught_score_difference = caught_by_player - caught_by_opponent
        
        hook_difference = self.get_hook_difference(player_hook_pos, opponent_hook_pos, fish_positions, fish_scores, player_caught, opponent_caught) 
            
        total_evaluation = (
            30 * fish_score_difference +
            5 * caught_score_difference +
            hook_difference
        )
        print("Total evaluation :", total_evaluation)
        return total_evaluation
        
    def get_hook_difference(self, player_hook_pos, opponent_hook_pos, fish_positions, fish_scores, player_caught, opponent_caught):
        player_evaluation = 0
        opponent_evaluation = 0
        for fish_id, fish_pos in fish_positions.items():
            fish_score = fish_scores[fish_id]
            if fish_id != opponent_caught:
                player_evaluation += fish_score / self.get_distance(player_hook_pos, opponent_hook_pos, fish_pos) if self.get_distance(player_hook_pos, opponent_hook_pos, fish_pos) != 0 else fish_score*1.5
            elif fish_id != player_caught:
                opponent_evaluation += fish_score / self.get_distance(opponent_hook_pos, player_hook_pos, fish_pos) if self.get_distance(opponent_hook_pos, player_hook_pos, fish_pos) != 0 else fish_score*1.5
        return player_evaluation - opponent_evaluation
    
    def get_distance(self, pos_1, pos_2, fish_pos):
        if pos_1[0] < pos_2[0] and pos_2[0] < fish_pos[0]:
            return ((20-(pos_1[0] - fish_pos[0])) ** 2 + (pos_1[1] - fish_pos[1]) ** 2) ** 0.5
        elif pos_1[0] > pos_2[0] and pos_2[0] > fish_pos[0]:
            return ((20-(pos_1[0] - fish_pos[0])) ** 2 + (pos_1[1] - fish_pos[1]) ** 2) ** 0.5 
        return ((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2) ** 0.5
    
    def count_caught_fish(self, fish_scores, caught_fish, hook_pos):
        caught_fish = 0
        for fish_id, fish_score in fish_scores.items():
            if fish_id == caught_fish:
                caught_fish += fish_score / (20-hook_pos[1])
        return caught_fish
    



# mkvirtualenv -p C:\Users\Antoine\AppData\Local\Programs\Python\Python311\python.exe fishingderby


# C:\Users\Antoine\Envs\fishingderby\Scripts\activate.bat

# py -3.7 -m pip install numpy


# py -3.7 main.py settings.yml



    # def minimax(self, node, depth, alpha, beta):
    #     max_val = self.max_value(node, depth, alpha, beta)
    #     return max_val
    
    # def max_value(self, node, depth, alpha, beta):
    #     node.compute_and_get_children()
    #     if depth == 0:
    #         print("max")
    #         return self.evaluate(node.state)
        
    #     v = -float('inf')
    #     for child in node.children:
    #         v = max(v, self.min_value(child, depth - 1, alpha, beta))
    #         if v >= beta:
    #             return v
    #         alpha = max(alpha, v)
    #     return v
    
    # def min_value(self, node, depth, alpha, beta):
    #     node.compute_and_get_children()
    #     if depth == 0:
    #         print("min")
    #         return self.evaluate(node.state)
        
    #     v = float('inf')
    #     for child in node.children:
    #         v = min(v, self.max_value(child, depth - 1, alpha, beta))
    #         if v <= alpha:
    #             return v
    #         beta = min(beta, v)
    #     return v