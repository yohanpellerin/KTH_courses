#!/usr/bin/env python3
# import random

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
import time

history = {}


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
            history={}
            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    # def heuristic(self, node, depth):
    #     hook_position_0 = node.state.get_hook_positions()[0]
    #     hook_position_1 = node.state.get_hook_positions()[1]
    #     #node.compute_and_get_children()
    #     fish_positions = node.state.get_fish_positions()
    #     fish_scores = node.state.get_fish_scores()
    #     score = node.state.player_scores[0] - node.state.player_scores[1]
    #     value = 0
    #     sum_score = 0
    #
    #     # print(hook_position,fish_positions,node.move)
    #     for fish_indice, fish_position in fish_positions.items():
    #         sum_score += fish_scores[fish_indice]
    #         nb_move = self.nb_move(fish_position, hook_position_0)
    #         nb_move_1=self.nb_move(fish_position, hook_position_1)
    #         if nb_move == 0:
    #             value += fish_scores[fish_indice]
    #         else:
    #             value += fish_scores[fish_indice] / (1.5 * nb_move)
    #         if nb_move_1 == 0:
    #             value -= fish_scores[fish_indice]*2
    #         else:
    #             value -= fish_scores[fish_indice] / (1.5 * nb_move_1)
    #     return value + score + depth * 0.00001 #+len(node.children)
    #     # c'est mieux si au début y a moins de poisson soit attraper les poissons le plus vite
    def heuristic(self, node, depth):
        hook_positions = node.state.get_hook_positions()
        hook_position_0 = hook_positions[0]
        hook_position_1 = hook_positions[1]

        fish_positions = node.state.get_fish_positions()
        fish_scores = node.state.get_fish_scores()
        player_scores = node.state.player_scores

        score = player_scores[0] - player_scores[1]
        value = 0
        sum_score = 0
        # fishes_caught = node.state.get_caught()
        # if not fishes_caught[0]==None:
        #     score += fish_scores[fishes_caught[0]]
        #
        # if not fishes_caught[1]==None:
        #     score -= fish_scores[fishes_caught[1]]
        #
        # return score + depth*0.00001


        for fish_indice, fish_position in fish_positions.items():
             sum_score += fish_scores[fish_indice]
             nb_move = self.nb_move(fish_position, hook_position_0)
             nb_move_1 = self.nb_move(fish_position, hook_position_1)

             if nb_move == 0:
                 value += fish_scores[fish_indice]
             else:
                 value += fish_scores[fish_indice] / (1.5 * nb_move)

             if nb_move_1 == 0:
                 value -= fish_scores[fish_indice] * 2
             else:
                 value -= fish_scores[fish_indice] / (1.5 * nb_move_1)

        return value + score + depth * 0.00001

    def nb_move(self, fish_position, hook_position):
        return abs(fish_position[0] - hook_position[0]) + abs(fish_position[1] - hook_position[1])

    def minimax(self, position, depth, maximizing_player, alpha, beta, timer):
        if time.time() - timer >= 0.067:
            return -1000
        if depth == 0 or len(position.state.get_fish_positions()) == 0:  # or self.check_game_over(position): #
            return self.heuristic(position, depth)
        position.compute_and_get_children()
        ordered_moves = self.order_moves(position.children)
        if maximizing_player:
            best_score = float('-inf')
            for children in ordered_moves:
                score = self.minimax(children, depth - 1, False, alpha, beta, timer)
                best_score = max(best_score, score)
                alpha = max(alpha, score)
                if score == -1000:
                    best_score = -1000
                    break
                if beta <= alpha:
                    break
                self.update_history(children, score)
            return best_score
        else:
            best_score = float('inf')
            for children in ordered_moves:
                score = self.minimax(children, depth - 1, True, alpha, beta, timer)
                best_score = min(best_score, score)
                beta = min(beta, score)
                if score == -1000:
                    best_score = -1000
                    break
                if beta <= alpha:
                    break
                self.update_history(children, score)
            return best_score

    def choose_best_move(self, position, timer):

        alpha = float('-inf')
        beta = float('inf')

        # Initialise max depth
        max_depth = 12
        value_list = []
        move_list = []

        position.compute_and_get_children()

        for depth in range(2, max_depth + 1):
            #print("Depth :", depth)
            best_score = float('-inf')
            best_move = None
            ordered_moves = self.order_moves_by_history(position.children) if depth != 2 else position.children
            #print(len(ordered_moves))
            for child in ordered_moves:
                if time.time() - timer >= 0.064-(depth-2)*depth*0.001:
                    best_score = -1000
                    break
                score = self.minimax(child, depth=depth, maximizing_player=False, alpha=alpha, beta=beta, timer=timer)
                if score == -1000:
                    best_score = -1000
                    break
                if score > best_score:
                    best_score = score
                    best_move = child.move
                alpha = max(alpha, score) # pas de prunning possible
                self.update_history(child, score)  # Mettre à jour l'historique avec le score évalué
            value_list += [best_score]
            move_list += [best_move]
            if time.time() - timer >= 0.064-(depth-2)*depth*0.001:
                break

        return (move_list[-1],len(move_list)+1) if value_list[-1] != -1000 else (move_list[-2], len(move_list))
    history = {}
    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find the best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # Initialise time
        timer = time.time()
        #print('test')
        best_move, depth = self.choose_best_move(initial_tree_node, timer)
        #print("best_move", best_move, depth)
        #print(history)
        #history = {}
        return ACTION_TO_STR[best_move]

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        # random_move = random.randrange(5)

    def order_moves(self, children):
        capture_moves = []
        quiet_moves = []

        for move in children:
        # Vous devrez adapter cette logique à votre représentation de plateau
            if self.is_capture(move):  # Fonction pour déterminer si c'est une capture
                capture_moves.append(move)
            else:
                quiet_moves.append(move)
        return capture_moves + quiet_moves

    def is_capture(self, move):
        #print(move.state.get_caught())
        return move.state.get_caught()[0] != None
    def update_history(self, move, score):
        history[move] = score

# Fonction pour trier les mouvements en utilisant l'heuristique d'historique (score)
    def order_moves_by_history(self, moves):
        return sorted(moves, key=lambda move: history.get(move, -1000), reverse=True)