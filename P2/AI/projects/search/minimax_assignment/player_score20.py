#!/usr/bin/env python3
import time
import hashlib
# import random

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
        self.max_depth = 7
        self.transposition_table = {}
        self.start_time = None
        self.time_limit = 0.065


    def player_loop(self):
        first_msg = self.receiver()
       
        while True:
            msg = self.receiver()
            if msg["game_over"]:
                return

            node = Node(message=msg, player=0)
            self.start_time = time.time() 

            best_move = 0
            current_move = None
            for depth in range(1, self.max_depth + 1):
                try:
                    if time.time() - self.start_time < self.time_limit :
                        # print("depth",depth)
                        current_move = self.minimax(node, depth, True)
                        if current_move in ACTION_TO_STR : 
                            best_move = current_move
                    else:
                        break
                except TimeoutError:
                    break               

            # if best_move not in ACTION_TO_STR :
            #     best_move = random.randrange(5)        
            self.sender({"action": ACTION_TO_STR[best_move], "search_time": None})

    def get_state_key(self, state,depth):
        # Convert the state to a unique string that represents the game state
        # This could include the positions and states of all entities in the game
        # For example, we might use the positions of hooks and fish, and the current score
        # The following is a pseudo-code and should be adapted to your state's structure
        
        # Fetching player scores
        scores = state.get_player_scores()
        score_key = ','.join(map(str, scores))
        
        # Fetching positions of hooks and fish
        hook_positions = state.hook_positions
        fish_positions = state.get_fish_positions()

        # sorted_fish_positions = sorted(fish_positions.items(), key=lambda x: x[0])
        fish_positions_key = ','.join(f'{fish}:{pos}' for fish, pos in fish_positions.items())
        
        # Combining all the components to create a unique key
        state_key = f'{score_key}|{hook_positions[0]}|{hook_positions[1]}|{fish_positions_key}|{depth}'
        
        return hashlib.md5(state_key.encode()).hexdigest()

    def minimax(self, node, depth, maximizing_player, alpha=float('-inf'), beta=float('inf')):
        if time.time() - self.start_time >= self.time_limit:
            raise TimeoutError
                        
        if depth == 0 or self.is_terminal_node(node.state) :
            return self.quiescence_search(node, depth, maximizing_player, alpha, beta) 
            # return self.heuristic(node.state, depth)
        
        state_key = self.get_state_key(node.state,depth)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]
       
        children = node.compute_and_get_children()
        children.sort(key=lambda child: self.heuristic(child.state,depth), reverse=maximizing_player)

        if maximizing_player:
            max_eval = float('-inf')
            children.sort(key=lambda child: self.heuristic(child.state,depth), reverse=maximizing_player)

            best_move = None
            for child in children:

                eval = self.minimax(child, depth - 1, False, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = child.move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            if depth == self.max_depth:
                self.transposition_table[state_key] = best_move
                return best_move
            self.transposition_table[state_key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            children.sort(key=lambda child: self.heuristic(child.state,depth), reverse=maximizing_player)
            for child in children:
            
                eval = self.minimax(child, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.transposition_table[state_key] = min_eval
            return min_eval
    


    def quiescence_search(self, node, depth, maximizing_player, alpha, beta):       
        stand_pat = self.heuristic(node.state,depth)
        if stand_pat >= beta:
            return beta
       
        if alpha < stand_pat:
            alpha = stand_pat
            
        for child in self.generate_critical_moves(node):
            if time.time() - self.start_time >= self.time_limit:
                raise TimeoutError
            if maximizing_player:
                score = self.quiescence_search(child, depth, maximizing_player, alpha, beta, False)
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            else:
                score = self.quiescence_search(child, depth, maximizing_player, alpha, beta, True)
                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score

        return alpha if maximizing_player else beta
    
    def generate_critical_moves(self, node):
        all_moves = node.compute_and_get_children()  
        critical_moves = []

        for move in all_moves:
            if self.is_move_critical(move, node.state):
                critical_moves.append(move)
        return critical_moves

    def is_move_critical(self, move, state):
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        hook_positions = state.hook_positions

        for fish, pos in fish_positions.items():
            if fish_scores[fish] > 5:  
                if self.results_in_catch(move, pos, hook_positions):
                    return True

        return False

    def results_in_catch(self, move, fish_pos, hook_pos):
        new_hook_pos = self.calculate_new_hook_position(move, hook_pos)

        return new_hook_pos == fish_pos
    
    def calculate_new_hook_position(self, move, hook_pos):
        new_x, new_y = hook_pos
        if move == 'LEFT':
            new_x -= 1
        elif move == 'RIGHT':
            new_x += 1
        elif move == 'UP':
            new_y -= 1
        elif move == 'DOWN':
            new_y += 1

        new_x %= 20  
        new_y %= 20

        return new_x, new_y
      
      
    def adjusted_distance(self,hook_pos, fish_pos):
        dx, dy = min(abs(hook_pos[0] - fish_pos[0]), 20 - abs(hook_pos[0] - fish_pos[0])), \
                min(abs(hook_pos[1] - fish_pos[1]), 20 - abs(hook_pos[1] - fish_pos[1]))
        return max(dx, dy) if max(dx, dy) != 0 else 0.0001  # Avoid division by zero


    def translate(self, pos,ref_pos):
        return ((pos[0] - ref_pos[0]) % 20, (pos[1] - ref_pos[1]) % 20)
              
    def heuristic(self, state, depth):
     
        green_hook_pos = state.hook_positions[0]
        red_hook_pos = state.hook_positions[1]
        fish_positions = state.get_fish_positions()
        fish_values = state.get_fish_scores()
        
        green_score = state.get_player_scores()[0]
        red_score = state.get_player_scores()[1]
        score_diff = green_score - red_score

        translated_green_hook_pos = self.translate(green_hook_pos,red_hook_pos)
        translated_fish_positions = {fish: self.translate(fish_pos,red_hook_pos) for fish, fish_pos in fish_positions.items()}

        unavailable_fish_green = {fish for fish, fish_pos in fish_positions.items() if fish_pos == red_hook_pos}

        green_potential = sum(fish_values[fish] / self.adjusted_distance(translated_green_hook_pos, translated_fish_pos)
                            for fish, translated_fish_pos in translated_fish_positions.items()
                            if fish not in unavailable_fish_green)  # Exclude unavailable fish
        
        unavailable_fish_red = {fish for fish, fish_pos in fish_positions.items() if fish_pos == green_hook_pos}
        
        red_potential = sum(fish_values[fish] / self.adjusted_distance(self.translate(red_hook_pos,red_hook_pos), translated_fish_pos)
                            for fish, translated_fish_pos in translated_fish_positions.items()
                            if fish not in unavailable_fish_red) 

        high_value_fish_count = sum(1 for fish,value in fish_values.items() if value > 5)

        green_advantage = green_potential * high_value_fish_count

        red_advantage = red_potential * high_value_fish_count

        adjusted_score_diff = (score_diff + green_advantage - red_advantage)
        
        fish_remaining = len(fish_positions)
        
        depth_factor = 1 + (self.max_depth - depth) * 0.1  

        green_potential *= depth_factor
        red_potential *= depth_factor


        if fish_remaining <= 5:
            fish_value_factor = 1.5 + (depth * 0.1)
            adjusted_score_diff *= fish_value_factor

        return adjusted_score_diff
        
    def is_terminal_node(self, state):
        # The game is over when there are no fish left to catch
        return len(state.get_fish_positions()) == 0