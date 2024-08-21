# Artificial Intelligence

## This branch is meant to solve the exercise called "Search"

Due date 20 Novemeber


# Questions

## Question 1

- Possible States (S): The possible states of the KTH Fishingderby game refer to all the different configurations the game can be in. In this case, it includes the positions of the two fishing boats, the positions of the fish, the state of the fishing lines, and the remaining time on the clock.
- Initial State (s0): The initial state of the game is the starting configuration before any player makes a move. It could be defined by the initial positions of the boats, the initial positions of the fish, and the initial time remaining on the clock.
- Transition Function (µ): The transition function defines the legal moves that a player can make from a given state. For the KTH Fishingderby game, the transition function, denoted as µ, takes a player and a game state and returns the possible game states that the player may achieve with one legal move. This includes boat movements (LEFT, RIGHT, UP, DOWN, STAY) and changes in the position of the fishing lines.

## Question 2

The terminal states of the KTH Fishingderby game are the states where the game has reached a conclusion. In this context, a terminal state is reached when either of the following conditions is met:
- No fish left: If there are no fish left on the screen, the game is over, and this is a terminal state.
- Game time has passed: When the specified time for the fishing day has elapsed (as indicated by the clock), the game is over, and this is also a terminal state.
When the game reaches a terminal state, the winner is determined based on the utility function (γ), which evaluates the "usefulness" of the state for each player. The player with the higher utility is declared the winner, and the game is either won by Player A, won by Player B, or declared a tie.

## Question 3

The heuristic function ν(A, s) = Score(Green boat) - Score(Red boat) is a good heuristic for the KTH fishing derby because it captures the relative advantage of the Green boat (controlled by player A) over the Red boat (controlled by player B) in terms of the cumulative score. The goal of the game is to maximize the total score, and this heuristic provides a simple and direct measure of how well the Green boat is performing compared to the Red boat. It guides the agent to make decisions that lead to a state where the Green boat has a higher score than the Red boat.

## Question 4

ν best approximates the utility function when the heuristic accurately reflects the true value of the utility. This approximation is most accurate when the heuristic takes into account all relevant factors that contribute to the utility of a state. In the case of KTH fishing derby, if the heuristic considers the types of fish caught, their respective scores, penalties for catching undesirable fish, and any other relevant factors influencing the overall utility, it would better approximate the utility function.

## Question 5

Let's consider an example state:
- Green boat has caught fish of types A, B, and C, with scores 6, 4, and -3 respectively.
- Red boat has caught fish of types X, Y, and Z, with scores 5, 3, and -2 respectively.
Now, ν(A, s) = (6 + 4 - 3) - (5 + 3 - 2) = 1.

In the next turn, if B (Red boat) manages to catch a high-scoring fish, let's say another fish of type X with a score of 5, then ν(A, s') = (6 + 4 - 3) - (5 + 3 - 2 + 5) = -4, and B wins in the following turn if it is the final one.

## Question 6

Yes, η may suffer from similar problems as the evaluation function ν. The heuristic η relies on counting the number of winning and losing states accessible from a given state, but it doesn't necessarily capture the subtleties of the game dynamics or the impact of specific moves on the future states.

Example:
Consider a scenario in the KTH fishing derby game:
- A fish with a high score is present 11 and   other fish with the value of 2
- The fish with the value of 11 is really deep in the water while all the others are close to the surface

If the green boat can capture the fish with the score 11, he will have much more chance to win the game. He will propably get there because it is the fish with the highest score and he might be able to catch the other fishes. But if the red boat only focus on the other fishes, he will propably capture them before the green boat. And the red boat would finally win. 

This heuristic has also another problem wich is the fact that it can estimate the value of a state only if it is a terminal state. Which means that the algorithm has to go really deep from the begining of the game.


In games like chess, similar heuristics can be problematic. For example, in a simplified scenario, counting the number of pieces on the board might suggest an advantage, but it doesn't account for the strategic value of each piece or the potential future moves. A chess position might look favorable based on ´, but a well-calculated move by the opponent can completely change the outcome in subsequent turns.
