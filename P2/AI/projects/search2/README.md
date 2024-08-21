# Artificial Intelligence

## This branch is meant to solve the exercise called "Search"

Due date 20 Novemeber


# Questions

## Question 1

- Possible States (S): The possible states of the KTH Fishingderby game refer to all the different configurations the game can be in. In this case, it includes the positions of the two fishing boats, the positions of the fish, the state of the fishing lines, and the remaining time on the clock.
- Initial State (s0): The initial state of the game is the starting configuration before any player makes a move. It could be defined by the initial positions of the boats, the position of the fishing lines, the initial positions of the fish, and the initial time remaining on the clock.
- Transition Function (µ): The transition function defines the legal moves that a player can make from a given state. For the KTH Fishingderby game, the transition function, denoted as µ, takes a player and a game state and returns the possible game states that the player may achieve with one legal move. This includes boat movements and changes in the position of the fishing lines (LEFT, RIGHT, UP, DOWN, STAY).

## Question 2

The terminal states of the KTH Fishingderby game are the states where the game has reached a conclusion. In this context, a terminal state is reached when either of the following conditions is met:
- No fish left: If there are no fish left on the screen, the game is over, and this is a terminal state.
- Game time has passed: When the specified time for the fishing day has elapsed (as indicated by the clock), the game is over, and this is also a terminal state.
When the game reaches a terminal state, the winner is determined based on the utility function (γ), which evaluates the "usefulness" of the state for each player. The player with the higher utility is declared the winner, and the game is either won by Player A, won by Player B, or declared a tie.

## Question 3

The heuristic function ν(A, s) = Score(Green boat) - Score(Red boat) is a good heuristic for the KTH fishing derby because it captures the relative advantage of the Green boat (controlled by player A) over the Red boat (controlled by player B) in terms of the cumulative score. The goal of the game is to maximize the total score, and this heuristic provides a simple and direct measure of how well the Green boat is performing compared to the Red boat. It guides the agent to make decisions that lead to a state where the Green boat has a higher score than the Red boat.

## Question 4

ν best approximates the utility function when the heuristic accurately reflects the true value of the utility. This approximation is most accurate when the heuristic takes into account all relevant factors that contribute to the utility of a state. In the case of KTH fishing derby, if the heuristic considers the types of fish caught, their respective scores, penalties for catching undesirable fish, and any other relevant factors influencing the overall utility, it would better approximate the utility function.
In the case of this specific ν, it best approximate the score when it reaches a final state. When it is no a final state, there is a possible problem which is that a boat (and possibly the red boat) has captured a new fish. It will best estimates the result in a terminal state because the score will not change anymore.

## Question 5

Let's consider an example state:
- Green boat has caught fish of types A, B, and C, with scores 6, 4, and -3 respectively.
- Red boat has caught fish of types X, Y, and Z, with scores 5, 3, and -2 respectively.
Now, ν(A, s) = (6 + 4 - 3) - (5 + 3 - 2) = 1.

In the next turn, if B (Red boat) manages to catch a high-scoring fish, let's say another fish of type X with a score of 5, then ν(A, s') = (6 + 4 - 3) - (5 + 3 - 2 + 5) = -4, and B wins in the following turn if it is the final one.

## Question 6

Yes, η may suffer from similar problems as the evaluation function ν. If we consider the scenario, where one move leads to have two winning final states, and an other move which leads to 6 winning final states but also has two loosing final states. This heurestic would lead to the second choice even if you may have created a scenarion were you loose if you play against a good player. To sum up, this heurestic doesn't take in account the move of your opponent. That could kill all the possibility of winning. 
