# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

import math


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostpositions = successorGameState.getGhostPositions()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] ###

        "*** YOUR CODE HERE ***"

        M_distances_2_pellets = [(abs(newPos[0] - i[0]) + abs(newPos[1] - i[1])) for i in newFood]
        M_distances_2_ghosts = [(abs(newPos[0] - i[0]) + abs(newPos[1] - i[1])) for i in newGhostpositions]

        if M_distances_2_pellets == []:
            a = 0
        else:
            a = min(M_distances_2_pellets)

        eval = -1 * a + min(M_distances_2_ghosts) + successorGameState.getScore()
        if action == 'Stop':
            eval -= 1
        if successorGameState.isLose():
            eval -= 2
        return eval


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        "*** YOUR CODE HERE ***"

        def minimax_recursion(gameState, agentIndex, depth):

            v = float(-math.inf)
            if agentIndex > 0:
                v *= -1

            # generate legal actions from agent
            LEGAL_MOVES = gameState.getLegalActions(agentIndex)
            # check if the state is a terminal state
            if gameState.isWin() == True or gameState.isLose() == True or depth == self.depth:
                return self.evaluationFunction(gameState)

            for action in LEGAL_MOVES:

                # look into new game states after the action for the agent is done

                new_gameState = gameState.generateSuccessor(agentIndex, action)

                # if the next agent is the pacman, then we want to increment the depth by one, anytime all agents make at least one move that's one ply

                if agentIndex + 1 == gameState.getNumAgents():
                    successor_state_cost = minimax_recursion(new_gameState, 0, depth + 1)
                else:
                    successor_state_cost = minimax_recursion(new_gameState, agentIndex + 1, depth)

                # sometimes the successor_state returns a value or a tuple, if its a tuple were only interested in the first value

                if isinstance(successor_state_cost, tuple):
                    # if it is the tuple (v, move) were only interested in v in our recursion!
                    successor_state_cost = successor_state_cost[0]

                # if were looking at pacman, we want to maximize the v with the corresponding action
                if agentIndex == 0 and successor_state_cost > v:
                    v = successor_state_cost
                    move = action

                # if were looking at the ghost, we want to minimize the v as much as possible
                elif agentIndex != 0 and successor_state_cost < v:
                    v = successor_state_cost
                    move = action

            # i return a tuple because i need the final move to be fed into getAction, and I need the utilities to be fed back into the recursion for comparison of v
            return (v, move)

        return (minimax_recursion(gameState, 0, 0))[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        # self.alpha = -float(math.inf)
        # self.beta = float(math.inf)
        def minimax_recursion_ab(gameState, agentIndex, depth, alpha = -float(math.inf), beta = float(math.inf)):

            v = float(-math.inf)
            if agentIndex > 0:
                v *= -1

            LEGAL_MOVES = gameState.getLegalActions(agentIndex)

            if gameState.isWin() == True or gameState.isLose() == True or depth == self.depth:
                return self.evaluationFunction(gameState)

            for action in LEGAL_MOVES:

                new_gameState = gameState.generateSuccessor(agentIndex, action)

                if agentIndex + 1 == gameState.getNumAgents():
                    successor_state_cost = minimax_recursion_ab(new_gameState, 0, depth + 1, alpha, beta)
                else:
                    successor_state_cost = minimax_recursion_ab(new_gameState, agentIndex + 1, depth, alpha, beta)

                if isinstance(successor_state_cost, tuple):
                    successor_state_cost = successor_state_cost[0]

                # same thing as before except with the comparison to alpha and beta inequalities

                if agentIndex == 0 and successor_state_cost > v:
                    v = successor_state_cost
                    move = action
                    if v > beta:  # not pruning on equality
                        return (v, move)
                    else:
                        alpha = max(alpha, v)

                elif agentIndex != 0 and successor_state_cost < v:
                    v = successor_state_cost
                    move = action
                    if v < alpha:  # not pruning on equality
                        return (v, move)
                    else:
                        beta = min(beta, v)
            return (v, move)

        return (minimax_recursion_ab(gameState, 0, 0))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax_recursion(gameState, agentIndex, depth):

            v = float(-math.inf)
            if agentIndex > 0:
                v = 0

            LEGAL_MOVES = gameState.getLegalActions(agentIndex)

            if gameState.isWin() == True or gameState.isLose() == True or depth == self.depth:
                return self.evaluationFunction(gameState)

            for action in LEGAL_MOVES:

                new_gameState = gameState.generateSuccessor(agentIndex, action)

                if agentIndex + 1 == gameState.getNumAgents():
                    successor_state_cost = expectimax_recursion(new_gameState, 0, depth + 1)
                else:
                    successor_state_cost = expectimax_recursion(new_gameState, agentIndex + 1, depth)

                if isinstance(successor_state_cost, tuple):
                    successor_state_cost = successor_state_cost[0]

                if agentIndex == 0 and successor_state_cost > v:
                    move = action
                    v = successor_state_cost
                # the difference here is that were not making any comparison in the min-value portion of the code
                elif agentIndex != 0:
                    move = action
                    p = (1/len(LEGAL_MOVES))
                    v += p * successor_state_cost
            return (v, move)

        return (expectimax_recursion(gameState, 0, 0))[1]


def betterEvaluationFunction(currentGameState):




    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newGhostpositions = currentGameState.getGhostPositions()
    M_distances_2_pellets = [(abs(newPos[0] - i[0]) + abs(newPos[1] - i[1])) for i in currentGameState.getFood().asList()]
    M_distances_2_ghosts = [(abs(newPos[0] - i[0]) + abs(newPos[1] - i[1])) for i in newGhostpositions]
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    if M_distances_2_pellets == []:
        a = 0
    else:
        a = min(M_distances_2_pellets)

    eval = -1 * a + min(M_distances_2_ghosts) + currentGameState.getScore() + sum(newScaredTimes)

    if currentGameState.isLose():
        eval -= 2
    return eval


# Abbreviation
better = betterEvaluationFunction
