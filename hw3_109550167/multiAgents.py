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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (par1-1)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.


        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        def minimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth ==self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0: #maximize(pacman)
                value = float("-inf")
                actions = gameState.getLegalActions(0)
                for action in actions:
                    value = max(value, minimax(1, depth, gameState.getNextState(0, action)))
                return value
            else :  #minimize(ghosts)
                value = float("inf")
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                actions = gameState.getLegalActions(agent)
                for action in actions:
                    value = min(value, minimax(nextAgent, depth, gameState.getNextState(agent, action)))
                return value
        
        best_value = float("-inf")
        final_action = Directions.NORTH
        actions = gameState.getLegalActions(0)  #pacman's action(first step)
        for action in actions:
            value = minimax(1, 0, gameState.getNextState(0, action))
            if value > best_value:
                best_value = value
                final_action = action

        return final_action 
        # End your code

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (part1-2)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        def expectimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth ==self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0: #maximize(pacman)
                value = float("-inf")
                actions = gameState.getLegalActions(0)
                for action in actions:
                    value = max(value, expectimax(1, depth, gameState.getNextState(0, action)))
                return value
            else :  #minimize(ghosts)
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                actions = gameState.getLegalActions(agent)
                value = 0
                for action in actions:
                    value += expectimax(nextAgent, depth, gameState.getNextState(agent, action))
                value /= float(len(gameState.getLegalActions(agent)))
                return value
        best_value = float("-inf")
        final_action = Directions.NORTH
        actions = gameState.getLegalActions(0)  #pacman's action(first step)
        for action in actions:
            value = expectimax(1, 0, gameState.getNextState(0, action))
            if value > best_value:
                best_value = value
                final_action = action

        return final_action 
        # End your code

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (part1-3).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Begin your code
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()
    scared = sum(newScaredTimes)

    food_dist = 50

    for food in newFood:
        food_dist = min(food_dist, manhattanDistance(newPos, food))

    food_dist = 1.0/food_dist


    cap_dist = 50

    for capsule in newCapsules:
        cap_dist = min(cap_dist, manhattanDistance(newPos, capsule))

    cap_dist = 1.0/cap_dist
    

    return currentGameState.getScore() + food_dist + cap_dist + scared
    # End your code

# Abbreviation
"""
If you complete this part, please replace scoreEvaluationFunction with betterEvaluationFunction ! !
"""
better = betterEvaluationFunction # betterEvaluationFunction or scoreEvaluationFunction
