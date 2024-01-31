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
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        """
        1.  want to get close to food (use newFood, newPos)
        2.  want to stay away from ghosts (use newGhostStates, newPos)
        3. want the newScaredTimes to be as big as possible

        
        print("currentState: ", currentGameState)
        print("currentPos: ", currentGameState.getPacmanPosition())
        print("successorState: ", successorGameState)
        print("newPos: ", newPos)
        print("newFood: ", newFood.asList())
        print("newGhostStates: ", newGhostStates)
        print("newScaredTimes: ", newScaredTimes)
        print("---------------------------------------------")
        """

        """
        1.
        foodEval: sum of distance bw foods and agent
        we want it to be as small as possible!
        """

        foodEval = 0

        for food in newFood.asList():
            foodEval += manhattanDistance(food, newPos)

        if len(newFood.asList()) != 0:
            foodEval /= len(newFood.asList())  # get the avg

        """
        2.
        ghostEval: sum of distance bw ghosts and agent
        we want it to be as big as possible!
        """
        ghostEval = 0

        for ghostState in newGhostStates:
            ghostEval += manhattanDistance(ghostState.getPosition(), newPos)

        if len(newGhostStates) != 0:
            ghostEval /= len(newGhostStates)  # avg

        """
        3.
        scaredEval: sum of newScaredTimes
        as big as possible
        """

        scaredEval = 0

        for scaredTime in newScaredTimes:
            scaredEval += scaredTime

        if len(newScaredTimes) != 0:
            scaredEval /= len(newScaredTimes)  # avg

        # once we find our variables, it is all about finding a good weight balance
        return (scaredEval * 3) + (ghostEval * 1.1) - (foodEval * 1.25) + (successorGameState.getScore() * 4)


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        actions = gameState.getLegalActions(0)

        actionValues = [self.getValue(gameState.generateSuccessor(0, action), self.depth, 1) for action in actions]

        maxIndex = actionValues.index(max(actionValues))

        # print("actions: ", actions)
        # print("actionValues: ", actionValues)
        # print("actionChosen: ", actions[maxIndex])

        return actions[maxIndex]

    def getValue(self, gameState: GameState, depth: int, agentIndex: int):

        # print("inside getValue. Agent no:", agentIndex)
        if depth == 0:  # if terminal state
            # print("depth 0, evaluate")
            returnVal = self.evaluationFunction(gameState)
            return returnVal

        if gameState.isWin():
            # print("W baby")
            returnVal = self.evaluationFunction(gameState)
            return returnVal

        if gameState.isLose():
            # print("took that L")
            returnVal = self.evaluationFunction(gameState)
            return returnVal

        if (gameState.getNumAgents() == agentIndex + 1) and (depth > 0):  # last agent move for the current depth
            depth -= 1
        if agentIndex >= 1:  # min agent (ghost)
            # print("gonna go min")
            return self.minValue(gameState, depth, agentIndex)
        if agentIndex == 0:  # max agent (pacman)
            # print("gonna go max")
            return self.maxValue(gameState, depth, agentIndex)

    def minValue(self, gameState: GameState, depth: int, agentIndex: int):

        actions = gameState.getLegalActions(agentIndex)
        minVal = 9999

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            minVal = min(minVal, self.getValue(successor, depth, (agentIndex + 1) % gameState.getNumAgents()))

        # print("minVal for ", agentIndex, " :", minVal)
        return minVal

    def maxValue(self, gameState: GameState, depth: int, agentIndex: int):

        actions = gameState.getLegalActions(agentIndex)
        maxVal = -9999

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            maxVal = max(maxVal, self.getValue(successor, depth, (agentIndex + 1) % gameState.getNumAgents()))

        # print("maxVal for ", agentIndex, " :", maxVal)
        return maxVal


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)

        alpha = -9999
        beta = 9999

        maxVal = alpha

        actionToReturn = actions[0]

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            maxVal = max(maxVal, self.getValue(successor, self.depth, 1, alpha, beta))

            if maxVal > beta:
                return action

            if maxVal > alpha:
                actionToReturn = action
                alpha = maxVal

        # print("actions: ", actions)
        # print("actionValues: ", actionValues)
        # print("actionChosen: ", actions[maxIndex])

        return actionToReturn

    def getValue(self, gameState: GameState, depth: int, agentIndex: int, alpha: int, beta: int):

        # print("inside getValue. Agent no:", agentIndex)
        if depth == 0:  # if terminal state
            # print("depth 0, evaluate")
            returnVal = self.evaluationFunction(gameState)
            return returnVal

        if gameState.isWin():
            # print("W baby")
            returnVal = self.evaluationFunction(gameState)
            return returnVal

        if gameState.isLose():
            # print("took that L")
            returnVal = self.evaluationFunction(gameState)
            return returnVal

        if (gameState.getNumAgents() == agentIndex + 1) and (depth > 0):  # last agent move for the current depth
            depth -= 1
        if agentIndex >= 1:  # min agent (ghost)
            # print("gonna go min")
            return self.minValue(gameState, depth, agentIndex, alpha, beta)
        if agentIndex == 0:  # max agent (pacman)
            # print("gonna go max")
            return self.maxValue(gameState, depth, agentIndex, alpha, beta)

    def minValue(self, gameState: GameState, depth: int, agentIndex: int, alpha: int, beta: int):

        actions = gameState.getLegalActions(agentIndex)

        minVal = 9999

        for action in actions:

            successor = gameState.generateSuccessor(agentIndex, action)
            minVal = min(minVal,
                         self.getValue(successor, depth, (agentIndex + 1) % gameState.getNumAgents(), alpha, beta))

            if minVal < alpha:
                # print("return early!")
                return minVal

            beta = min(beta, minVal)

            # print("alpha beta: [", alpha, ", ", beta, "]")

        # print("minVal for ", agentIndex, " :", minVal)
        return minVal

    def maxValue(self, gameState: GameState, depth: int, agentIndex: int, alpha: int, beta: int):

        actions = gameState.getLegalActions(agentIndex)
        maxVal = -9999

        for action in actions:

            successor = gameState.generateSuccessor(agentIndex, action)
            maxVal = max(maxVal,
                         self.getValue(successor, depth, (agentIndex + 1) % gameState.getNumAgents(), alpha, beta))

            if maxVal > beta:
                return maxVal

            alpha = max(alpha, maxVal)

        # print("maxVal for ", agentIndex, " :", maxVal)
        return maxVal


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)

        actionValues = [self.getValue(gameState.generateSuccessor(0, action), self.depth, 1) for action in actions]

        maxIndex = actionValues.index(max(actionValues))

        return actions[maxIndex]

    def getValue(self, gameState: GameState, depth: int, agentIndex: int):

        # print("inside getValue. Agent no:", agentIndex)
        if depth == 0:  # if terminal state
            # print("depth 0, evaluate")
            returnVal = self.evaluationFunction(gameState)
            return returnVal

        if gameState.isWin():
            # print("W baby")
            returnVal = self.evaluationFunction(gameState)
            return returnVal

        if gameState.isLose():
            # print("took that L")
            returnVal = self.evaluationFunction(gameState)
            return returnVal

        if (gameState.getNumAgents() == agentIndex + 1) and (depth > 0):  # last agent move for the current depth
            depth -= 1

        # if isChance:
        #    return self.expectedValue(gameState, depth, agentIndex, not isChance)
        if agentIndex >= 1:  # min agent (ghost)
            # print("gonna go min")
            return self.expectedValue(gameState, depth, agentIndex)
        if agentIndex == 0:  # max agent (pacman)
            # print("gonna go max")
            return self.maxValue(gameState, depth, agentIndex)

    def maxValue(self, gameState: GameState, depth: int, agentIndex: int):

        actions = gameState.getLegalActions(agentIndex)
        maxVal = -9999

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            maxVal = max(maxVal, self.getValue(successor, depth, (agentIndex + 1) % gameState.getNumAgents()))

        # print("maxVal for ", agentIndex, " :", maxVal)
        return maxVal

    def expectedValue(self, gameState: GameState, depth: int, agentIndex: int):

        actions = gameState.getLegalActions(agentIndex)
        valSum = 0

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            val = self.getValue(successor, depth, (agentIndex + 1) % gameState.getNumAgents())
            # print("val ", val)
            valSum += val

        if len(actions) > 1:
            valSum /= len(actions)

        return valSum


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <I used the same eval function I have written in the first question, the only difference being this time my function only considers the current state.
    I consider the manhattan distance to foods (need it to be as low as possible), the manhattan distance to ghosts (need it to be as high as possible), scared times, and the score.>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    """
    1.
    foodEval: sum of distance bw foods and agent
    we want it to be as small as possible!
    """

    foodEval = 0

    for food in foods.asList():
        foodEval += manhattanDistance(food, pos)

    if len(foods.asList()) != 0:
        foodEval /= len(foods.asList())  # get the avg

    """
    2.
    ghostEval: sum of distance bw ghosts and agent
    we want it to be as big as possible!
    """
    ghostEval = 0

    for ghostState in ghostStates:
        ghostEval += manhattanDistance(ghostState.getPosition(), pos)

    if len(ghostStates) != 0:
        ghostEval /= len(ghostStates)  # avg

    """
    3.
    scaredEval: sum of newScaredTimes
    as big as possible
    """

    scaredEval = 0

    for scaredTime in scaredTimes:
        scaredEval += scaredTime

    if len(scaredTimes) != 0:
        scaredEval /= len(scaredTimes)  # avg

    # once we find our variables, it is all about finding a good weight balance
    return (scaredEval * 3) + (ghostEval * 1.1) - (foodEval * 1.25) + (currentGameState.getScore() * 4)


# Abbreviation
better = betterEvaluationFunction
