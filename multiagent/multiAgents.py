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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        #You don't want to be close to ghost.
        for ghost in newGhostStates:
            if abs(newPos[0]- newGhostStates[0].getPosition()[0]) + \
               abs(newPos[1]- newGhostStates[0].getPosition()[1]) == 1 \
               or newPos in newGhostStates[0].getPosition():
                return -float("inf") 

        #Calculate the distance to the nearest food
        dist = float("inf")
        for f in newFood.asList():
            dist = min(abs(newPos[0] - f[0]) + abs(newPos[1] - f[1]), dist)

        #returning score + reciprocal of the distance to nearest food
        # - number of foods left.
        return successorGameState.getScore() + 1/dist - len(newFood.asList())

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
        rst = self.minimax(gameState, 0, 0)

        return rst[1]

    def minimax(self, game_state, agent_num, depth, fn=False): #fn is for question 4
        if agent_num == game_state.getNumAgents():
            depth += 1
            agent_num = 0

        if game_state.isWin() or game_state.isLose() or depth == self.depth:
            return [self.evaluationFunction(game_state)]

        elif agent_num == 0: #agent is a pacman
            return self.calc_max(game_state, agent_num, depth, fn)

        elif agent_num > 0: #agent is a ghost
            return self.calc_min(game_state, agent_num, depth, fn)

    def calc_max(self, game_state, agent_num, depth, fn):
        legal_moves = game_state.getLegalActions(agent_num)

        max_val = -float("inf")
        mv = ''
        for lm in legal_moves:
            #move "lm" changed the game to new_state.
            new_state = game_state.generateSuccessor(agent_num, lm)
            #call next agent.
            eval_val  = self.minimax(new_state, agent_num + 1, depth, fn)

            if max_val < eval_val[0]:
                max_val = eval_val[0]
                mv = lm

        return [max_val, mv]

    def calc_min(self, game_state, agent_num, depth, fn):
        legal_moves = game_state.getLegalActions(agent_num)
        
        if fn: #for question 4
            min_val = 0
        else:
            min_val = float("inf")
        for i, lm in enumerate(legal_moves):
            #move "lm" changed the game to new_state.
            new_state = game_state.generateSuccessor(agent_num, lm)
            #call next agent.
            eval_val  = self.minimax(new_state, agent_num + 1, depth, fn)

            if not fn:
                if min_val > eval_val[0]:
                    min_val = eval_val[0]
            else:
                #min_val = (min_val*i + eval_val[0])/i+1
                min_val += eval_val[0]/float(len(legal_moves))
        return [min_val]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        rst = self.minimax(gameState, 0, 0, -float("inf"), float("inf"))
        return rst[1]

    def minimax(self, game_state, agent_num, depth, bst_max_val, bst_min_val):
        if agent_num == game_state.getNumAgents():
            depth += 1
            agent_num = 0

        if game_state.isWin() or game_state.isLose() or depth == self.depth:
            return [self.evaluationFunction(game_state)]

        elif agent_num == 0: #agent is a pacman
            return self.calc_max(game_state, agent_num, depth, bst_max_val, bst_min_val)

        elif agent_num > 0: #agent is a ghost
            return self.calc_min(game_state, agent_num, depth, bst_max_val, bst_min_val)

    def calc_max(self, game_state, agent_num, depth, bst_max_val, bst_min_val):
        legal_moves = game_state.getLegalActions(agent_num)

        max_val = -float("inf")
        mv = ''
        for lm in legal_moves:
            new_state = game_state.generateSuccessor(agent_num, lm)
            eval_val  = self.minimax(new_state, agent_num + 1, depth, bst_max_val, bst_min_val)

            #Prune if max_val is already greater than bst_min_val 
            if eval_val[0] > bst_min_val: 
                return [eval_val[0], mv]

            if max_val < eval_val[0]:
                max_val = eval_val[0]
                mv = lm

            bst_max_val = max(bst_max_val, max_val)

        return [max_val, mv]

    def calc_min(self, game_state, agent_num, depth, bst_max_val, bst_min_val):
        legal_moves = game_state.getLegalActions(agent_num)
        
        min_val = float("inf")
        for lm in legal_moves:
            new_state = game_state.generateSuccessor(agent_num, lm)
            eval_val  = self.minimax(new_state, agent_num + 1, depth, bst_max_val, bst_min_val)

            #Prune if min_val is already lesser than bst_max_val 
            if eval_val[0] < bst_max_val:
                return [eval_val[0]]

            if min_val > eval_val[0]:
                min_val = eval_val[0]

            bst_min_val = min(bst_min_val, min_val)

        return [min_val]


class ExpectimaxAgent(MinimaxAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        rst = self.minimax(gameState, 0, 0, True)

        return rst[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Points on the basis of which value is returned:
                 1. More the distance from ghost better the state.
                 2. Score should increase.
                 3. Should be close to the food. (1/distance to closest food)
                 4. Food items should reduce by each states.
                 5. Caps should also reduce but it's weight should be less than
                    number of food ites left.
    """
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    caps = currentGameState.getCapsules()

    g_dist = float("inf")
 
    #You don't want to be close to ghost.
    for ghost in ghostStates:
        g_dist = min(abs(currentPos[0] - ghost.getPosition()[0]) + abs(currentPos[1] - ghost.getPosition()[1]), g_dist)

    #Calculate the distance to the nearest food
    dist = float("inf")
    for f in currentFood.asList():
        dist = min(abs(currentPos[0] - f[0]) + abs(currentPos[1] - f[1]), dist)

    return g_dist + currentGameState.getScore() + 1/dist - len(currentFood.asList()) -0.9*len(caps) 

# Abbreviation
better = betterEvaluationFunction
