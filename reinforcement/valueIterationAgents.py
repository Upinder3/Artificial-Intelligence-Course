# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** MY CODE STARTS HERE ***"
        for i in range(self.iterations):
            vals = self.values.copy()
            for s in self.mdp.getStates():
                pval_s = []
                if self.mdp.isTerminal(s):
                    continue
                for a in self.mdp.getPossibleActions(s):
                    pval_sa = 0 
                    for ns, tpns in self.mdp.getTransitionStatesAndProbs(s,a):
                        pval_sa +=tpns * (self.mdp.getReward(s, a, ns) + self.discount * self.values[ns])
                    pval_s.append(pval_sa)

                vals[s] = max(pval_s)    
            self.values = vals
        "*** MY CODE ENDS HERE ***"

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** MY CODE STARTS HERE ***"
        qval_sa = 0
        for ns, tpns in self.mdp.getTransitionStatesAndProbs(state, action):
            qval_sa += tpns * (self.mdp.getReward(state, action, ns) + self.discount * self.values[ns])
        return qval_sa
        "*** MY CODE ENDS HERE ***"

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          in self.mdp.getPossibleActions(s): there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** MY CODE STARTS FROM HERE ***"
        max_val = float('-inf')
        act = None
        for a in self.mdp.getPossibleActions(state):
            #pval_a = 0
            #for ns, tpns in self.mdp.getTransitionStatesAndProbs(state,a):
            #    pval_a += tpns * (self.mdp.getReward(state, a, ns) + self.discount * \
            #                   self.values[ns])

            #Some refactoring after completing the project.
            pval_a = self.computeQValueFromValues(state, a)
            if pval_a > max_val:
                max_val = pval_a
                act = a
        return act
        "*** MY CODE ENDS HERE ***"

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** MY CODE STARTS HERE ***"
        len_s = len(self.mdp.getStates())

        for i in range(self.iterations):
            s = self.mdp.getStates()[i%len_s]
            pval_s = []
            if self.mdp.isTerminal(s):
                continue
            #for a in self.mdp.getPossibleActions(s):
               # pval_sa = 0.0 
               # for ns, tpns in self.mdp.getTransitionStatesAndProbs(s,a):
               #     pval_sa += tpns * (self.mdp.getReward(s, a, ns) + self.discount * self.values[ns])
               # pval_s.append(pval_sa)
            #self.values[s] = max(pval_s)

            #Some refactoring after completing the project.
            self.values[s] = self.calc_max_qval(s)

    def calc_max_qval(self, state):
        qval_max = float('-inf')
        for a in self.mdp.getPossibleActions(state):
            qval_max = max(self.computeQValueFromValues(state,a), qval_max)
        return qval_max
        "*** MY CODE ENDS HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** MY CODE STARTS HERE ***"
        predecessors = {st : set() for st in self.mdp.getStates()}
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                for ns, _ in self.mdp.getTransitionStatesAndProbs(s,a):
                    predecessors[ns].add(s)

        pqueue = util.PriorityQueue()

        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            pqueue.push(s, -abs(self.calc_max_qval(s) - self.values[s]))

        for i in range(self.iterations):
            if(pqueue.isEmpty()):
                break

            s = pqueue.pop()
            if self.mdp.isTerminal(s):
                continue
            
            self.values[s] = self.calc_max_qval(s)

            for pre_s in predecessors[s]:
                diff = abs(self.calc_max_qval(pre_s) - self.values[pre_s])
                if diff > self.theta:
                    pqueue.update(pre_s, -diff)

#    def calc_max_qval(self, state):
#        qval_max = float('-inf')
#        for a in self.mdp.getPossibleActions(state):
#            qval_max = max(self.computeQValueFromValues(state,a), qval_max)
#        return qval_max
        "*** MY CODE ENDS HERE ***"
