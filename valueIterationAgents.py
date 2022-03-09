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


from numpy import argmax
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
        "*** YOUR CODE HERE ***"
        # for iteration in range(self.iterations):
        #     prev_vk = self.values.copy()
        #     for state in self.mdp.getStates():
        #         if self.mdp.isTerminal(state):
        #             self.values[state] = 0
        #         else:
        #             qValueList = []
        #             for action in self.mdp.getPossibleActions(state):
        #                 if action == "exit":
        #                     # exiting
        #                     newQValue = self.mdp.getReward(state, action, (self.mdp.getTransitionStatesAndProbs(state, action)))
        #                     qValueList.append(newQValue)
        #                 else:
        #                     sprime_sum = 0
        #                     for sprime in self.mdp.getTransitionStatesAndProbs(state, action):
        #                         trans_prob = sprime[1]
        #                         immediate_reward = self.mdp.getReward(state, action, sprime[0])
        #                         next_reward = self.discount * prev_vk[sprime[0]]
        #                         sprime_sum = sprime_sum + (trans_prob * (immediate_reward + next_reward))
        #                     qValueList.append(sprime_sum)
        #     self.values[state] = max(qValueList)
        # states = self.mdp.getStates()
        
        # for state in states:
        #     self.values[state] = 0.0
                    
        
        for iteration in range(self.iterations):
            vk_plus1 = self.values.copy()
            
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    vk_plus1[state] = 0
                else:
                    q_list = []
                    for action in self.mdp.getPossibleActions(state):
                        q_a = self.computeQValueFromValues(state, action)
                        q_list.append(q_a)
                        
                    vk_plus1[state] = max(q_list)
            
            self.values = vk_plus1
                            


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
        "*** YOUR CODE HERE ***"
        prev_vk = self.values.copy()
        sprime_sum = 0
        
        for sprime in self.mdp.getTransitionStatesAndProbs(state, action):
                                trans_prob = sprime[1]
                                immediate_reward = self.mdp.getReward(state, action, sprime[0])
                                next_reward = self.discount * prev_vk[sprime[0]]
                                sprime_sum = sprime_sum + (trans_prob * (immediate_reward + next_reward))
        
        return sprime_sum
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        action_qVals = util.Counter()
        
        if self.mdp.isTerminal(state):
            return None
        else:
            for action in self.mdp.getPossibleActions(state):
                action_qVals[action] = self.computeQValueFromValues(state, action)
            return action_qVals.argMax()
        # actions = self.mdp.getPossibleActions(state)
        # qMax = -float('inf')
        # bestAction = None
        # for action in actions:
        #     qValue = self.computeQValueFromValues(state, action)
        #     if qValue > qMax:
        #         qMax = qValue
        #         bestAction = action
        # return bestAction
        
        # util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"
        iteration = 0

        # while iterCounter < self.iterations:
        while iteration < self.iterations:
          for state in self.mdp.getStates():
            # Q_s = util.Counter() # the Q values for given the specified state
            
            if self.mdp.isTerminal(state):
                    self.values[state] = 0
            else:
                q_list = []
                for action in self.mdp.getPossibleActions(state):
                    #Q_s[action] = self.computeQValueFromValues(state, action)
                    q_list.append(self.computeQValueFromValues(state, action))
            
                #self.values[state] = Q_s[Q_s.argMax()]
                self.values[state] = max(q_list)
            
            iteration = iteration + 1
            if iteration >= self.iterations:
              return


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
        "*** YOUR CODE HERE ***"
        
        predecessors = {} # dict
        for state in self.mdp.getStates():
          predecessors[state] = set()

        priority_queue = util.PriorityQueue()

        for state in self.mdp.getStates():
          q_list = []

          for action in self.mdp.getPossibleActions(state):
            transAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
            for transAndProb in transAndProbs:
              if transAndProb[1] != 0:
                predecessors[transAndProb[0]].add(state)

            q_list.append(self.computeQValueFromValues(state, action))

          if not self.mdp.isTerminal(state):
            max_qValue = max(q_list)
            diff = abs(self.values[state] - max_qValue)
            priority_queue.update(state, -diff)


        for iteration in range(self.iterations):
          if priority_queue.isEmpty():
            return

          state = priority_queue.pop()

          if not self.mdp.isTerminal(state):
            q_list = []
            for action in self.mdp.getPossibleActions(state):
              q_list.append(self.computeQValueFromValues(state, action))

            self.values[state] = max(q_list)

          for predecessor in predecessors[state]:
            pred_q_list = []
            for action in self.mdp.getPossibleActions(predecessor):
              pred_q_list.append(self.computeQValueFromValues(predecessor, action))

            max_pred_qValue = max(pred_q_list)
            diff = abs(self.values[predecessor] - max_pred_qValue)
              
            if diff > self.theta:
              priority_queue.update(predecessor, -diff)

