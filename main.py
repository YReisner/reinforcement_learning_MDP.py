# -*- coding: utf-8 -*-

from World import World
import numpy as np


def get_states(world):
      states = range(world.nStates)
      return states

def create_transition_matrix():
     p1 = np.zeros((world.nStates, world.nStates))
     p2 = np.zeros((world.nStates, world.nStates))
     p3 = np.zeros((world.nStates, world.nStates))
     p4 = np.zeros((world.nStates, world.nStates))

     P = np.array([p1,p2,p3,p4])

     # Create transition matrix for each action
     for a in range(world.nActions):
          for i in range(world.nStates):
               if i+1 in world.stateHoles:
                    P[a][i][i] = 1
               elif i+1 == world.stateGoal[0]:
                    P[a][i][i] = 1
               elif a+1 == 1:
                    if i+1 in [5,9]:
                         P[a][i][i] = 0.8
                    else:
                         P[a][i][i-1] = 0.8
                    if i+1 in [2,3,4]:
                         P[a][i][i] = 0.1
                    else:
                         P[a][i][i-4] = 0.1
                    if i+1 == 16:
                         P[a][i][i] = 0.1
                    else:
                         P[a][i][i + 4] = 0.1
               elif a+1 == 2:
                    if i+1 ==16:
                         P[a][i][i] = 0.9
                    else:
                         P[a][i][i+4] = 0.8
                    if i+1 in [4,8,12]:
                         P[a][i][i] = 0.1
                    elif i+1 !=16:
                         P[a][i][i+1] = 0.1
                    if i+1 in [5,9]:
                         P[a][i][i] = 0.1
                    else:
                         P[a][i][i -1] = 0.1
               elif a+1 == 3:
                    if i+1 in [4,16]:
                         P[a][i][i] = 0.9
                    elif i+1 in [8,12]:
                         P[a][i][i] = 0.8
                    else:
                         P[a][i][i+1] = 0.8
                    if i+1 in [2,3]:
                         P[a][i][i] = 0.1
                    elif i+1 != 4:
                         P[a][i][i-4] = 0.1
                    if i+1 != 16:
                         P[a][i][i + 4] = 0.1
               else:
                    if i+1 ==4:
                         P[a][i][i] = 0.9
                    elif i+1 in [2,3]:
                         P[a][i][i] = 0.8
                    else:
                         P[a][i][i-4] = 0.8
                    if i+1 in [5,9]:
                         P[a][i][i] = 0.1
                    else:
                         P[a][i][i-1] = 0.1
                    if i+1 in [8,12,16]:
                         P[a][i][i] = 0.1
                    elif i+1 not in [4,16]:
                         P[a][i][i+1] = 0.1

     # Make sure we created a valid Markov transition matrix (all rows sum to 1)
     for n in range(world.nActions):
          assert np.sum(np.sum(P[n],axis=1))==world.nStates, "Matrix %d doesn't sum to 16!" %(n)

     return P

def create_rewards(world,regular):
     # Create reward vector
     R = np.zeros((world.nStates,1)) + regular
     for i in world.stateHoles:
          R[i-1] = -1
     R[world.stateGoal[0]-1] = 1
     return R


def value_iteration(world,states,rewards,transition,discount,threshold):
     # Return optimal value and policy
     V = np.zeros((world.nStates,1)) # initiate all values to zero
     unstable_values = True
     while unstable_values:
          delta = 0
          for s in states:
               v = np.copy(V[s])
               possibilities = np.zeros((world.nActions,1))
               for i in range(world.nActions):
                    possibilities[i] = np.dot(transition[i][s],rewards) + discount*np.dot(transition[i][s],V)
               if s not in [0,6,12,13,14]:
                    V[s] = np.amax(possibilities)
               delta = np.maximum(delta,abs(v-V[s]))
          if delta < threshold:
               unstable_values = False
     policy = np.zeros((world.nStates,1))
     options = np.zeros((world.nStates,4))
     for i in range(world.nActions):
          options[:,[i]] = np.dot(transition[i],rewards) + discount * np.dot(transition[i],V)
     policy = np.argmax(options,axis=1)
     policy = policy + 1
     print("Value Iteration found optimal values and policy for discount %f and threshold %f" %(discount,threshold))
     return V,policy


def policy_iteration(world,states,rewards,transition,discount,threshold):
     # Return optimal value and policy
     Pi = np.full((world.nStates, world.nActions), 1 / world.nActions)
     unstable_policy = True
     while unstable_policy:
          values = policy_evaluation(world,transition,rewards,states,Pi,discount,threshold)
          Pi_next = policy_improvement(world,transition,rewards,states,values,discount)
          world.plot_value(values)
          world.plot_policy(np.argmax(Pi_next, axis=1) + 1)
          if (Pi_next == Pi).all():
               unstable_policy = False
          Pi = Pi_next.copy()
     Pi = np.argmax(Pi, axis=1) + 1
     print("Policy Iteration found optimal values and policy for discount %f and threshold %f" % (discount, threshold))
     return Pi, values


def policy_evaluation(world,transition, rewards,states, Pi,discount,threshold):
     # Evaluate values for given policy Pi
     V = np.zeros((world.nStates, 1))
     unstable_values = True
     while unstable_values:
          delta = 0
          for s in states:
             v = V[s].copy()
             temp_value_sum = 0
             temp_reward_sum = 0
             if s not in [0, 6, 12, 13, 14]:
                  for i in range(world.nActions):
                       temp_value_sum += Pi[s,i] * np.dot(transition[i][s], V)
                       temp_reward_sum += Pi[s,i] * np.dot(transition[i][s],rewards)
                  V[s] = temp_reward_sum + discount * temp_value_sum
             delta = np.maximum(delta, abs(v - V[s]))
          if delta < threshold:
               return V

def policy_improvement(world,transition,rewards,states,values,discount):
     # Improve policy for give values
     q = np.zeros((world.nStates, world.nActions))
     Pi_next = np.zeros((world.nStates, world.nActions))
     for s in states:
          for a in range(world.nActions):
               if s not in [0, 6, 12, 13, 14]:
                    temp = np.dot(transition[a][s], values)
               else:
                    temp = 0
               q[s, a] = np.dot(transition[a][s],rewards) + discount * temp
     for s in states:
          if s not in [0, 6, 12, 13, 14]:
               max_val = np.max(q[s, :])
               best_actions = q[s, :] == max_val
               Pi_next[s, :] = best_actions / sum(best_actions)
     return Pi_next


if __name__ == "__main__":

     world = World()
     #Part A
     P = create_transition_matrix()
     states = get_states(world)
     rewards = create_rewards(world,-0.04)
     #Part B
     value, policy = value_iteration(world, states, rewards, P, 1, 10**-4)
     world.plot_value(value)
     world.plot_policy(policy)
     #Part C
     value, policy = value_iteration(world, states, rewards, P, 0.9, 10**-4)
     world.plot_value(value)
     world.plot_policy(policy)
     #Part D
     rewards = create_rewards(world, -0.02)
     value, policy = value_iteration(world, states, rewards, P, 1, 10 ** -4)
     world.plot_value(value)
     world.plot_policy(policy)
     #Part E
     rewards = create_rewards(world, -0.04)
     value, policy = policy_iteration(world, states, rewards, P, 0.9, 10 ** -4)