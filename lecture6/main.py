import sys
import gym
import gym_environments
import numpy as np
from agent1 import E_SARSA
from agent import SARSA
import pandas as pd
import matplotlib.pyplot as plt





def calculate_states_size(env):
    max = env.observation_space.high
    min = env.observation_space.low
    sizes = (max - min) * np.array([10, 100]) + 1
    return int(sizes[0]) * int(sizes[1])


def calculate_state(env, value):
    min = env.observation_space.low
    values = (value - min) * np.array([10, 100])
    return int(values[1]) * 19 + int(values[0])


def run(env, agent, selection_method, episodes):
    for episode in range(1, episodes + 1):
        if episode % 100 == 0:
            print(f"Episode: {episode}")
        observation, _ = env.reset()
        action = agent.get_action(calculate_state(env, observation), selection_method)
        terminated, truncated = False, False
        while not (terminated or truncated):
            new_observation, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.get_action(
                calculate_state(env, new_observation), selection_method
            )
            
            agent.update(
                calculate_state(env, observation),
                action,
                calculate_state(env, new_observation),
                next_action,
                reward,
                terminated,
                truncated,
            )
            observation = new_observation
            action = next_action

if __name__ == "__main__":
    episodes = 4000 if len(sys.argv) == 1 else int(sys.argv[1])

    env = gym.make("MountainCar-v0")

    agent = E_SARSA(
        calculate_states_size(env),
        env.action_space.n,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
    )
 


   
    # Train
    run(env, agent, "epsilon-greedy", episodes)
    env.close()
    # Play1

for _ in range(1): 
    env = gym.make("MountainCar-v0", render_mode="human")
    run(env, agent, "greedy", 1)
    agent.render()
    buff1 = 0
    buff1 = agent.step
    bufferin = [buff1]

    print("step 1:__ ",bufferin)
    env.close() 


    # Play2
    
    for _ in range(1): 
        env = gym.make("MountainCar-v0", render_mode="human")
    run(env, agent, "greedy", 1)
    agent.render()
    buff2 = 0
    buff2 = agent.step
    bufferin1 = [buff2]

    print("step 2:__ ",bufferin1)
   
    env.close() 

      # Play3
    
    for _ in range(1): 
        env = gym.make("MountainCar-v0", render_mode="human")
    run(env, agent, "greedy", 1)
    agent.render()
    buff3 = 0
    buff3 = agent.step
    bufferin2 = [buff3]

    print("step 3:__ ",bufferin2)
   
    env.close() 

      # Play4
    
    for _ in range(1): 
        env = gym.make("MountainCar-v0", render_mode="human")
    run(env, agent, "greedy", 1)
    agent.render()
    buff4 = 0
    buff4 = agent.step
    bufferin3 = [buff4]

    print("step 4:__ ",bufferin3)
   
    env.close() 

      # Play5
    
    for _ in range(1): 
        env = gym.make("MountainCar-v0", render_mode="human")
    run(env, agent, "greedy", 1)
    agent.render()
    buff5 = 0
    buff5 = agent.step
    bufferin4 = [buff5]

    print("step 5:__ ",bufferin4)
   
    env.close() 






valores = [] # inicia#lizar una lista vacía
valores.append(buff1) # 
valores.append(buff2) # 
valores.append(buff3) # 
valores.append(buff4) # 
valores.append(buff5) # 

total = sum(valores) # sumar todos los valores de la lista y asignarlo a la variable total
avg = total / 5
print("average return: ",avg) # print sum states



   #plot

# x-axis values
x = [0,0.1]
# y-axis values
y = [0,avg]
  
# plotting points as a scatter plot
plt.scatter(x, y, label= "stars", color= "green", 
            marker= "*", s=30)
  
# x-axis label
plt.xlabel('x - alpha α')
# frequency label
plt.ylabel('y - average return')
# plot title
plt.title('E_SARSA PERFORMANCE - 5 run 4000 episodes')
# showing legend
plt.legend()
  
# function to show the plot
plt.show()