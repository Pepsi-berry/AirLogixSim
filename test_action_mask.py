import sys
import os
sys.path.append('.')

from examples.run_simulation import *

# Test the first few steps to see if action mask is working correctly
observation, info = env.reset()
print('=== Initial State ===')
for agent in env.possible_agents:
    print(f'{agent}: action_mask={observation[agent]["action_mask"]}, status={env.agent_status[agent]}')

# First step: vehicle moves
action = {'vehicle_0_0': 0}  # Vehicle goes to center node of first cluster
print(f'\n=== Vehicle Action: {action} ===')
observation, reward, termination, truncation, info = env.step(action, training=True)

print('\n=== After Vehicle Move ===')
for agent in env.possible_agents:
    if agent in env.UAVs:
        print(f'{agent}: action_mask={observation[agent]["action_mask"]}, status={env.agent_status[agent]}, docked={env.isUAVDocked(agent)}')
    else:
        print(f'{agent}: action_mask={observation[agent]["action_mask"]}, status={env.agent_status[agent]}')

# Test UAV action
action = {'UAV_0_0': 5}  # UAV delivers to task 5
print(f'\n=== UAV Action: {action} ===')
observation, reward, termination, truncation, info = env.step(action, training=True)

print('\n=== After UAV Delivery ===')
for agent in env.possible_agents:
    if agent in env.UAVs:
        print(f'{agent}: action_mask={observation[agent]["action_mask"]}, status={env.agent_status[agent]}, docked={env.isUAVDocked(agent)}')
    else:
        print(f'{agent}: action_mask={observation[agent]["action_mask"]}, status={env.agent_status[agent]}')

# Test UAV return
action = {'UAV_0_0': 10}  # UAV returns to vehicle
print(f'\n=== UAV Return Action: {action} ===')
observation, reward, termination, truncation, info = env.step(action, training=True)

print('\n=== After UAV Return ===')
for agent in env.possible_agents:
    if agent in env.UAVs:
        print(f'{agent}: action_mask={observation[agent]["action_mask"]}, status={env.agent_status[agent]}, docked={env.isUAVDocked(agent)}')
    else:
        print(f'{agent}: action_mask={observation[agent]["action_mask"]}, status={env.agent_status[agent]}')

print('\n=== Test completed ===')
env.close() 