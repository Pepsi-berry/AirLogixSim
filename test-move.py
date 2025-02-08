import pygame
import time
from env.MultiAgentEnv import DeliveryEnv, PackageState

frame_rate = 1
env = DeliveryEnv({"uav_num": 10, "uav_velocity": 3/frame_rate, "truck_velocity": 1/frame_rate})
observation, info = env.reset()

print(observation)
print("==" * 20)
print(info)
# exit(0)

running = True
index = 1
while running:
    env.render()
    action = {}
    if observation['truck_0_0']['action_mask'] == 1 and observation['truck_0_0']['node_mask'][0] == PackageState.WAITING.value:
        action['truck_0_0'] = 0
    else:
        action['truck_0_0'] = int(info['num_customer'])

    for x in range(env.uav_num):
        if observation[f'uav_0_{x}']['action_mask'] == 1:
            action[f'uav_0_{x}'] = index
            index = index + 1 if index < int(info['num_customer']) else index
    observation, reward, termination, truncation, info = env.step(action)

    running = not truncation and any([not t for t in termination.values()])
    print(termination)

    time.sleep(1/frame_rate)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
print(observation)
print("==" * 20)
print(info)
input()

env.close()
