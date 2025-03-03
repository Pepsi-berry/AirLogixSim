import pygame
import time
from env.MultiAgentEnv import DeliveryEnv, PackageState, UAVChoice

frame_rate = 24
env = DeliveryEnv({"uav_num": 2, "uav_velocity": 5/frame_rate, "truck_velocity": 2/frame_rate, "uav_power": 20, "power_coefficient": 0.4})
observation, info = env.reset()
env.render()

# print(env.kmeans.labels_)
# print(observation)
# print("==" * 20)
# print(info)
# exit(0)

input()

running = True
index = 1
cluster = 0
while running:
    # input()
    action = {}
    print(info['cur_time_step'])
    # print(observation)
    for agent in env.possible_agents:
        print(agent, observation[agent]['action_mask'])

    if observation['truck_0_0']['action_mask'] == 1:
        action['truck_0_0'] = info['center_node'][cluster] if cluster < len(info['center_node']) else int(info['num_customer'])
        cluster += 1
    # if observation['truck_0_0']['action_mask'] == 1 and observation['truck_0_0']['node_mask'][0] == PackageState.WAITING.value:
    #     action['truck_0_0'] = 0
    # elif observation['truck_0_0']['action_mask'] == 1:
    #     action['truck_0_0'] = int(info['num_customer'])

    # for x in range(env.uav_num):
    #     if observation[f'uav_0_{x}']['action_mask'] == 1:
    #         action[f'uav_0_{x}'] = index
    #         index = index + 1 if index < int(info['num_customer']) else index
    for x in range(env.uav_num):
        if observation[f'uav_0_{x}']['action_mask'] == 0:
            continue
        else:
            for index, i in enumerate(observation[f'uav_0_{x}']['choice_mask']):
                if i == UAVChoice.FEASIBLE.value and (all(act != index for act in action.values()) or index == env.num_customer):
                    action[f'uav_0_{x}'] = index
                    break
            # action[f'uav_0_{x}'] = 0
    observation, reward, termination, truncation, info = env.step(action)

    running = not truncation and any([not t for t in termination.values()])

    print("action     ", action)
    print("reward     ", reward)
    print("capacity   ", env.cur_uav_capacity)
    print("power      ", env.cur_uav_power)
    print("distance   ", env.cur_uav_travel_distance)
    print("termination", termination)
    print("==" * 20)

    env.render()
    time.sleep(1/frame_rate)
    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         running = False
print("==" * 20)
print(observation)
print("==" * 20)
input()

env.close()
