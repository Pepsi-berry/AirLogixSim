import random

import torch
from env.MultiAgentEnv import DeliveryEnv
from agent.newModel import UAVActor
from agent.truck import plan_truck_route
from torch.distributions import Categorical
import time
from util.get_device import get_device
from util.vrp_solver import solve_vrp, calc_matrix


@torch.no_grad()
def run(env: DeliveryEnv, model_path: str, fps: int = -1) -> int:
    # select device and load model
    device = get_device()
    static_dict = torch.load(model_path, weights_only=False, map_location=device)
    actor = UAVActor().to(device)
    actor.load_state_dict(static_dict['actor'])
    actor.eval()

    # prepare the env and truck route
    obs, info = env.reset(options={"redistribute": False})
    data = {
        "distance_matrix": calc_matrix(env, obs, info),
        "num_vehicles": env.group_num,
        "depot": 0,
    }
    routes = solve_vrp(data)
    centers = list(info['center_node'].values())
    for route in routes:
        for x in range(len(route)):
            if route[x] == 0:
                route[x] = env.num_customer
            else:
                route[x] = centers[route[x]-1]
    routes = [route[1:] for route in routes]
    assert routes is not None and sum(len(route) for route in routes) == env.cluster_number + 2
    # start prediction
    env.render()
    dones = False
    while not dones:
        actions = {}
        for x in range(env.group_num):
            route = routes[x]
            if obs[f"truck_{x}_0"]["action_mask"] == 1:
                actions[f"truck_{x}_0"] = route.pop(0) if len(route) > 0 else env.num_customer
            # elif len(route) == 0:
            #     continue
            else:
                for y in range(env.uav_num):
                    agent = f"uav_{x}_{y}"
                    if agent.startswith("uav") and obs[agent]["action_mask"] == 1:
                        scores = actor([obs[agent]])
                        dist = Categorical(logits=scores)
                        action = dist.sample().tolist()[0]
                        actions[agent] = action
                        break
        obs, _, termination, _, _ = env.step(actions, training=True)
        dones = all(termination.values())
        env.render()
        if env.render_mode == "human" and fps > 0:
            time.sleep(1/fps)
        else:
            pass
            # print(env.time_step, end='\r')
    return env.time_step


@torch.no_grad()
def demo(env: DeliveryEnv, model_path: str, fps: int = -1) -> int:
    # select device and load model
    device = get_device()
    static_dict = torch.load(model_path, weights_only=False, map_location=device)
    actor = UAVActor().to(device)
    actor.load_state_dict(static_dict['actor'])
    actor.eval()
    seed = random.randint(1, 2**31 - 1)
    """
    1379949951, 604171994
    """
    print(seed)
    # prepare the env and truck route
    obs, info = env.reset(seed=604171994, options={"redistribute": False})
    data = {
        "distance_matrix": calc_matrix(env, obs, info),
        "num_vehicles": env.group_num,
        "depot": 0,
    }
    routes = solve_vrp(data)
    centers = list(info['center_node'].values())
    for route in routes:
        for x in range(len(route)):
            if route[x] == 0:
                route[x] = env.num_customer
            else:
                route[x] = centers[route[x]-1]
    routes = [route[1:] for route in routes]
    assert routes is not None and sum(len(route) for route in routes) == env.cluster_number + 2
    # start prediction
    env.render()
    time.sleep(5)
    dones = False
    while not dones:
        actions = {}
        for x in range(env.group_num):
            route = routes[x]
            if obs[f"truck_{x}_0"]["action_mask"] == 1:
                actions[f"truck_{x}_0"] = route.pop(0) if len(route) > 0 else env.num_customer
            # elif len(route) == 0:
            #     continue
            else:
                for y in range(env.uav_num):
                    agent = f"uav_{x}_{y}"
                    if agent.startswith("uav") and obs[agent]["action_mask"] == 1:
                        scores = actor([obs[agent]])
                        dist = Categorical(logits=scores)
                        action = dist.sample().tolist()[0]
                        actions[agent] = action
                        break
        obs, _, termination, _, _ = env.step(actions, training=False)
        dones = all(termination.values())
        env.render()
        if env.render_mode == "human" and fps > 0:
            time.sleep(1/fps)
        else:
            pass
            # print(env.time_step, end='\r')
    time.sleep(5)
    env.close()
    return env.time_step


if __name__ == '__main__':
    fps = 24
    env = DeliveryEnv(config={
        "uav_num": 2,
        "group_num": 2,
        "uav_velocity": 7/fps,
        "uav_power": 20,
        "power_coefficient": 0.2,
        "truck_velocity": 10/fps,
        "max_step": 10_000_000,
        "num_customer": 80,
        "space_width": 15,
        "space_height": 15,
        "cluster_number": 6,
        "render_mode": "human",
        })
    ret = demo(env, "run/20250423-223838/last.pt", fps)
    print(ret)
