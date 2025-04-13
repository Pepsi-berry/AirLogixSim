import torch
from env.MultiAgentEnv import DeliveryEnv
from agent.newModel import UAVActor
from agent.truck import plan_truck_route
from torch.distributions import Categorical
import time
from util.get_device import get_device


@torch.no_grad()
def run():
    device = get_device()
    fps = 24
    seed = 42
    env = DeliveryEnv({
        "uav_num": 1,
        "group_num": 1,
        "uav_velocity": 3/fps,
        "uav_power": 20,
        "power_coefficient": 0.3,
        "truck_velocity": 10/fps,
        "max_step": 10_000_000,
        "num_customer": 100,
        "space_width": 20,
        "space_height": 10,
        "cluster_number": 5,
        "render_mode": "human"
    })
    static_dict = torch.load("run/20250411-105637/model.pt", weights_only=False)
    actor = UAVActor().to(device)
    actor.load_state_dict(static_dict['actor'])
    actor.eval()

    obs, info = env.reset(seed=seed)
    env.render()
    route = plan_truck_route(obs["truck_0_0"]["nodes"].tolist(),
                             list(info['center_node'].values()), env.warehouse)
    route[1].append(env.num_customer)
    route = route[1]

    done = False
    termination = {agent: False for agent in env.possible_agents}
    input()
    while not done:
        if obs["truck_0_0"]["action_mask"] == 1 and len(route) > 0:
            obs, _, termination, _, _ = env.step({"truck_0_0": route.pop(0)}, training=False)
        elif obs["uav_0_0"]["action_mask"] == 1:
            scores = actor([obs["uav_0_0"]])
            dist = Categorical(logits=scores)
            actions = dist.sample().tolist()[0]
            obs, _, termination, _, _ = env.step({"uav_0_0": actions}, training=False)
        else:
            obs, _, termination, _, _ = env.step({}, training=False)
        done = all(termination.values())
        env.render()
        time.sleep(1/fps)

    print(termination)
    env.close()


if __name__ == '__main__':
    run()
