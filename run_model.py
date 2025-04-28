import torch
from env.MultiAgentEnv import DeliveryEnv
from agent.newModel import UAVActor
from agent.truck import plan_truck_route
from torch.distributions import Categorical
import time
from util.get_device import get_device


@torch.no_grad()
def run(env: DeliveryEnv, model_path: str, fps: int = 24) -> int:
    device = get_device()
    seed = 200
    static_dict = torch.load(model_path, weights_only=False, map_location=device)
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
    # input("Press Enter to start the simulation...")
    while not done:
        if obs["truck_0_0"]["action_mask"] == 1 and len(route) > 0:
            obs, _, termination, _, _ = env.step({"truck_0_0": route.pop(0)}, training=False)
        # elif obs["uav_0_0"]["action_mask"] == 1:
        #     scores = actor([obs["uav_0_0"]])
        #     dist = Categorical(logits=scores)
        #     actions = dist.sample().tolist()[0]
        #     obs, _, termination, _, _ = env.step({"uav_0_0": actions}, training=False)
        else:
            actions = {}
            for agent in env.possible_agents:
                if obs[agent]["action_mask"] == 1:
                    scores = actor([obs[agent]])
                    dist = Categorical(logits=scores)
                    action = dist.sample().tolist()[0]
                    actions[agent] = action
                    break
            obs, _, termination, _, _ = env.step(actions, training=False)
        done = all(termination.values())
        env.render()
        if env.render_mode == "human":
            time.sleep(1/fps)
        else:
            pass
            # print(env.time_step, end='\r')
    return env.time_step


if __name__ == '__main__':
    fps = 1
    env = DeliveryEnv(config = {
        "uav_num": 1,
        "group_num": 1,
        "uav_velocity": 7/fps,
        "uav_power": 20,
        "power_coefficient": 0.2,
        "truck_velocity": 10/fps,
        "max_step": 10_000_000,
        "num_customer": 100,
        "space_width": 15,
        "space_height": 10,
        "cluster_number": 5,
        "render_mode": "human",
        })
    ret = run(env, "run/20250423-223838/last.pt", fps)
    print(ret)