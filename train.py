import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import datetime
import matplotlib.pyplot as plt

from env.MultiAgentEnv import DeliveryEnv
from agent.newModel import UAVActor, LSTMCritic
from agent.truck import plan_truck_route
from tqdm import tqdm
import os
from util.get_device import get_device

# Set device
device = get_device()


def train(config=None, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3, num_episodes=1000):
    """
    score = actor(obs)  # (batch, num_candidates)
    value = critic(obs)  # (batch, num_candidates)
    :param config:
    :param gamma:
    :param lr_actor:
    :param lr_critic:
    :param num_episodes:
    :return:
    """
    cur_dir = os.path.join("run", datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    os.mkdir(cur_dir)
    if config is None:
        config = dict()
    env = DeliveryEnv(config)
    uav_ids = [agent for agent in env.possible_agents if agent.startswith("uav")]
    actor = UAVActor(embed_dim=128,
        attention_nhead=8,
        attention_num_layers=3,
        lstm_num_layers=4).to(device)
    critic = LSTMCritic().to(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    stats = {'Actor Loss': [], 'Critic Loss': [], 'Returns': [], 'Time cost': []}

    for episode in tqdm(range(1, num_episodes + 1)):
        obs, info = env.reset(seed=None, options={'redistribute': False if episode % 2000 != 0 else True})
        episode_return = 0
        env_termination = {agent: False for agent in env.possible_agents}

        truck_route = plan_truck_route(obs["truck_0_0"]["nodes"].tolist(), list(info['center_node'].values()),
                                       env.warehouse)
        truck_route[1].append(env.num_customer)
        # print("==" * 50)
        while not all(env_termination.values()):
            action_dict = {}

            # Execute truck action if available
            if obs['truck_0_0']['action_mask'] == 1 and len(truck_route[1]) > 0:
                action_dict['truck_0_0'] = truck_route[1].pop(0)

            # Execute UAV action if all UAV agents are ready
            if all(obs[agent]['action_mask'] == 1 for agent in uav_ids):
                score = actor(obs)
                dist = Categorical(score)
                action = dist.sample()
                for i, agent in enumerate(uav_ids):
                    action_dict[agent] = int(action[i].item())

            next_obs, rewards, terminations, truncation, info = env.step(action_dict, training=True)

            # If a UAV action was taken, update actor and critic using A2C
            if any(agent in action_dict for agent in uav_ids):
                reward = rewards['uav_0_0']
                value = critic(obs)
                next_value = critic(next_obs)
                done_flag = all(terminations[agent] for agent in uav_ids)
                td_target = reward + gamma * next_value * (1 - done_flag)
                advantage = td_target - value

                critic_loss = F.mse_loss(value, td_target.detach())
                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                log_prob = dist.log_prob(action)
                actor_loss = -log_prob * advantage.detach()
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

                stats['Actor Loss'].append(actor_loss.item())
                stats['Critic Loss'].append(critic_loss.item())

            episode_return += sum(rewards.get(agent, 0) for agent in uav_ids)
            env_termination = terminations
            obs = next_obs

        stats['Returns'].append(episode_return)
        stats['Time cost'].append(info['cur_time_step'])
        # print(f"Episode {episode + 1}/{num_episodes} | Return: {episode_return}")

    torch.save(actor.state_dict(), os.path.join(cur_dir, f"uav_actor.pth"))
    torch.save(critic.state_dict(), os.path.join(cur_dir, f"uav_critic.pth"))

    # draw episode-return graph and save
    plt.plot(stats['Returns'])
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Episode-Return Graph')
    plt.savefig(os.path.join(cur_dir, f'episode_return_graph.png'))

    # draw episode-timecost graph and save
    plt.figure()
    plt.plot(stats['Time cost'])
    plt.xlabel('Episode')
    plt.ylabel('Time cost')
    plt.title('Episode-Time Cost Graph')
    plt.savefig(os.path.join(cur_dir, f'episode_timecost_graph.png'))

    plt.figure()
    plt.plot(stats['Actor Loss'])
    plt.xlabel('Episode')
    plt.ylabel('Actor Loss')
    plt.title('Episode-Actor Loss Graph')
    plt.savefig(os.path.join(cur_dir, f'episode_actorloss_graph.png'))

    plt.figure()
    plt.plot(stats['Critic Loss'])
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.title('Episode-Critic Loss Graph')
    plt.savefig(os.path.join(cur_dir, f'episode_criticloss_graph.png'))

    env.close()


if __name__ == "__main__":
    training_config = {
        "uav_num": 1,
        "uav_velocity": 5,
        "truck_velocity": 3,
        "uav_power": 40,
        "power_coefficient": 0.3,
        "num_customer": 300,
        "space_width": 30,
        "space_height": 25,
        "cluster_number": 7,
        "max_step": 10_000,
        "render_mode": "rgb_array",
    }
    testing_config = {
        "uav_num": 1,
        "num_customer": 10,
        "render_mode": "rgb_array",
    }
    # train(training_config, num_episodes=10000)
    train(testing_config, num_episodes=30, lr_actor=10e-4, lr_critic=10e-4, gamma=0.9)
