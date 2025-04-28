import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import datetime
import matplotlib.pyplot as plt

from env.MultiAgentEnv import DeliveryEnv
from agent.uav import UAVActor, LSTMCritic, UAVCritics
from agent.truck import plan_truck_route
from tqdm import tqdm
import os
from util.get_device import get_device

# Set device
device = get_device()


def train(config=None, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3, num_episodes=1000, change_map=2000, n_step=5):
    """
    score = actor(obs)  # (batch, num_candidates)
    value = critic(obs)  # (batch, num_candidates)
    :param n_step:
    :param change_map:
    :param config:
    :param gamma:
    :param lr_actor:
    :param lr_critic:
    :param num_episodes:
    :return:
    """

    if config is None:
        config = dict()
    env = DeliveryEnv(config)
    uav_ids = [agent for agent in env.possible_agents if agent.startswith("uav")]
    actor = UAVActor(embed_dim=128,
                     attention_nhead=8,
                     attention_num_layers=3,
                     lstm_num_layers=4).to(device)
    critic = LSTMCritic().to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        actor = torch.nn.DataParallel(actor)

    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    stats = {'Actor Loss': [], 'Critic Loss': [], 'Returns': [], 'Time cost': [], 'advantage': []}
    actor.train()
    critic.train()

    for episode in tqdm(range(1, num_episodes + 1)):
        obs, info = env.reset(seed=None, options={'redistribute': False if episode % change_map != 0 else True})
        episode_return = 0
        env_termination = {agent: False for agent in env.possible_agents}

        if episode % change_map == 0 or episode == 1:
            route = plan_truck_route(obs["truck_0_0"]["nodes"].tolist(), list(info['center_node'].values()),
                                     env.warehouse)
            route[1].append(env.num_customer)
        truck_route = list(route[1])
        actor_losses = []
        critic_losses = []
        while not all(env_termination.values()):
            action_dict = {}

            # Execute truck action if available
            if obs['truck_0_0']['action_mask'] == 1 and len(truck_route) > 0:
                action_dict['truck_0_0'] = truck_route.pop(0)

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
                td_target = td_target.detach()
                advantage = td_target - value

                critic_loss = F.mse_loss(value, td_target)
                # critic_loss = -value * advantage.detach()
                optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer_critic.step()

                log_prob = dist.log_prob(action)
                actor_loss = -log_prob * advantage.detach()
                optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer_actor.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                stats['advantage'].append(advantage.item())

            episode_return += sum(rewards.get(agent, 0) for agent in uav_ids)
            env_termination = terminations
            obs = next_obs

        stats['Returns'].append(episode_return)
        stats['Time cost'].append(info['cur_time_step'])
        stats['Actor Loss'].append(sum(actor_losses) / len(actor_losses) if len(actor_losses) > 0 else 0)
        stats['Critic Loss'].append(sum(critic_losses) / len(critic_losses) if len(critic_losses) > 0 else 0)
        # print(f"Episode {episode + 1}/{num_episodes} | Return: {episode_return}")

    cur_dir = os.path.join("run", datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    os.mkdir(cur_dir)

    torch.save(actor.state_dict(), os.path.join(cur_dir, f"uav_actor.pth"))
    torch.save(critic.state_dict(), os.path.join(cur_dir, f"uav_critic.pth"))

    plt.figure(figsize=(10, 6))
    # draw episode-return graph and save
    plt.plot(stats['Returns'])
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Episode-Return Graph')
    plt.savefig(os.path.join(cur_dir, f'episode_return_graph.png'))

    # draw episode-timecost graph and save
    # plt.figure()
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

    plt.figure()
    plt.plot(stats['advantage'])
    plt.xlabel('Step')
    plt.ylabel('Advantage')
    plt.title('Step-advantage Graph')
    plt.savefig(os.path.join(cur_dir, f'step_advantage_graph.png'))

    env.close()


if __name__ == "__main__":
    training_config = {
        "uav_num": 1,
        "uav_velocity": 5,
        "truck_velocity": 3,
        "uav_power": 40,
        "power_coefficient": 0.3,
        "num_customer": 100,
        "space_width": 20,
        "space_height": 20,
        "cluster_number": 7,
        "max_step": 10_000,
        "render_mode": "rgb_array",
    }
    testing_config = {
        "uav_num": 1,
        "num_customer": 50,
        "cluster_number": 4,
        "render_mode": "rgb_array",
    }
    # train(training_config, num_episodes=10000)
    train(testing_config, num_episodes=3, lr_actor=10e-4, lr_critic=10e-4, gamma=0.99)
