import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Categorical
import datetime
import matplotlib.pyplot as plt

from env.MultiAgentEnv import DeliveryEnv
from agent.uav import UAVActor, LSTMCritic
from agent.truck import plan_truck_route
from tqdm import tqdm
import os
from util.get_device import get_device

# Set device
device = get_device()


def train(config=None, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3, num_episodes=1000, change_map=2000, n_step=10):
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
    actor.train()
    critic.train()
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    scheduler_actor = StepLR(optimizer_actor, step_size=10, gamma=0.1)
    scheduler_critic = StepLR(optimizer_critic, step_size=10, gamma=0.1)
    stats = {'Actor Loss': [], 'Critic Loss': [], 'Returns': [], 'Time cost': []}

    for episode in tqdm(range(1, num_episodes + 1)):
        obs, info = env.reset(seed=None, options={'redistribute': False if episode % change_map != 0 else True})
        episode_return = 0
        terminations = {agent: False for agent in env.possible_agents}

        if episode % change_map == 0 or episode == 1:
            route = plan_truck_route(obs["truck_0_0"]["nodes"].tolist(), list(info['center_node'].values()),
                                     env.warehouse)
            route[1].append(env.num_customer)
        truck_route = list(route[1])
        while not all(terminations.values()):
            log_probs_list = []
            values_list = []
            rewards_list = []
            dones_list = []

            for step in range(n_step):
                action_dict = {}

                # Execute truck action if available
                if obs['truck_0_0']['action_mask'] == 1 and len(truck_route) > 0:
                    action_dict['truck_0_0'] = truck_route.pop(0)

                # Execute UAV action if all UAV agents are ready
                if all(obs[agent]['action_mask'] == 1 for agent in uav_ids):
                    score = actor(obs)
                    value = critic(obs)
                    dist = Categorical(score)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    for i, agent in enumerate(uav_ids):
                        action_dict[agent] = int(action[i].item())

                        # print(int(action[i].item()), obs['uav_0_0']['choice_mask'][int(action[i].item())])
                next_obs, rewards, terminations, truncation, info = env.step(action_dict, training=True)
                if all(obs[agent]['action_mask'] == 1 for agent in uav_ids):
                    log_probs_list.append(log_prob)
                    values_list.append(value)
                    rewards_list.append(rewards['uav_0_0'])
                    dones_list.append(all(terminations.values()))
                    episode_return += rewards['uav_0_0']

                obs = next_obs
                if all(terminations.values()):
                    break

            # Bootstrap: if the episode is done, there is no next value
            if all(terminations.values()):
                next_value = 0
            else:
                next_value = critic(obs)
                next_value = next_value.item()

            # Compute returns by bootstrapping from next_value
            returns = []
            values_list.append(torch.FloatTensor([next_value]).to(device))
            len_batch = len(rewards_list)
            for x in range(len_batch):
                pass
            for reward, done_flag in zip(reversed(values_list), reversed(dones_list)):
                if done_flag:
                    R = 0  # Reset return if the episode ended
                R = reward + gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(device)

            values_list = torch.cat(values_list).to(device)
            log_probs_list = torch.stack(log_probs_list)

            # Compute the advantage (returns - value estimates)
            advantage = returns.unsqueeze(1) - values_list

            # Compute losses
            actor_loss = -(log_probs_list * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            # Total loss: you can add an entropy bonus here if desired for more exploration
            # loss = actor_loss + 0.5 * critic_loss

            # Update network parameters
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizer_actor.step()
            optimizer_critic.step()

            stats['Actor Loss'].append(actor_loss.item())
            stats['Critic Loss'].append(critic_loss.item())
        scheduler_actor.step()
        scheduler_critic.step()
        # # If a UAV action was taken, update actor and critic using A2C
        # if any(agent in action_dict for agent in uav_ids):
        #     reward = rewards['uav_0_0']
        #     value = critic(obs)
        #     next_value = critic(next_obs)
        #     done_flag = all(terminations[agent] for agent in uav_ids)
        #     td_target = reward + gamma * next_value * (1 - done_flag)
        #     advantage = td_target - value
        #
        #     critic_loss = F.mse_loss(value, td_target.detach())
        #     optimizer_critic.zero_grad()
        #     critic_loss.backward()
        #     optimizer_critic.step()
        #
        #     log_prob = dist.log_prob(action)
        #     actor_loss = -log_prob * advantage.detach()
        #     optimizer_actor.zero_grad()
        #     actor_loss.backward()
        #     optimizer_actor.step()
        #
        #     stats['Actor Loss'].append(actor_loss.item())
        #     stats['Critic Loss'].append(critic_loss.item())
        #
        # episode_return += sum(rewards.get(agent, 0) for agent in uav_ids)
        # obs = next_obs

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
        "num_customer": 100,
        "space_width": 20,
        "space_height": 20,
        "cluster_number": 8,
        "max_step": 10_000,
        "render_mode": "rgb_array",
    }
    testing_config = {
        "uav_num": 1,
        "num_customer": 20,
        "space_width": 10,
        "space_height": 10,
        "render_mode": "rgb_array",
    }
    # train(training_config, num_episodes=1000, lr_actor=10e-3, lr_critic=10e-3, gamma=0.9, n_step=8)
    train(testing_config, num_episodes=1000, lr_actor=10e-4, lr_critic=10e-4, gamma=0.9)
