import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import datetime
import matplotlib.pyplot as plt

from env.MultiAgentEnv import DeliveryEnv, UAVActionRet
from agent.truck import plan_truck_route
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def build_batch(obs):
    uav_ids = [agent for agent in obs.keys() if agent.startswith("uav")]
    keys = ["nodes", "parcel", "truck", "coordinate", "power", "capacity", "travel_distance", "choice_mask"]
    batch = {}
    for key in keys:
        # Convert each field to a tensor and stack over UAV agents.
        batch[key] = torch.stack([torch.tensor(obs[agent][key]) for agent in uav_ids], dim=0).float().to(device)
    return batch


def buildStateTensor(obs):
    # Process customer nodes.
    customers = obs["nodes"]  # (batch, num_customer, 2)
    if obs["parcel"].dim() == 2:
        parcel = obs["parcel"].unsqueeze(-1)  # (batch, num_customer, 1)
    else:
        parcel = obs["parcel"]
    if obs["choice_mask"].dim() == 2:
        choice_mask = obs["choice_mask"].unsqueeze(-1)
    else:
        choice_mask = obs["choice_mask"]
    customers = torch.cat([customers, parcel], dim=-1)  # (batch, num_customer, 3)

    # Process truck nodes.
    trucks = obs["truck"]  # (batch, truck_num, 2)
    dummy_weight = torch.zeros(trucks.size(0), 1, device=trucks.device)
    trucks = torch.cat([trucks, dummy_weight], dim=-1).unsqueeze(1)  # (batch, truck_num, 3)

    # Concatenate candidates: customers first, then trucks.
    candidates = torch.cat([customers, trucks], dim=1)  # (batch, num_candidates, 3)
    candidates = torch.cat([candidates, choice_mask], dim=-1)  # (batch, num_candidates, 4)

    return candidates


class UAVCritics(nn.Module):
    def __init__(self, feature_dim=4, candidate_embed_dim=128):
        super(UAVCritics, self).__init__()
        # Critic branch: projects the query vector to a scalar value.
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, candidate_embed_dim),
            nn.ReLU(),
            nn.Linear(candidate_embed_dim, 1)
        )

    def forward(self, obs):
        """
        Args:
            obs: Dictionary of observation tensors.
        Returns:
            value: Tensor of shape (batch, 1) with the state value estimate.
        """
        batch = build_batch(obs)
        state_tensor = buildStateTensor(batch)
        value = self.critic(state_tensor).squeeze(-1)
        value = F.softmax(value, dim=-1)
        return value


class LSTMCritic(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super().__init__()
        # input_dim = feature_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        # Map the last hidden state to a scalar value
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        """
        x: shape [batch_size, seq_length, feature_dim]
        Returns a value estimate of shape [batch_size, 1]
        """
        batch = build_batch(obs)
        x = buildStateTensor(batch)
        # out: [batch_size, seq_length, hidden_dim]
        # (h_n, c_n): last hidden & cell states, shape of h_n is [num_layers, batch_size, hidden_dim]
        out, (h_n, c_n) = self.lstm(x)

        # Take the top layer’s final hidden state: h_n[-1] => [batch_size, hidden_dim]
        last_hidden = h_n[-1]
        value = self.fc(last_hidden)  # [batch_size, 1]
        return value


class DecisionAttention(nn.Module):
    def __init__(self, embed_dim=128):
        super(DecisionAttention, self).__init__()
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.comb_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dist_linear = nn.Linear(1, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, dist, hn, proj, embed):
        query = hn[-1].unsqueeze(0)
        combined = torch.cat([query, proj, embed], dim=0)
        combined = self.linear(combined).sum(dim=0, keepdim=True)
        combined_expanded = combined.unsqueeze(1).repeat(1, dist.size(1), 1)
        combined_expanded = self.comb_linear(combined_expanded)  # [1, 21, 128]
        dist_expanded = dist.unsqueeze(-1)  # [1, 21, 1]
        dist_emb = self.dist_linear(dist_expanded)  # [1, 21, 128]
        attn_input = torch.tanh(combined_expanded + dist_emb)  # [1, 21, 128]
        scores = self.v(attn_input).squeeze(-1)  # [1, 21]
        scores = F.softmax(scores, dim=1)
        return scores  # softmax is added later


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim=128, nhead=4, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                      nhead=nhead,
                                                      dropout=dropout,
                                                      batch_first=True)
        self.feed_forward = nn.Linear(embed_dim, embed_dim)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        after_transformer = self.transformer(x)
        add1 = x + after_transformer
        bn1 = self.bn1(add1.transpose(1, 2))
        bn1 = bn1.transpose(1, 2)
        after_feat = self.feed_forward(bn1)
        add2 = bn1 + after_feat
        bn2 = self.bn2(add2.transpose(1, 2))
        return bn2.transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 nhead=4,
                 num_layers=2,
                 feature_dim=4
                 ):
        super(Encoder, self).__init__()
        self.linear_proj = nn.Linear(feature_dim, embed_dim)
        layers = [AttentionLayer(embed_dim, nhead) for _ in range(num_layers)]
        self.attention = nn.Sequential(*layers)

    def forward(self, obs):
        candidates = buildStateTensor(obs)
        linear_proj = self.linear_proj(candidates)  # (batch, num_candidates, candidate_embed_dim)
        embed = self.attention(linear_proj)
        return embed


class Decoder(nn.Module):
    def __init__(self, embed_dim=128, num_layers=3, hidden_dim=128):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.status_linear_proj = nn.Linear(3, embed_dim)
        self.attention = DecisionAttention(embed_dim)

    def forward(self, embed_last, embed_mean, obs, latest):
        nodes = obs["nodes"]
        nodes = torch.cat([nodes, obs["truck"].unsqueeze(1)], dim=1)
        dist_vec = torch.cdist(nodes, nodes, p=2)[:, latest, :]
        status = torch.cat([obs["power"], obs["capacity"], obs["travel_distance"]], dim=-1)
        proj = self.status_linear_proj(status).unsqueeze(1).transpose(0, 1)
        lstm_out, (h_n, c_n) = self.lstm(embed_last)
        # print(f"dist_vec: {dist_vec.shape}, proj: {proj.shape}, hn: {h_n.shape}, embed_mean: {embed_mean.shape}")
        score = self.attention(dist_vec, h_n, proj, embed_mean)
        return score


class UAVActor(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 attention_nhead=4,
                 attention_num_layers=2,
                 lstm_num_layers=2,
                 lstm_hidden_dim=128,
                 ):
        super(UAVActor, self).__init__()
        self.encoder = Encoder(embed_dim, attention_nhead, attention_num_layers)
        self.decoder = Decoder(embed_dim, lstm_num_layers, lstm_hidden_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, obs):
        latest = np.where(obs["uav_0_0"]["choice_mask"] == UAVActionRet.SAME_TARGET.value)[0][0]
        batch = build_batch(obs)
        x = self.encoder(batch)
        x_mean = x.mean(dim=1)
        # print(x_mean.shape)  # (batch, candidate_embed_dim)
        last_target = x[:, latest, :]
        score = self.decoder(last_target, x_mean, batch, latest) / self.scale

        # masked softmax
        infeasible = torch.Tensor(batch["choice_mask"] != 0)
        score = score.masked_fill(infeasible, -1e9)
        probs = F.softmax(score, dim=-1)
        return probs


def old_train(config=None, gamma=0.99, lr_actor=1e-1, lr_critic=1e-1, num_episodes=1000):
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
    if config is None:
        config = dict()
    env = DeliveryEnv(config)
    uav_ids = [agent for agent in env.possible_agents if agent.startswith("uav")]
    actor = UAVActor().to(device)
    critic = LSTMCritic().to(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    stats = {'Actor Loss': [], 'Critic Loss': [], 'Returns': [], 'Time cost': []}

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=None, options={'redistribute': False})
        done = False
        episode_return = 0
        env_termination = {agent: False for agent in env.possible_agents}

        truck_route = plan_truck_route(obs["truck_0_0"]["nodes"].tolist(), list(info['center_node'].values()),
                                       env.warehouse)
        truck_route[1].append(env.num_customer)
        print("==" * 50)
        while not all(env_termination.values()):
            if obs['truck_0_0']['action_mask'] == 1:
                try:
                    action_dict = {f'truck_0_0': truck_route[1].pop(0)}
                except IndexError as e:
                    break
                next_obs, rewards, terminations, truncation, info = env.step(action_dict, training=True)
                # cur_reward = sum(rewards[agent] for agent in uav_ids)
                # done = all(terminations.values())
                #
                # value = critic(obs)
                # next_value = critic(next_obs)

                # td_target = cur_reward + gamma * next_value * (1 - done)
                # critic_loss = F.mse_loss(value, td_target.detach())
                # optimizer_critic.zero_grad()
                # critic_loss.backward()
                # optimizer_critic.step()

                episode_return += sum(rewards[agent] for agent in uav_ids)
                env_termination = terminations
                obs = next_obs
            elif all(obs[agent]['action_mask'] == 1 for agent in uav_ids):
                while not all(done if agent.startswith("uav") else True for agent, done in env_termination.items()):
                    if obs['uav_0_0']['action_mask'] != 1:
                        continue

                    score = actor(obs)
                    print(score)
                    dist = Categorical(score)
                    action = dist.sample()
                    # Construct action dictionary for UAV agents.
                    action_dict = {}
                    for i, agent in enumerate(uav_ids):
                        if obs[agent]['action_mask'] == 1:
                            action_dict[agent] = int(action[i].item())
                    next_obs, rewards, terminations, truncation, info = env.step(action_dict, training=True)
                    reward = rewards['uav_0_0']
                    done = all(done if agent.startswith("uav") else True for agent, done in env_termination.items())

                    value = critic(obs)
                    next_value = critic(next_obs)

                    td_target = reward + gamma * next_value * (1 - done)
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

                    env_termination = terminations
                    obs = next_obs
                    cur_reward = 0

                    stats['Actor Loss'].append(actor_loss.item())
                    stats['Critic Loss'].append(critic_loss.item())

                    # env.render()
            else:
                action_dict = {}
                next_obs, rewards, terminations, truncation, info = env.step(action_dict)

                episode_return += sum(rewards[agent] for agent in uav_ids)
                env_termination = terminations
                obs = next_obs

            # env.render()

        stats['Returns'].append(episode_return)
        stats['Time cost'].append(info['cur_time_step'])
        # print(f"Episode {episode + 1}/{num_episodes} | Return: {episode_return}")

    torch.save(actor.state_dict(), f"uav_actor_{datetime.datetime.now()}.pth")
    torch.save(critic.state_dict(), f"uav_critic_{datetime.datetime.now()}.pth")

    # draw episode-return graph and save
    plt.plot(stats['Returns'])
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Episode-Return Graph')
    plt.savefig(f'episode_return_graph_{datetime.datetime.now()}.png')

    # draw episode-timecost graph and save
    plt.figure()
    plt.plot(stats['Time cost'])
    plt.xlabel('Episode')
    plt.ylabel('Time cost')
    plt.title('Episode-Time Cost Graph')
    plt.savefig(f'episode_timecost_graph_{datetime.datetime.now()}.png')

    env.close()


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
    if config is None:
        config = dict()
    env = DeliveryEnv(config)
    uav_ids = [agent for agent in env.possible_agents if agent.startswith("uav")]
    actor = UAVActor().to(device)
    critic = LSTMCritic().to(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    stats = {'Actor Loss': [], 'Critic Loss': [], 'Returns': [], 'Time cost': []}

    for episode in tqdm(range(1, num_episodes + 1)):
        obs, info = env.reset(seed=None, options={'redistribute': False if episode % 20 != 0 else True})
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

    torch.save(actor.state_dict(), f"uav_actor_{datetime.datetime.now()}.pth")
    torch.save(critic.state_dict(), f"uav_critic_{datetime.datetime.now()}.pth")

    # draw episode-return graph and save
    plt.plot(stats['Returns'])
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Episode-Return Graph')
    plt.savefig(f'episode_return_graph_{datetime.datetime.now()}.png')

    # draw episode-timecost graph and save
    plt.figure()
    plt.plot(stats['Time cost'])
    plt.xlabel('Episode')
    plt.ylabel('Time cost')
    plt.title('Episode-Time Cost Graph')
    plt.savefig(f'episode_timecost_graph_{datetime.datetime.now()}.png')

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
    }
    train(training_config, num_episodes=100)
