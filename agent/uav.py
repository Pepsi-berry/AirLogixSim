import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

from env.MultiAgentEnv import DeliveryEnv
from agent.truck import plan_truck_route

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class UAVCritics(nn.Module):
    def __init__(self, candidate_embed_dim):
        super(UAVCritics, self).__init__()
        # Critic branch: projects the query vector to a scalar value.
        self.critic = nn.Sequential(
            nn.Linear(candidate_embed_dim, candidate_embed_dim),
            nn.ReLU(),
            nn.Linear(candidate_embed_dim, 1)
        )

    def forward(self, query):
        """
        Args:
            query: Tensor of shape (batch, candidate_embed_dim) with the query vector.
        Returns:
            value: Tensor of shape (batch, 1) with the state value estimate.
        """
        value = self.critic(query)
        return value


class DecisionAttention(nn.Module):
    def __init__(self, embed_dim=128):
        super(DecisionAttention, self).__init__()
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.comb_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dist_linear = nn.Linear(1, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, dist, hn, proj, embed):
        query = hn[-1]
        combined = torch.cat([query, proj, embed], dim=0)
        combined = self.linear(combined).sum()
        combined_expanded = combined.unsqueeze(1).repeat(1, dist.size(1), 1)
        combined_expanded = self.comb_linear(combined_expanded)  # [1, 21, 128]
        dist_expanded = dist.unsqueeze(-1)  # [1, 21, 1]
        dist_emb = self.dist_linear(dist_expanded)  # [1, 21, 128]
        attn_input = torch.tanh(combined_expanded + dist_emb)  # [1, 21, 128]
        scores = self.v(attn_input).squeeze(-1)  # [1, 21]
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
        bn1 = self.bn1(add1)
        after_feat = self.feed_forward(bn1)
        add2 = bn1 + after_feat
        bn2 = self.bn2(add2)
        return bn2


class Encoder(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 nhead=4,
                 num_layers=2,
                 ):
        super(Encoder, self).__init__()
        self.linear_proj = nn.Linear(4, embed_dim)
        self.attention = nn.ModuleList([AttentionLayer(embed_dim, nhead) for _ in range(num_layers)])

    def forward(self, obs):
        # Process customer nodes.
        customers = obs["nodes"]  # (batch, num_customer, 2)
        if obs["parcel"].dim() == 2:
            parcel = obs["parcel"].unsqueeze(-1)  # (batch, num_customer, 1)
        else:
            parcel = obs["parcel"]
        customers = torch.cat([customers, parcel], dim=-1)  # (batch, num_customer, 3)

        # Process truck nodes.
        trucks = obs["truck"]  # (batch, truck_num, 2)
        dummy_weight = torch.zeros(trucks.size(0), 1, device=trucks.device)
        trucks = torch.cat([trucks, dummy_weight], dim=-1).unsqueeze(1)  # (batch, truck_num, 3)

        # Concatenate candidates: customers first, then trucks.
        candidates = torch.cat([customers, trucks], dim=1)  # (batch, num_candidates, 3)
        linear_proj = self.linear_proj(candidates)  # (batch, num_candidates, candidate_embed_dim)
        embed = self.attention(linear_proj)
        return embed


class UAVActor(nn.Module):
    def __init__(self,
                 candidate_embed_dim=128,
                 local_feat_dim=128,
                 num_transformer_layers=2,
                 nhead=4):
        """
        Shared actor–critic model for UAV agents.
        This model:
          - Builds candidate embeddings for customers and trucks (each candidate has features [x, y, weight]),
          - Encodes them with a transformer encoder,
          - Processes UAV’s local features (its coordinate, power, capacity, and travel distance)
            through an MLP to produce a query vector,
          - Computes logits via dot–product attention (scaled) between the query and each candidate embedding,
          - Applies a provided choice mask (where 0 means feasible) to obtain action probabilities,
          - And uses the query to also produce a state value estimate.
        """
        super(UAVActor, self).__init__()
        self.candidate_embed_dim = candidate_embed_dim

        # For candidate nodes: customers are given with their weight, trucks get a dummy weight 0.
        self.candidate_proj = nn.Linear(4, candidate_embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=candidate_embed_dim,
                                                   nhead=nhead,
                                                   dropout=0.1,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Process UAV local features.
        # Assume local features: coordinate (2) + power (1) + capacity (1) + travel_distance (1) = 5 features.
        self.local_feat_proj = nn.Sequential(
            nn.Linear(5, local_feat_dim),
            nn.ReLU(),
            nn.Linear(local_feat_dim, candidate_embed_dim)
        )

        # Scaling factor for dot-product attention.
        self.scale = math.sqrt(candidate_embed_dim)

    def forward(self, obs):
        """
        Args:
            obs (dict): Dictionary with keys (each a tensor with batch dimension):
                - "nodes": shape (batch, num_customer, 2) customer coordinates.
                - "parcel": shape (batch, num_customer) or (batch, num_customer, 1) customer weights.
                - "truck": shape (batch, truck_num, 2) truck coordinates.
                - "coordinate": shape (batch, 2) UAV’s own coordinate.
                - "power": shape (batch,) or (batch, 1)
                - "capacity": shape (batch,) or (batch, 1)
                - "travel_distance": shape (batch,) or (batch, 1)
                - "choice_mask": shape (batch, num_customer + truck_num); 0 means feasible.
        Returns:
            probs: Tensor of shape (batch, num_candidates) giving action probabilities.
            value: Tensor of shape (batch, 1) with the state value estimate.
        """
        # Process customer nodes.
        customers = obs["nodes"]  # (batch, num_customer, 2)
        if obs["parcel"].dim() == 2:
            parcel = obs["parcel"].unsqueeze(-1)  # (batch, num_customer, 1)
        else:
            parcel = obs["parcel"]
        customers = torch.cat([customers, parcel], dim=-1)  # (batch, num_customer, 3)

        # Process truck nodes.
        trucks = obs["truck"]  # (batch, truck_num, 2)
        dummy_weight = torch.zeros(trucks.size(0), 1, device=trucks.device)
        trucks = torch.cat([trucks, dummy_weight], dim=-1).unsqueeze(1)  # (batch, truck_num, 3)

        # Concatenate candidates: customers first, then trucks.
        candidates = torch.cat([customers, trucks], dim=1)  # (batch, num_candidates, 3)
        candidate_embed = self.candidate_proj(candidates)  # (batch, num_candidates, candidate_embed_dim)
        candidate_embed = self.transformer_encoder(candidate_embed)  # (batch, num_candidates, candidate_embed_dim)

        # Process UAV local features.
        # Ensure power, capacity, travel_distance are of shape (batch, 1).
        power = obs["power"].unsqueeze(-1) if obs["power"].dim() == 1 else obs["power"]
        capacity = obs["capacity"].unsqueeze(-1) if obs["capacity"].dim() == 1 else obs["capacity"]
        travel_distance = obs["travel_distance"].unsqueeze(-1) if obs["travel_distance"].dim() == 1 else obs[
            "travel_distance"]
        local_feats = torch.cat([obs["coordinate"], power, capacity, travel_distance], dim=-1)  # (batch, 5)
        query = self.local_feat_proj(local_feats)  # (batch, candidate_embed_dim)

        # Compute logits via scaled dot-product attention.
        logits = torch.bmm(candidate_embed, query.unsqueeze(2)).squeeze(2)  # (batch, num_candidates)
        logits = logits / self.scale
        # Mask infeasible actions: assume obs["choice_mask"] is integer; 0 indicates feasible.
        infeasible = (obs["choice_mask"] != 0)
        logits = logits.masked_fill(infeasible, -1e9)
        probs = F.softmax(logits, dim=-1)
        # Value estimate from the query.
        return probs, query


##############################################
# Shared UAV Policy (Actor–Critic)
##############################################
class UAVPolicy(nn.Module):
    def __init__(self,
                 candidate_embed_dim=128,
                 local_feat_dim=128,
                 num_transformer_layers=2,
                 nhead=4):
        """
        Shared actor–critic model for UAV agents.
        This model:
          - Builds candidate embeddings for customers and trucks (each candidate has features [x, y, weight]),
          - Encodes them with a transformer encoder,
          - Processes UAV’s local features (its coordinate, power, capacity, and travel distance)
            through an MLP to produce a query vector,
          - Computes logits via dot–product attention (scaled) between the query and each candidate embedding,
          - Applies a provided choice mask (where 0 means feasible) to obtain action probabilities,
          - And uses the query to also produce a state value estimate.
        """
        super(UAVPolicy, self).__init__()
        self.candidate_embed_dim = candidate_embed_dim

        # For candidate nodes: customers are given with their weight, trucks get a dummy weight 0.
        self.candidate_proj = nn.Linear(3, candidate_embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=candidate_embed_dim,
                                                   nhead=nhead,
                                                   dropout=0.1,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Process UAV local features.
        # Assume local features: coordinate (2) + power (1) + capacity (1) + travel_distance (1) = 5 features.
        self.local_feat_proj = nn.Sequential(
            nn.Linear(5, local_feat_dim),
            nn.ReLU(),
            nn.Linear(local_feat_dim, candidate_embed_dim)
        )

        # Scaling factor for dot-product attention.
        self.scale = math.sqrt(candidate_embed_dim)

        # Critic branch: projects the query vector to a scalar value.
        self.critic = nn.Sequential(
            nn.Linear(candidate_embed_dim, candidate_embed_dim),
            nn.ReLU(),
            nn.Linear(candidate_embed_dim, 1)
        )

    def forward(self, obs):
        """
        Args:
            obs (dict): Dictionary with keys (each a tensor with batch dimension):
                - "nodes": shape (batch, num_customer, 2) customer coordinates.
                - "parcel": shape (batch, num_customer) or (batch, num_customer, 1) customer weights.
                - "truck": shape (batch, truck_num, 2) truck coordinates.
                - "coordinate": shape (batch, 2) UAV’s own coordinate.
                - "power": shape (batch,) or (batch, 1)
                - "capacity": shape (batch,) or (batch, 1)
                - "travel_distance": shape (batch,) or (batch, 1)
                - "choice_mask": shape (batch, num_customer + truck_num); 0 means feasible.
        Returns:
            probs: Tensor of shape (batch, num_candidates) giving action probabilities.
            value: Tensor of shape (batch, 1) with the state value estimate.
        """
        # Process customer nodes.
        customers = obs["nodes"]  # (batch, num_customer, 2)
        if obs["parcel"].dim() == 2:
            parcel = obs["parcel"].unsqueeze(-1)  # (batch, num_customer, 1)
        else:
            parcel = obs["parcel"]
        customers = torch.cat([customers, parcel], dim=-1)  # (batch, num_customer, 3)

        # Process truck nodes.
        trucks = obs["truck"]  # (batch, truck_num, 2)
        dummy_weight = torch.zeros(trucks.size(0), 1, device=trucks.device)
        trucks = torch.cat([trucks, dummy_weight], dim=-1).unsqueeze(1)  # (batch, truck_num, 3)

        # Concatenate candidates: customers first, then trucks.
        candidates = torch.cat([customers, trucks], dim=1)  # (batch, num_candidates, 3)
        candidate_embed = self.candidate_proj(candidates)  # (batch, num_candidates, candidate_embed_dim)
        candidate_embed = self.transformer_encoder(candidate_embed)  # (batch, num_candidates, candidate_embed_dim)

        # Process UAV local features.
        # Ensure power, capacity, travel_distance are of shape (batch, 1).
        power = obs["power"].unsqueeze(-1) if obs["power"].dim() == 1 else obs["power"]
        capacity = obs["capacity"].unsqueeze(-1) if obs["capacity"].dim() == 1 else obs["capacity"]
        travel_distance = obs["travel_distance"].unsqueeze(-1) if obs["travel_distance"].dim() == 1 else obs[
            "travel_distance"]
        local_feats = torch.cat([obs["coordinate"], power, capacity, travel_distance], dim=-1)  # (batch, 5)
        query = self.local_feat_proj(local_feats)  # (batch, candidate_embed_dim)

        # Compute logits via scaled dot-product attention.
        logits = torch.bmm(candidate_embed, query.unsqueeze(2)).squeeze(2)  # (batch, num_candidates)
        logits = logits / self.scale
        # Mask infeasible actions: assume obs["choice_mask"] is integer; 0 indicates feasible.
        infeasible = (obs["choice_mask"] != 0)
        logits = logits.masked_fill(infeasible, -1e9)
        probs = F.softmax(logits, dim=-1)
        # Value estimate from the query.
        value = self.critic(query)
        return probs, value


##############################################
# Episode Sampling Function for UAVs
##############################################
def sample_uav_episode(model, env, device):
    """
    Runs one full episode in the multi–agent environment for all UAVs.

    Returns:
        log_prob_sum: Sum of log probabilities over all steps and agents.
        total_reward: Total reward (or negative cost) accumulated.
        final_value: The (mean) value estimate at episode end.
    """
    # Reset environment.
    obs, info = env.reset()
    # Extract UAV agent IDs (assumed to start with "uav").
    uav_ids = [agent for agent in obs.keys() if agent.startswith("uav")]
    truck_route = plan_truck_route(obs["truck_0_0"]["nodes"].tolist(), list(info['center_node'].values()),
                                   env.warehouse)
    truck_route[1].append(env.num_customer)

    # Function to build a batched observation dictionary for UAVs.
    def build_batch(obs):
        keys = ["nodes", "parcel", "truck", "coordinate", "power", "capacity", "travel_distance", "choice_mask"]
        batch = {}
        for key in keys:
            # Convert each field to a tensor and stack over UAV agents.
            batch[key] = torch.stack([torch.tensor(obs[agent][key]) for agent in uav_ids], dim=0).float().to(device)
        return batch

    batch = build_batch(obs)
    env_termination = {agent: False for agent in env.possible_agents}
    log_prob_sum = 0.0
    total_reward = 0.0
    entropy_sum = 0.0
    info = None

    # Run until all UAV agents are done.
    while not all(env_termination.values()):
        if obs['truck_0_0']['action_mask'] == 1:
            try:
                action_dict = {f'truck_0_0': truck_route[1].pop(0)}
            except IndexError as e:
                break
            # print(action_dict)
            next_obs, rewards, terminations, truncation, info = env.step(action_dict)
            # Accumulate rewards for UAV agents.
            total_reward += sum(rewards[agent] for agent in uav_ids)
            env_termination = terminations
            batch = build_batch(next_obs)
            obs = next_obs
        elif all(obs[agent]['action_mask'] == 1 for agent in uav_ids):
            while not all(done if agent.startswith("uav") else True for agent, done in env_termination.items()):
                probs, value = model(batch)  # probs: (batch, num_candidates), value: (batch, 1)
                dist = torch.distributions.Categorical(probs)
                entropy_sum += dist.entropy().sum()
                actions = dist.sample()  # (batch,)
                log_probs = dist.log_prob(actions)
                log_prob_sum += log_probs.sum()

                # Construct action dictionary for UAV agents.
                action_dict = {}
                for i, agent in enumerate(uav_ids):
                    if obs[agent]['action_mask'] == 1:
                        action_dict[agent] = int(actions[i].item())
                # print(action_dict)
                next_obs, rewards, terminations, truncation, info = env.step(action_dict)
                # Accumulate rewards for UAV agents.
                total_reward += sum(rewards[agent] for agent in uav_ids)
                env_termination = terminations
                batch = build_batch(next_obs)
                obs = next_obs
                # env.render()
                # time.sleep(0.5)
        else:
            action_dict = {}
            next_obs, rewards, terminations, truncation, info = env.step(action_dict)
            # print(action_dict)
            # Accumulate rewards for UAV agents.
            total_reward += sum(rewards[agent] for agent in uav_ids)
            env_termination = terminations
            batch = build_batch(next_obs)
            obs = next_obs

        # env.render()
        # time.sleep(0.5)

    # Get final value estimate (average over UAVs).
    _, final_value = model(batch)
    final_value = final_value.mean()
    return log_prob_sum, total_reward, final_value, entropy_sum


##############################################
# Training Loop (A2C)
##############################################
def train_uav_policy(config, num_epochs=100, lr=1e-4):
    # Initialize the multi–agent environment.
    env = DeliveryEnv(config)
    # Initialize the shared UAV policy (actor–critic).
    model = UAVPolicy(candidate_embed_dim=128, local_feat_dim=128, num_transformer_layers=2, nhead=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    epoch_history = []

    # Hyperparameters for loss.
    critic_coef = 0.5
    entropy_coef = 0.01

    try:
        for epoch in range(1, num_epochs + 1):
            log_prob_sum, total_reward, final_value, entropy_sum = sample_uav_episode(model, env, device)
            # Here, total_reward is the return (if you're minimizing cost, you might use R = -total_reward).
            R = total_reward
            advantage = R - final_value.detach()
            # Actor loss: negative log probability times advantage.
            actor_loss = -log_prob_sum * advantage
            # Critic loss: MSE between value estimate and actual return.
            critic_loss = F.mse_loss(final_value, torch.tensor(R, dtype=torch.float32, device=device))
            # Total loss: actor loss + (critic coefficient * critic loss) - (entropy coefficient * entropy bonus).
            loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy_sum

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            epoch_history.append(epoch)

            # if epoch % 10 == 0:
            print(f"Epoch {epoch}: Return = {R: .3f}, Loss = {loss.item(): .3f}, Advantage = {advantage.item(): .3f}")

        torch.save(model.state_dict(), f"uav_policy_{datetime.datetime.now()}.pth")

        # Plot the loss curve.
        plt.figure(figsize=(8, 5))
        plt.plot(epoch_history, loss_history, label='Training Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss vs. Epoch")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_plot.png", dpi=300)
    finally:
        env.close()
        # plt.show()


def train(config, epochs=100, lr_actor=10e-3, lr_critic=10e-3, gamma=0.99):
    env = DeliveryEnv(config)
    actor = UAVActor(candidate_embed_dim=128, local_feat_dim=128, num_transformer_layers=2, nhead=4).to(device)
    critic = UAVCritics(candidate_embed_dim=128).to(device)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    loss_history = []
    epoch_history = []
    return_history = []

    for epoch in range(1, epochs + 1):
        # Reset environment.
        obs, info = env.reset()
        # Extract UAV agent IDs (assumed to start with "uav").
        uav_ids = [agent for agent in obs.keys() if agent.startswith("uav")]
        truck_route = plan_truck_route(obs["truck_0_0"]["nodes"].tolist(), list(info['center_node'].values()),
                                       env.warehouse)
        truck_route[1].append(env.num_customer)


if __name__ == '__main__':
    # For reproducibility.
    import random

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    env_config = {
        "uav_num": 1,
        "uav_velocity": 5,
        "truck_velocity": 3,
        "uav_power": 20,
        "power_coefficient": 0.2,
        "num_customer": 100,
        "space_width": 25,
        "space_height": 25,
        "cluster_number": 10,
        "max_step": 10_000
    }
    train_uav_policy(config=env_config, num_epochs=1000, lr=1e-4)
