import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import datetime

from env.MultiAgentEnv import DeliveryEnv
from env.Runner import Runner
from env.VecEnv import SubprocVectorizedMultiAgentEnv
from agent.newModel import UAVActor, LSTMCritic
from agent.truck import plan_truck_route
from tqdm import tqdm
import os
from util.get_device import get_device
from util.scheduler import InverseLinearTimeDecay
from util.make_env import make_env
from util.draw import print_graph

device = get_device()


class TrainingModel:
    def __init__(self, n_updates, lr=7e-4, alpha=0.99, epsilon=1e-5, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5):
        self.actor = UAVActor(embed_dim=128,
                              attention_nhead=8,
                              attention_num_layers=3,
                              lstm_num_layers=4).to(device)
        self.critic = LSTMCritic().to(device)
        self.actor.train()
        self.critic.train()

        self.n_updates = n_updates
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
            alpha=alpha,
            eps=epsilon
        )
        self.scheduler = InverseLinearTimeDecay(self.optimizer, lr, n_updates)

    def train(self, obs, rewards, actions, values):
        """
        Train the model using the given observations, rewards, actions, and values.
        :param obs: batch of observations
        :param rewards: batch of rewards
        :param actions: batch of actions
        :param values: batch of values
        :return:
        """
        rewards = torch.tensor(rewards).to(device)
        actions = torch.tensor(actions).to(device)
        values = torch.tensor(values).to(device)
        # Compute advantages
        advantages = rewards - values

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass through actor and critic
        policy_latent = self.actor(obs)
        dist = Categorical(logits=policy_latent)

        neglogpac = -dist.log_prob(actions)

        entropy = torch.mean(dist.entropy())

        vpred = self.critic(obs)

        # Compute losses
        vf_loss = F.mse_loss(vpred, rewards.unsqueeze(1))
        pg_loss = torch.mean(advantages * neglogpac)

        loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        # Backward pass and optimization step
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        return pg_loss.item(), vf_loss.item(), entropy.item()

    @torch.no_grad()
    def eval(self, config, seed=42):
        self.actor.eval()
        env = DeliveryEnv(config)
        obs, info = env.reset(seed=seed)
        route = plan_truck_route(obs["truck_0_0"]["nodes"].tolist(),
                                 list(info['center_node'].values()), env.warehouse)
        route[1].append(env.num_customer)
        route = route[1]

        done = False
        while not done:
            if obs["truck_0_0"]["action_mask"] == 1 and len(route) > 0:
                obs, _, termination, _, info = env.step({"truck_0_0": route.pop(0)}, training=True)
            else:
                scores = self.actor([obs["uav_0_0"]])
                dist = Categorical(logits=scores)
                actions = dist.sample().tolist()[0]
                obs, _, termination, _, info = env.step({"uav_0_0": actions}, training=True)
            done = all(termination.values())

        env.close()
        self.actor.train()
        return info['cur_time_step']


def train(
        config,
        n_updates,
        num_envs=4,
        nsteps=5,
        lr=7e-4,
        gamma=0.99,
        alpha=0.99,
        epsilon=1e-5,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        log_interval=100,
        total_timesteps=int(80e6),
        save_path=None,
):
    """
    Train the model using the given environment and parameters.
    :param nsteps: after nsteps of sampling to update parameters
    :param config: Environment configuration
    :param num_envs: Number of parallel environments
    :param n_updates: Number of updates
    :param lr: Learning rate
    :param gamma: Discount factor
    :param alpha: RMSprop decay factor
    :param epsilon: RMSprop epsilon
    :param ent_coef: Entropy coefficient
    :param vf_coef: Value function coefficient
    :param max_grad_norm: Maximum gradient norm
    :param total_timesteps: Total number of timesteps to train
    :param save_path: Path to save the model
    """

    envs = [make_env(config) for _ in range(num_envs)]
    vec_env = SubprocVectorizedMultiAgentEnv(envs)
    model = TrainingModel(n_updates=n_updates, lr=lr, alpha=alpha, epsilon=epsilon, ent_coef=ent_coef,
                          vf_coef=vf_coef, max_grad_norm=max_grad_norm)
    runner = Runner(vec_env, model, nsteps=nsteps, gamma=gamma)
    stats = {'Actor Loss': [], 'Critic Loss': [], 'Entropy': [], 'Returns': [], 'Time cost': [], 'Updates': [], 'Time Cost': []}

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model.actor = torch.nn.DataParallel(model.actor)
        model.critic = torch.nn.DataParallel(model.critic)

    # Training loop
    for update in tqdm(range(1, n_updates + 1)):
        obs, rewards, actions, values = runner.run()

        # Train the model
        pg_loss, vf_loss, entropy = model.train(obs, rewards, actions, values)

        if update % log_interval == 0 or update == 1:
            # eval actor
            time_cost = model.eval(config)

            # Log statistics
            stats['Time Cost'].append(time_cost)
            stats['Actor Loss'].append(pg_loss)
            stats['Critic Loss'].append(vf_loss)
            stats['Entropy'].append(entropy)
            stats['Returns'].append(rewards.mean())
            print(f"Update {update}/{n_updates} | "
                  f"Actor Loss: {pg_loss: .4f} | Critic Loss: {vf_loss: .4f} | Entropy: {entropy: .4f} | "
                  f"Returns: {rewards.mean(): .4f} | Time Cost: {time_cost}")

    if save_path is not None:
        output_dir = os.path.join(os.getcwd(), save_path)
        os.makedirs(output_dir, exist_ok=True)
        save_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)

        torch.save(
            {
                "actor": model.actor.state_dict(),
                "critic": model.critic.state_dict(),
                "optimizer": model.optimizer.state_dict(),
                "stats": stats,
            },
            os.path.join(save_dir, 'model.pt')
        )
        print_graph(stats['Time Cost'], f'per {log_interval} updates', 'Time Cost', 'Time Cost',
                    os.path.join(save_dir, f"updates-time-cost.png"))
        print_graph(stats['Actor Loss'], f'per {log_interval} updates', 'Actor Loss', 'Actor Loss',
                    os.path.join(save_dir, f"updates-actor-loss.png"))
        print_graph(stats['Critic Loss'], f'per {log_interval} updates', 'Critic Loss', 'Critic Loss',
                    os.path.join(save_dir, f"updates-critic-loss.png"))
        print_graph(stats['Entropy'], f'per {log_interval} updates', 'Entropy', 'Entropy',
                    os.path.join(save_dir, f"updates-entropy.png"))
        print_graph(stats['Returns'], f'per {log_interval} updates', 'Returns', 'Returns',
                    os.path.join(save_dir, f"updates-return.png"))
    runner.env.close()
    return model, stats


if __name__ == "__main__":
    config_ = {
        "uav_num": 1,
        "group_num": 1,
        "uav_velocity": 3,
        "max_step": 1000,
        "num_customer": 10,
        "space_width": 10,
        "space_height": 10,
        "cluster_number": 2,
        "render_mode": "rgb_array"
    }
    num_envs_ = 4
    n_updates_ = 5000
    nsteps_ = 5
    train(config_, n_updates_, num_envs_, nsteps_, save_path='run')
