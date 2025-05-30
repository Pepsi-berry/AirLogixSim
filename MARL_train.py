import math
import toml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, LambdaLR
import numpy as np
import datetime
from collections import deque

from env.MultiAgentEnv import DeliveryEnv
from env.MARLRunner import Runner
from env.MARLVecEnv import SubprocVectorizedMultiAgentEnv
from agent.MARLModel import UAVActor, UAVCritics
from agent.truck import plan_truck_route
from tqdm import tqdm
import os
from util.get_device import get_device
from util.scheduler import InverseLinearTimeDecay
from util.make_env import make_env
from util.draw import print_graph
from EntScheduler import StepEntCoef, LinearDecayEntCoef, ExponentialEntCoef

device = get_device()


class TrainingModel:
    def __init__(self, n_updates, config, lr=7e-4, alpha=0.99, epsilon=1e-5, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, model_path=None):
        # self.actor = UAVActor(embed_dim=128,
        #                       attention_nhead=8,
        #                       attention_num_layers=3,
        #                       lstm_num_layers=4,
        #                       mask_all=True).to(device)
        if model_path is None:
            self.actor = UAVActor(mask_all=True).to(device)
            self.critic = UAVCritics().to(device)
            self.optimizer = optim.RMSprop(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=lr,
                alpha=alpha,
                eps=epsilon
            )
        else:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            self.actor = UAVActor().to(device)
            self.critic = UAVCritics().to(device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer = optim.RMSprop(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=lr,
                alpha=alpha,
                eps=epsilon
            )
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=lr
            )
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.actor.train()
        self.critic.train()

        self.env = DeliveryEnv(config)
        self.n_updates = n_updates
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # self.scheduler = InverseLinearTimeDecay(self.optimizer, lr, n_updates)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9998)
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: max(1 - epoch / float(n_updates), 0))
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=n_updates, eta_min=0)

        # self.ent_scheduler = StepEntCoef(ent_coef=self.ent_coef, step_size=2000, decay=0.1)
        self.ent_scheduler = ExponentialEntCoef(ent_coef=self.ent_coef, decay=0.999)
        # self.ent_scheduler = LinearDecayEntCoef(ent_coef=self.ent_coef, n_updates=n_updates)

    def train(self, obs, rewards, actions, values, agents):
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
        policy_latent = self.actor(obs, agents)
        dist = Categorical(logits=policy_latent)

        entropy = torch.mean(dist.entropy())

        vpred = self.critic(obs)

        # Compute losses
        vf_loss = F.mse_loss(vpred, rewards.unsqueeze(1))
        pg_loss = -torch.mean(advantages.detach() * dist.log_prob(actions))

        cur_ent_coef = self.ent_scheduler.get_ent_coef()
        loss = pg_loss - entropy * cur_ent_coef + vf_loss * self.vf_coef

        # Backward pass and optimization step
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()
        self.ent_scheduler.step()

        return pg_loss.item(), vf_loss.item(), entropy.item()

    @torch.no_grad()
    def eval(self, config, seed):
        self.actor.eval()
        obs, info = self.env.reset(seed=seed)
        route = plan_truck_route(obs["truck_0_0"]["nodes"].tolist(),
                                 list(info['center_node'].values()), self.env.warehouse)
        route[1].append(self.env.num_customer)
        route = route[1]

        done = False
        # total_reward = {agent: 0 for agent in env.possible_agents if agent.startswith("uav")}
        while not done:
            if obs["truck_0_0"]["action_mask"] == 1 and len(route) > 0:
                obs, _, termination, _, info = self.env.step({"truck_0_0": route.pop(0)}, training=True)
            else:
                for x in range(self.env.uav_num):
                    agent = f"uav_0_{x}"
                    if obs[agent]["action_mask"] == 1:
                        scores = self.actor([obs], [x])
                        # greedy_action = torch.argmax(scores, dim=-1).item()
                        # obs, reward, termination, _, info = env.step({"uav_0_0": greedy_action}, training=True)
                        dist = Categorical(logits=scores)
                        actions = dist.sample().tolist()[0]
                        obs, reward, termination, _, info = self.env.step({agent: actions}, training=True)

                        # total_reward[agent] += reward[agent]
                        done = all(termination.values())

                        break
        total_reward = np.array([info["r"][agent] for agent in self.env.possible_agents if agent.startswith("uav")]).mean()
        self.actor.train()
        return self.env.time_step, total_reward

    def __del__(self):
        self.env.close()


def train(
        config,
        n_updates,
        seed=42,
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
        model_path=None
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
    vec_env = SubprocVectorizedMultiAgentEnv(envs, seed=seed)
    model = TrainingModel(config=config, n_updates=n_updates, lr=lr, alpha=alpha, epsilon=epsilon, ent_coef=ent_coef,
                          vf_coef=vf_coef, max_grad_norm=max_grad_norm, model_path=model_path)
    runner = Runner(vec_env=vec_env, model=model, nsteps=nsteps, gamma=gamma, uav_num=config["uav_num"])
    epinfo_buf = deque(maxlen=100)
    if model_path is None:
        stats = {'Actor Loss': [], 'Critic Loss': [], 'Entropy': [], 'Returns': [], 'Time cost': [], 'Updates': [], 'Time Cost': [], "ep_reward": []}
    else:
        stats = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)['stats']

    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     model.actor = torch.nn.DataParallel(model.actor)
    #     model.critic = torch.nn.DataParallel(model.critic)

    if save_path is not None:
        output_dir = os.path.join(os.getcwd(), save_path)
        os.makedirs(output_dir, exist_ok=True)
        save_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    min_cost = math.inf
    # Training loop
    for update in tqdm(range(1, n_updates + 1)):
        model.actor.eval()
        model.critic.eval()
        obs, rewards, actions, values, agents, ep_info = runner.run()
        model.actor.train()
        model.critic.train()
        epinfo_buf.extend(ep_info)

        # Train the model
        pg_loss, vf_loss, entropy = model.train(obs, rewards, actions, values, agents)

        stats['Actor Loss'].append(pg_loss)
        stats['Critic Loss'].append(vf_loss)
        stats['Entropy'].append(entropy)
        stats['Returns'].append(rewards.mean())

        if update % log_interval == 0 or update == 1:
            # eval actor
            time_cost, env_reward = model.eval(config, seed=seed)
            stats['Time Cost'].append(time_cost)
            stats["ep_reward"].append(safemean(list(epinfo_buf)))

            if time_cost < min_cost:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    {
                        "actor": model.actor.state_dict(),
                        "critic": model.critic.state_dict(),
                        "optimizer": model.optimizer.state_dict(),
                        "stats": stats,
                    },
                    os.path.join(save_dir, 'best.pt')
                )
                min_cost = time_cost

            # Log statistics
            # print(f"Update {update}/{n_updates} | "
                  # f"Actor Loss: {pg_loss: .4f} | Critic Loss: {vf_loss: .4f} | Entropy: {entropy: .4f} | "
            print(f"Returns: {rewards.mean(): .4f} | Time Cost: {time_cost} | Env Reward: {env_reward} | "
                  f"ep_reward: {safemean(list(epinfo_buf))}")

    if save_path is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "actor": model.actor.state_dict(),
                "critic": model.critic.state_dict(),
                "optimizer": model.optimizer.state_dict(),
                "stats": stats,
            },
            os.path.join(save_dir, 'last.pt')
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

        parameters = {
            "env": config,
            "hyperparameters": {
                "n_updates": n_updates,
                "sample": {
                    "seed": seed,
                    "num_envs": num_envs,
                    "nsteps": nsteps,
                    "gamma": gamma
                },
                "optimizer": {
                    "optimizer": "RMSprop",
                    "scheduler": "Linear",
                    "alpha": alpha,
                    "epsilon": epsilon,
                    "lr": lr,
                    "max_grad_norm": max_grad_norm
                },
                "entropy": {
                    "ent_coef": ent_coef,
                    "ent_scheduler": "Exponential",
                    "ent_scheduler_decay": model.ent_scheduler.decay_rate
                },
                "critic": {
                    "vf_coef": vf_coef,
                    "critic": "UAVCritics",
                }
            }
        }
        with open(os.path.join(save_dir, 'config.toml'), 'w') as f:
            toml.dump(parameters, f)
    runner.env.close()
    return model, stats


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


if __name__ == "__main__":
    config_ = {
        "uav_num": 2,
        "group_num": 1,
        "uav_velocity": 3,
        "uav_power": 20,
        "power_coefficient": 0.3,
        "truck_velocity": 100,
        "max_step": 300,
        "num_customer": 100,
        "space_width": 20,
        "space_height": 10,
        "cluster_number": 5,
        "render_mode": "rgb_array"
    }
    num_envs_ = 4
    n_updates_ = 4000
    nsteps_ = 10
    train(
        config=config_,
        n_updates=n_updates_,
        seed=200,
        num_envs=num_envs_,
        nsteps=nsteps_,
        lr=3e-4,
        gamma=0.99,
        alpha=0.99,
        epsilon=1e-5,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        log_interval=100,
        save_path='run',
        model_path=None
    )
