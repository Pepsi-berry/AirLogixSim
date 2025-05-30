import numpy as np
import torch
from torch.distributions import Categorical

from env.MARLVecEnv import SubprocVectorizedMultiAgentEnv
from copy import copy
from util.get_device import get_device


class Runner:
    """
    Used to generate batch
    """
    def __init__(self, uav_num, vec_env: SubprocVectorizedMultiAgentEnv, model, nsteps=5, gamma=0.99):
        self.env = vec_env
        self.uav_num = uav_num
        self.num_env = vec_env.n_envs
        self.model = model
        self.nsteps = nsteps
        self.gamma = gamma

        obs, _, dones, _, _, self.agent = vec_env.init()
        self.obs = obs
        self.dones = dones
        self.device = get_device()

    @torch.no_grad()
    def run(self):
        """
        Generate batch of experiences
        :return: obs, rewards, actions, values
        """
        mb_obs, mb_rewards, mb_dones, mb_actions, mb_values, mb_agents = [], [], [], [], [], []  # (num_env, 1)
        ep_info = []
        for _ in range(self.nsteps):
            agent = copy(self.agent)
            scores = self.model.actor(self.obs, agent)   # (num_env, num_actions)
            values = self.model.critic(self.obs)  # (num_env, 1)
            dist = Categorical(logits=scores)
            actions = dist.sample()         # (num_env, 1)

            mb_obs.append(copy(self.obs))
            mb_actions.append(actions.cpu())
            mb_values.append(values.cpu())
            mb_dones.append(copy(self.dones))  # (num_env, 1)
            mb_agents.append(agent)  # (num_env, 1)

            self.obs[:], rewards, self.dones[:], _, infos, self.agent[:] = self.env.step(actions.tolist())
            for info in infos:
                if info.get("end", False):
                    final_reward = np.array([info["r"][f"uav_0_{agent_}"] for agent_ in range(self.uav_num)])
                    ep_info.append(final_reward.mean())
            mb_rewards.append([reward_[f"uav_0_{agent_}"] for reward_, agent_ in zip(rewards, agent)])
        mb_dones.append(self.dones)

        mb_obs = sf01(np.asarray(mb_obs))
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = sf01(np.asarray(mb_actions, dtype=actions.cpu().numpy().dtype))
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_agents = sf01(np.asarray(mb_agents))
        mb_dones = np.asarray(mb_dones, dtype=np.bool_).swapaxes(1, 0)
        mb_dones = mb_dones[:, 1:]

        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.critic(self.obs).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+value, dones+[False], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        return mb_obs, mb_rewards, mb_actions, mb_values, mb_agents, ep_info


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


if __name__ == "__main__":
    from util.make_env import make_env
    from multitrain import TrainingModel
    config = {
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
    num_envs = 2
    envs = [make_env(config) for _ in range(num_envs)]
    parallel_env = SubprocVectorizedMultiAgentEnv(envs)
    model = TrainingModel(n_updates=100)
    runner = Runner(parallel_env, model, nsteps=5)

    obs_, rewards_, actions_, values_ = runner.run()
    print(type(obs_))
    print(obs_)
    print("==" * 50)

    print(type(rewards_))
    print(rewards_)
    print("==" * 50)

    print(type(actions_))
    print(actions_)
    print("==" * 50)

    print(type(values_))
    print(values_)

    runner.env.close()
