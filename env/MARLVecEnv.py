import multiprocessing as mp
from copy import copy

import cloudpickle
import pickle
from env.MultiAgentEnv import UAVActionRet
from agent.truck import plan_truck_route
import random


class CloudpickleWrapper:
    """
    利用 cloudpickle 对环境构造函数进行包装，
    以解决 multiprocessing 默认 pickle 无法序列化局部函数的问题
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper, seed):
    """
    Subprocess worker func. Receive cmd and data to execute init/step/close operation.
    :param remote: pipe to communicate with parent process
    :param parent_remote: pipe to communicate with parent process
    :param env_fn_wrapper: env builder func wrapper
    """
    parent_remote.close()  # 关闭不必要的 pipe 端
    env = env_fn_wrapper.x()
    seed = random.randint(0, 2**32 - 1)
    obs, info = env.reset(seed=seed, options={'redistribute': False})
    route = plan_truck_route(obs["truck_0_0"]["nodes"].tolist(),
                             list(info['center_node'].values()), env.warehouse)
    route[1].append(env.num_customer)
    route = route[1]
    truck_route = copy(route)
    agent = None
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "init":
                observation, reward, termination, truncation, info = env.step({"truck_0_0": truck_route.pop(0)},
                                                                              training=True)
                # print("truck move")
                agent = next(iter(uav for uav, obs in observation.items() if uav.startswith("uav") and obs["action_mask"] == 1))
                remote.send((observation, reward, termination, truncation, info, int(agent.split("_")[-1])))
            elif cmd == "step":
                # result: (observations, rewards, terminations, truncation, infos)
                observation, reward, termination, truncation, info = env.step({agent: data}, training=True)
                # due to multi-agent env, should act truck when needed
                # when truck moves, reward of uav is 0, so previous reward is sent to uav
                # execute this in order to remove effects of trucks on uav
                if observation["truck_0_0"]["action_mask"] == 1 and len(truck_route) > 0:
                    observation, _, termination, truncation, info = env.step({"truck_0_0": truck_route.pop(0)},
                                                                             training=True)
                # if an env is done, reset and return new observation
                if all(termination.values()):
                    # print("done")
                    env.reset(seed=seed, options={'redistribute': False})
                    truck_route = copy(route)
                    observation, _, _, _, _ = env.step({"truck_0_0": truck_route.pop(0)}, training=True)
                    # print("truck move")
                agent = next(iter(uav for uav, obs in observation.items() if uav.startswith("uav") and obs["action_mask"] == 1))
                remote.send((observation, reward, termination, truncation, info, int(agent.split("_")[-1])))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        print("Worker: received KeyboardInterrupt")


class SubprocVectorizedMultiAgentEnv:
    """
    并行多进程多智能体环境包装器
    每个子进程运行一个 DeliveryEnv 实例
    使用 dict 存储 index -> remote, process, 等，方便单独操作
    """

    def __init__(self, env_fns, seed: int = 42):
        """
        :param env_fns: 一个列表，每个元素为创建环境实例的无参函数
        """
        self.n_envs = len(env_fns)
        self.remotes = {}
        self.work_remotes = {}
        self.processes = {}
        self.seed = seed
        for i, env_fn in enumerate(env_fns):
            parent_remote, child_remote = mp.Pipe()
            self.remotes[i] = parent_remote
            self.work_remotes[i] = child_remote
            p = mp.Process(target=worker, args=(child_remote, parent_remote, CloudpickleWrapper(env_fn), seed))
            p.daemon = True  # 主进程结束时子进程自动结束
            p.start()
            self.processes[i] = p
            child_remote.close()  # 子进程中使用 work_remote

    def init(self):
        """
        Init every env.
        :return: a dict，key is the index of each env, value is (observations, rewards, terminations, truncation, infos) tuple.
        """
        indices = list(self.remotes.keys())
        for i in indices:
            self.remotes[i].send(("init", None))

        observations, rewards, dones, truncations, infos, agents = [], [], [], [], [], []
        for i in range(self.n_envs):
            obs, reward, done, truncation, info, agent = self.remotes[i].recv()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncations.append(truncation)
            infos.append(info)
            agents.append(agent)
        return observations, rewards, dones, truncations, infos, agents

    def step(self, actions_list):
        """
        Give action to each env. An env will reset by its worker if the env is done, and new observation will be returned.
        :param uav_id:
        :param actions_list: a list containing actions with len of num envs
        :return: a dict, key is the index of each env, value is (observations, rewards, terminations, truncations, infos) tuple.
        """
        for i, actions in enumerate(actions_list):
            self.remotes[i].send(("step", actions))
        observations, rewards, dones, truncations, infos, agents = [], [], [], [], [], []
        for i in range(len(actions_list)):
            obs, reward, done, truncation, info, agent = self.remotes[i].recv()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncations.append(truncation)
            infos.append(info)
            agents.append(agent)
        return observations, rewards, dones, truncations, infos, agents

    def close(self, indices=None):
        """
        关闭指定的环境，如果 indices 为 None 则关闭所有环境
        """
        if indices is None:
            indices = list(self.remotes.keys())
        for i in indices:
            self.remotes[i].send(("close", None))
        for i in indices:
            self.processes[i].join()
