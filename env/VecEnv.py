import multiprocessing as mp
from copy import copy

import cloudpickle
import pickle
from env.MultiAgentEnv import UAVActionRet
from agent.truck import plan_truck_route


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

    obs, info = env.reset(seed=seed, options={'redistribute': False})
    route = plan_truck_route(obs["truck_0_0"]["nodes"].tolist(),
                             list(info['center_node'].values()), env.warehouse)
    route[1].append(env.num_customer)
    route = route[1]
    truck_route = copy(route)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "init":
                observation, reward, termination, truncation, info = env.step({"truck_0_0": truck_route.pop(0)},
                                                                              training=True)
                # print("truck move")
                remote.send((observation["uav_0_0"], reward["uav_0_0"], all(termination.values()), truncation, info))
            elif cmd == "step":
                # result: (observations, rewards, terminations, truncation, infos)
                observation, reward, termination, truncation, info = env.step({"uav_0_0": data}, training=True)
                # due to multi-agent env, should act truck when needed
                # when truck moves, reward of uav is 0, so previous reward is sent to uav
                # execute this in order to remove effects of trucks on uav
                if observation["truck_0_0"]["action_mask"] == 1 and len(truck_route) > 0:
                    observation, _, termination, truncation, info = env.step({"truck_0_0": truck_route.pop(0)},
                                                                             training=True)
                    # print("truck move")
                # if an env is done, reset and return new observation
                if all(termination.values()):
                    # print("done")
                    env.reset(seed=seed, options={'redistribute': False})
                    truck_route = copy(route)
                    observation, _, _, _, _ = env.step({"truck_0_0": truck_route.pop(0)}, training=True)
                    # print("truck move")

                remote.send((observation['uav_0_0'], reward['uav_0_0'], all(termination.values()), truncation, info))
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

        observations, rewards, dones, truncations, infos = [], [], [], [], []
        for i in range(self.n_envs):
            obs, reward, done, truncation, info = self.remotes[i].recv()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncations.append(truncation)
            infos.append(info)
        return observations, rewards, dones, truncations, infos

    def step(self, actions_list):
        """
        Give action to each env. An env will reset by its worker if the env is done, and new observation will be returned.
        :param actions_list: a list containing actions with len of num envs
        :return: a dict, key is the index of each env, value is (observations, rewards, terminations, truncations, infos) tuple.
        """
        for i, actions in enumerate(actions_list):
            self.remotes[i].send(("step", actions))
        observations, rewards, dones, truncations, infos = [], [], [], [], []
        for i in range(len(actions_list)):
            obs, reward, done, truncation, info = self.remotes[i].recv()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncations.append(truncation)
            infos.append(info)
        return observations, rewards, dones, truncations, infos

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


if __name__ == '__main__':
    from util.make_env import make_env

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
    obs, rewards, dones, _, _ = parallel_env.init()


    def num2status(choice_mask):
        ret = []
        for x in choice_mask:
            if x == UAVActionRet.FEASIBLE.value:
                ret.append("feasible")
            elif x == UAVActionRet.SAME_TARGET.value:
                ret.append("same")
            elif x == UAVActionRet.CLOSED_NODE.value:
                ret.append("closed")
            elif x == UAVActionRet.OUT_OF_CLUSTER.value:
                ret.append("out")
            else:
                ret.append("illegal")
        return ret


    for _ in range(40):
        for x in range(num_envs):
            print(dones[x], num2status(obs[x]["choice_mask"]))
        print([type(x) for x in obs])
        action_list = []
        for x in range(num_envs):
            observation_ = obs[x]
            for index, i in enumerate(observation_['choice_mask']):
                if i == UAVActionRet.FEASIBLE.value or index == config["num_customer"]:
                    action_list.append(index)
                    break
        print(action_list)
        obs, rewards, dones, _, _ = parallel_env.step(action_list)
        print(rewards)
        print("==" * 50)
