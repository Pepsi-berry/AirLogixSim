import math
import random
from abc import ABC, abstractmethod
from copy import copy
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from env.MultiAgentEnv import DeliveryEnv, UAVActionRet
from run_model import run
from util.vrp_solver import solve_vrp, calc_matrix
import json
import datetime


class _Baseline(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def eval(self, env: DeliveryEnv, truck: List[int] | int):
        """
        evaluate the baseline
        :param env: instance of DeliveryEnv
        :param truck: can be truck route, or truck numbers for TSP and CVRP
        :return:
        """
        pass


class SingleVisitBaseline(_Baseline):
    def __init__(self):
        super(SingleVisitBaseline, self).__init__()

    def eval(self, env: DeliveryEnv, routes: List[List[int]]):
        """
        Single visit baseline for classification tasks.
        """
        assert routes is not None and sum(len(route) for route in routes) == env.cluster_number + 2
        obs, info = env.reset(options={"redistribute": False})
        env.render()
        dones = False
        while not dones:
            action = {}
            for x in range(env.group_num):
                route = routes[x]
                if obs[f"truck_{x}_0"]["action_mask"] == 1 and len(route) > 0:
                    action[f"truck_{x}_0"] = route.pop(0) if len(route) > 0 else env.num_customer
                else:
                    for y in range(env.uav_num):
                        agent = f"uav_{x}_{y}"
                        if agent.startswith("uav") and obs[agent]["action_mask"] == 1:
                            if obs[agent]["choice_mask"][-1] == UAVActionRet.FEASIBLE.value:
                                action[agent] = env.num_customer
                            else:
                                action[agent] = int(
                                    np.random.choice(np.where(obs[agent]["choice_mask"] == UAVActionRet.FEASIBLE.value)[0]))
                            break
            obs, _, terminations, _, _ = env.step(action, training=True)
            dones = all(terminations.values())
            env.render()
        return env.time_step


class RandomBaseline(_Baseline):
    def __init__(self):
        super(RandomBaseline, self).__init__()

    def eval(self, env: DeliveryEnv, routes: List[List[int]]):
        """
        Random baseline for classification tasks.
        """
        assert routes is not None and sum(len(route) for route in routes) == env.cluster_number + 2
        obs, info = env.reset(options={"redistribute": False})
        env.render()
        dones = False
        while not dones:
            action = {}
            for x in range(env.group_num):
                route = routes[x]
                if obs[f"truck_{x}_0"]["action_mask"] == 1 and len(route) > 0:
                    action[f"truck_{x}_0"] = route.pop(0) if len(route) > 0 else env.num_customer
                # elif len(route) == 0:
                #     continue
                else:
                    for y in range(env.uav_num):
                        agent = f"uav_{x}_{y}"
                        if agent.startswith("uav") and obs[agent]["action_mask"] == 1:
                            try:
                                action[agent] = int(
                                    np.random.choice(np.where(obs[agent]["choice_mask"] == UAVActionRet.FEASIBLE.value)[0]))
                            except Exception as e:
                                print("!!!" * 30)
                                print(agent)
                                print(env.agent_coordinates[agent])
                                raise e
                            break
            obs, _, terminations, _, _ = env.step(action, training=True)
            dones = all(terminations.values())
            env.render()
        return env.time_step


class GreedyBaseline(_Baseline):
    def __init__(self):
        super(GreedyBaseline, self).__init__()

    def eval(self, env: DeliveryEnv, routes: List[List[int]]):
        """
        Greedy baseline for classification tasks.
        """
        assert routes is not None and sum(len(route) for route in routes) == env.cluster_number + 2
        obs, info = env.reset(options={"redistribute": False})
        env.render()
        dones = False
        while not dones:
            action = {}
            for x in range(env.group_num):
                route = routes[x]
                if obs[f"truck_{x}_0"]["action_mask"] == 1 and len(route) > 0:
                    action[f"truck_{x}_0"] = route.pop(0) if len(route) > 0 else env.num_customer
                # elif len(route) == 0:
                #     continue
                else:
                    for y in range(env.uav_num):
                        agent = f"uav_{x}_{y}"
                        if agent.startswith("uav") and obs[agent]["action_mask"] == 1:
                            for index, i in enumerate(obs[agent]["choice_mask"]):
                                if i == UAVActionRet.FEASIBLE.value:
                                    action[agent] = index
                                    break
                            break
            obs, _, terminations, _, _ = env.step(action, training=True)
            dones = all(terminations.values())
            env.render()
        return env.time_step


class TSPBaseline(_Baseline):
    def __init__(self):
        super(TSPBaseline, self).__init__()

    def eval(self, env: DeliveryEnv, truck: int):
        """
        TSP baseline implementation using nearest neighbor algorithm
        :param env: instance of DeliveryEnv
        :param truck: number of trucks available
        :return: total time consumption
        """
        env.reset(options={"redistribute": False})

        # Get node coordinates and insert warehouse at the beginning
        nodes_list = env.nodes_location.tolist()
        nodes_list.insert(0, env.warehouse)
        dist_mat = np.abs(np.array(nodes_list)[:, None, :] - np.array(nodes_list)[None, :, :]).sum(axis=2)

        # Implement nearest neighbor TSP algorithm
        def solve_tsp_nearest_neighbor(dist_matrix):
            n = len(dist_matrix)
            unvisited = set(range(1, n))  # Skip the warehouse (index 0)
            route = [0]  # Start at the warehouse

            while unvisited:
                current = route[-1]
                # Find the nearest unvisited node
                nearest = min(unvisited, key=lambda x: dist_matrix[current][x])
                route.append(nearest)
                unvisited.remove(nearest)

            # Return to warehouse
            route.append(0)
            return route

        # Solve full TSP tour
        tsp_route = solve_tsp_nearest_neighbor(dist_mat)

        # Calculate total route length
        total_length = sum(dist_mat[tsp_route[i]][tsp_route[i + 1]] for i in range(len(tsp_route) - 1))

        # If we have multiple trucks, we can split the route
        if truck > 1:
            # Calculate optimal splits for the TSP route
            # First, convert to customer-only route (remove warehouse at beginning and end)
            customer_route = tsp_route[1:-1]

            # Divide the route into truck segments as evenly as possible
            segment_size = len(customer_route) // truck
            remainder = len(customer_route) % truck

            segments = []
            start_idx = 0

            for i in range(truck):
                # Add one extra node to the first 'remainder' segments
                segment_length = segment_size + (1 if i < remainder else 0)
                segments.append(customer_route[start_idx:start_idx + segment_length])
                start_idx += segment_length

            # Calculate length of each segment (including travel to and from warehouse)
            segment_lengths = []
            for segment in segments:
                if not segment:  # Handle empty segments
                    segment_lengths.append(0)
                    continue

                # Start from warehouse to first node
                length = dist_mat[0][segment[0]]

                # Add distances between consecutive nodes in the segment
                for i in range(len(segment) - 1):
                    length += dist_mat[segment[i]][segment[i + 1]]

                # Return to warehouse from last node
                length += dist_mat[segment[-1]][0]
                segment_lengths.append(length)

            # Time is determined by the longest segment
            return max(segment_lengths) / env.truck_velocity if segment_lengths else 0
        else:
            # With just one truck, return the full route length
            return total_length / env.truck_velocity


class ClarkeWrightBaseline(_Baseline):
    def __init__(self):
        super(ClarkeWrightBaseline, self).__init__()

    @staticmethod
    def clarke_wright_savings(
            distance_matrix: List[List[float]],
            demands: List[int],
            vehicle_capacity: int,
    ) -> List[List[int]]:
        """
        Clarke–Wright 节约值算法（并行版）实现

        参数：
            distance_matrix: (n+1)x(n+1) 距离矩阵，第0号为仓库
            demands: 大小为 n+1 的需求列表，demands[0] 对应仓库，通常为 0
            vehicle_capacity: 车辆最大载量

        返回：
            routes: 路线列表，每条路线为节点索引序列（含首尾仓库0）
        """
        n = len(demands) - 1  # 客户数量
        assert len(distance_matrix) == n + 1 and all(len(row) == n + 1 for row in distance_matrix)

        # 1. 初始化：每个客户单独一条路线 [0, i, 0]
        routes = {i: [0, i, 0] for i in range(1, n + 1)}
        load = {i: demands[i] for i in range(1, n + 1)}  # 当前每条小路线的载量

        # 2. 计算所有(i,j)的节约值 S_ij = d(0,i) + d(0,j) - d(i,j)
        savings: List[Tuple[float, int, int]] = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                savings.append((s, i, j))
        # 按节约值从大到小排序
        savings.sort(reverse=True, key=lambda x: x[0])

        # 3. 依次考虑每对 (i,j)，尝试合并它们各自所在的路线
        for s, i, j in savings:
            # 找到 i、j 各自当前所在的路线编号
            route_i = next((r_id for r_id, r in routes.items() if r[1] == i), None)
            route_j = next((r_id for r_id, r in routes.items() if r[-2] == j), None)
            # 只在 i 在某条路线的末端且 j 在另一条路线的开头时才合并
            if route_i is None or route_j is None or route_i == route_j:
                continue

            # 合并后载量是否超出
            if load[route_i] + load[route_j] > vehicle_capacity:
                continue

            # 合并：去掉各自的尾零与首零，然后首尾拼接
            new_route = routes[route_i][:-1] + routes[route_j][1:]
            # 更新路线
            routes[route_i] = new_route
            load[route_i] += load[route_j]
            # 删除被合并的那条路线
            del routes[route_j]
            del load[route_j]

        # 返回所有最终路线
        return list(routes.values())

    @staticmethod
    def calc_manhattan_dist(coords):
        """
        计算曼哈顿距离矩阵（NumPy 向量化版）

        参数：
            coords: List[Tuple[float, float]] 或者 np.ndarray，形状 (n,2)
        返回：
            dist_mat: np.ndarray，形状 (n,n)
        """
        # 转成 numpy 数组，形状 (n,2)
        arr = np.array(coords)
        # arr[:, None, :] 变成 (n,1,2)，arr[None, :, :] 变成 (1,n,2)
        # 之差后取绝对值并沿最后一个维度求和，得到 (n,n)
        dist = np.abs(arr[:, None, :] - arr[None, :, :]).sum(axis=2)
        return dist

    def eval(self, env: DeliveryEnv, truck: int):
        env.reset(options={"redistribute": False})

        # Get node coordinates and insert warehouse at the beginning
        nodes_list = env.nodes_location.tolist()
        nodes_list.insert(0, env.warehouse)
        dist_mat = self.calc_manhattan_dist(nodes_list)
        demands = env.nodes_weight.tolist()
        demands.insert(0, 0)
        capacity = env.uav_capacity * 3

        routes = self.clarke_wright_savings(dist_mat, demands, capacity)

        length = []
        for idx, r in enumerate(routes, 1):
            l = 0
            for i in range(len(r) - 1):
                l += dist_mat[r[i]][r[i + 1]]
            length.append(l)

        # calculate time consumption according to the length and truck number
        if not length:
            return 0

        if len(routes) <= truck:
            # If we have enough trucks, each route can be handled by a separate truck
            # The completion time is determined by the longest route
            return max(length) / env.truck_velocity
        else:
            # If we have more routes than trucks, we need to assign multiple routes to each truck
            # Initialize truck times
            truck_times = [0] * truck

            # Sort routes by length in descending order to optimize assignment
            # (longest routes first strategy)
            sorted_routes = sorted(enumerate(length), key=lambda x: x[1], reverse=True)

            # Assign each route to the truck with the least current workload
            for route_idx, route_len in sorted_routes:
                min_time_truck = min(range(truck), key=lambda i: truck_times[i])
                truck_times[min_time_truck] += route_len

            # The total completion time is determined by the truck that finishes last
            return max(truck_times) / env.truck_velocity


def copy_route(routes):
    new_routes = list()
    for route in routes:
        new_route = copy(route)
        new_routes.append(new_route)
    return new_routes


def vrp_eval_baseline(env: DeliveryEnv, truck: int, baseline: List[str] | str):
    """
    Evaluate the baseline
    :param env: instance of DeliveryEnv
    :param truck: number of trucks available
    :param baseline: baseline name
    :return: time consumption
    """
    seed = random.randint(1, 2**31-1)
    # print(seed)
    # 1118035048
    # 1948869753
    obs, info = env.reset(seed=seed, options={"redistribute": True})
    data = {
        "distance_matrix": calc_matrix(env, obs, info),
        "num_vehicles": truck,
        "depot": 0,
    }
    routes = solve_vrp(data)
    centers = list(info['center_node'].values())
    for route in routes:
        for x in range(len(route)):
            if route[x] == 0:
                route[x] = env.num_customer
            else:
                route[x] = centers[route[x]-1]
    routes = [route[1:] for route in routes]
    # print(routes)

    if isinstance(baseline, str) and baseline == "all":
        baseline = ["random", "greedy", "tsp", "clarke_wright", "single_visit"]
    elif isinstance(baseline, str):
        baseline = [baseline]

    ret = {}
    for b in baseline:
        # print(f"Evaluating {b} baseline...")
        ret[b] = math.inf
        count = 0
        while ret[b] > env.max_step:
            if b == "random":
                ret["random"] = RandomBaseline().eval(env, copy_route(routes))
            elif b == "greedy":
                ret["greedy"] = GreedyBaseline().eval(env, copy_route(routes))
            elif b == "tsp":
                ret["tsp"] = TSPBaseline().eval(env, 1)
            elif b == "clarke_wright":
                ret["clarke_wright"] = ClarkeWrightBaseline().eval(env, truck)
            elif b == "single_visit":
                ret["single_visit"] = SingleVisitBaseline().eval(env, copy_route(routes))
            else:
                raise ValueError(f"Unknown baseline: {b}")
            count+=1
            if count > 3:
                raise ValueError(f"Baseline {b} took too long to converge.")
    return ret


if __name__ == "__main__":
    config = [
        # {
        #     "group_num": 2,
        #     "uav_num": 2,
        #     "uav_velocity": 3,
        #     "truck_velocity": 1,
        #     "uav_power": 20,
        #     "power_coefficient": 0.2,
        #     "max_step": 400,
        #     "num_customer": 20,
        #     "space_width": 5,
        #     "space_height": 5,
        #     "cluster_number": 2,
        #     "render_mode": "human"
        # },
        # {
        #     "group_num": 2,
        #     "uav_num": 2,
        #     "uav_velocity": 3,
        #     "truck_velocity": 1,
        #     "uav_power": 20,
        #     "power_coefficient": 0.2,
        #     "max_step": 400,
        #     "num_customer": 100,
        #     "space_width": 15,
        #     "space_height": 10,
        #     "cluster_number": 5,
        #     "render_mode": "human"
        # },
        # {
        #     "group_num": 2,
        #     "uav_num": 2,
        #     "uav_velocity": 3,
        #     "truck_velocity": 1,
        #     "uav_power": 20,
        #     "power_coefficient": 0.2,
        #     "max_step": 400,
        #     "num_customer": 100,
        #     "space_width": 20,
        #     "space_height": 10,
        #     "cluster_number": 5,
        #     "render_mode": "rgb_array"
        # },
        {
            "group_num": 2,
            "uav_num": 2,
            "uav_velocity": 3,
            "truck_velocity": 1,
            "uav_power": 20,
            "power_coefficient": 0.2,
            "max_step": 500,
            "num_customer": 150,
            "space_width": 20,
            "space_height": 10,
            "cluster_number": 8,
            "render_mode": "human"
        },

    ]
    num = 100
    exp_data = []
    for i, c in enumerate(config):
        stat = {"random": [], "greedy": [], "tsp": [], "clarke_wright": [], "single_visit": [], "model_best": [], "model_last": []}
        env = DeliveryEnv(c)
        for _ in tqdm(range(num)):
            ret = vrp_eval_baseline(env, truck=2, baseline="all")
            stat["random"].append(ret["random"])
            stat["greedy"].append(ret["greedy"])
            stat["tsp"].append(ret["tsp"])
            stat["clarke_wright"].append(ret["clarke_wright"])
            stat["single_visit"].append(ret["single_visit"])

            stat["model_best"].append(run(env, "run/20250423-223838/best.pt"))
            stat["model_last"].append(run(env, "run/20250423-223838/last.pt"))

            means = {k: np.mean(v) for k, v in stat.items()}
            stds = {k: np.std(v) for k, v in stat.items()}

        with open(f'exp/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump({
                "config": c,
                "mean": means,
                "std": stds,
                "stat": stat,
            }, f, indent=4)
        env.close()
