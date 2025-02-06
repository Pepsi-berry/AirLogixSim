# TODO: uav can only serve customers in the current cluster
import functools
from copy import copy
from enum import Enum
import re
import os

import numpy as np
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, Box, Tuple
from pettingzoo import ParallelEnv
from gymnasium.utils import seeding
from sklearn.cluster import KMeans

import pygame


class UAVState(Enum):
    IDLE = 0
    DELIVERING = 1
    LANDING = 2
    RETURNING = 3
    INIT = 4


class TruckState(Enum):
    MOVING = 1
    LANDING = 0


class PackageState(Enum):
    WAITING = 0
    ASSIGNED = 1
    DELIVERED = 2


class ColorFamily:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)


class DeliveryEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "delivery_env_v1",
    }

    def __init__(self, config=None):
        """initialize the environment.

        Attributes of config:
            - uav_num: the number of uavs in one group
            - truck_num: the number of trucks in one group
            - group_num: the number of groups
            - uav_velocity: the velocity of uavs
            - truck_velocity: the velocity of trucks
            - uav_weight: the weight of uavs
            - uav_capacity: the capacity of uavs
            - uav_power: the power of uavs
            - max_step: the maximum number of steps
            - num_customer: the number of customers
            - render_mode: the mode of rendering
            - space_width: half of the width of the space (2 * space_width + 1 = width of the space)
            - space_height: half of the height of the space (2 * space_height + 1 = height of the space)
            - low_power_threshold: the threshold of low power
            - power_coefficient: the coefficient of power consumption

        :param config: a dictionary containing the configuration of the environment
        """
        if config is None:
            config = {}

        # random number generator
        self.RNG, _ = seeding.np_random()

        # visualization
        self.screen = None
        self.element_size = 25
        self.screen_width = None
        self.screen_height = None
        self.material_library = {}

        # hyperparameters
        self.uav_num = config.get("uav_num", 2)
        self.truck_num = 1
        self.group_num = config.get("group_num", 1)
        self.uav_velocity = config.get("uav_velocity", 3)
        self.truck_velocity = config.get("truck_velocity", 1)
        self.uav_weight = config.get("uav_weight", 1)
        self.uav_capacity = config.get("uav_capacity", 10)
        self.uav_power = config.get("uav_power", 100)
        self.max_step = config.get("max_step", 10_000)
        self.num_customer = config.get("num_customer", 20)
        self.render_mode = config.get("render_mode", "human")
        self.space_width = config.get("space_width", 10)
        self.space_height = config.get("space_height", 10)
        self.low_power_threshold = config.get("low_power_threshold", 30)
        self.power_coefficient = config.get("power_coefficient", 0.1)
        self.cluster_number = config.get("cluster_number", 3)
        self.map_width = 2 * self.space_width + 1
        self.map_height = 2 * self.space_height + 1

        self.means = config.get("means", [0, 0])
        self.std_dev = config.get("std_devs", [1, 1])

        # agent definition
        self.possible_agents = ([f"uav_{i}_{j}" for j in range(self.uav_num) for i in range(self.group_num)]
                                + [f"truck_{i}_{j}" for j in range(self.truck_num) for i in range(self.group_num)])
        self.agents = copy(self.possible_agents)
        self.agent_name_mapping = dict(
            zip(self.possible_agents, range(len(self.possible_agents)))
        )

        # variables can be reset
        self.time_step = 0
        self.nodes_location = None
        self.nodes_weight = None
        self.node_mask = np.zeros(self.num_customer)
        self.warehouse = (0, 0)
        self.cur_uav_capacity = None
        self.cur_uav_power = None
        self.cur_uav_travel_distance = None
        self.truck_loaded_uav = None  # uav carried by truck. Truck should carry uav when it moves.
        self.agent_coordinates = None
        self.agent_status = None
        self.agent_target = None
        self.terminations = None
        self.truncation = None
        self.infos = None
        self.kmeans = None
        self.group_assigned = None  # group -> k-means cluster mapping

        # define the observation space and action space for each agent
        self.observation_spaces = {}
        self.action_spaces = {}
        for agent in self.possible_agents:
            obs = {
                "action_mask": Discrete(2),
                "node_mask": MultiDiscrete([len(PackageState)] * self.num_customer),
                # "node_mask": MultiBinary(self.num_customer),
                "nodes": Tuple([
                                   Box(low=np.array([-self.space_width, -self.space_height]),
                                       high=np.array([self.space_width, self.space_height]),
                                       dtype=np.float32)
                               ] * self.num_customer),
                "parcel": Tuple([Box(low=0, high=self.uav_capacity, dtype=np.float16)] * self.num_customer),
                "coordinate": Box(
                    low=np.array([-self.space_width, -self.space_height]),
                    high=np.array([self.space_width, self.space_height]),
                    dtype=np.float32
                )
            }
            if agent.startswith("truck"):
                obs["uav_status"] = MultiDiscrete(np.array([len(UAVState)] * self.uav_num))
                self.observation_spaces[agent] = Dict(obs)
                self.action_spaces[agent] = Discrete(self.num_customer + 1)
            else:
                obs["truck"] = Tuple([
                                         Box(
                                             low=np.array([-self.space_width, -self.space_height]),
                                             high=np.array([self.space_width, self.space_height]),
                                             dtype=np.float32
                                         )
                                     ] * self.truck_num)
                obs["power"] = Box(0, high=self.uav_power, dtype=np.float32)
                obs["capacity"] = Box(0, high=self.uav_capacity, dtype=np.float32)
                obs["travel_distance"] = Box(0, high=self.space_width * self.space_height, dtype=np.float32)
                self.observation_spaces[agent] = Dict(obs)
                self.action_spaces[agent] = Discrete(self.num_customer + self.truck_num)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(
            self,
            seed: int | None = None,
            options: dict | None = None,
    ):
        """reset the environment

        :param seed: the seed for the environment
        :param options: the options for the environment
        :return: a tuple containing the observations and info
        """
        if seed is not None:
            self.RNG, _ = seeding.np_random(seed)

        # reset the variables
        self.agents = copy(self.possible_agents)
        self.time_step = 0
        self.nodes_location = self._generate_nodes()
        self.nodes_weight = self._generate_parcel_weight()
        self.node_mask = np.array([PackageState.WAITING.value] * self.num_customer)
        # self.node_mask = np.zeros(self.num_customer)
        self.warehouse = (0, 0)
        self.cur_uav_capacity = {
            agent: self.uav_capacity for agent in self.possible_agents if agent.startswith("uav")
        }
        self.cur_uav_power = {
            agent: self.uav_power for agent in self.possible_agents if agent.startswith("uav")
        }
        self.cur_uav_travel_distance = {
            agent: 0 for agent in self.possible_agents if agent.startswith("uav")
        }
        self.agent_coordinates = {
            agent: self.warehouse for agent in self.possible_agents
        }
        self.agent_status = {agent: UAVState.INIT.value if agent.startswith("uav") else TruckState.LANDING.value for
                             agent in self.possible_agents}
        self.agent_target = {agent: None for agent in self.possible_agents}
        self.truck_loaded_uav = {}
        for agent in self.possible_agents:
            if agent.startswith("truck"):
                group_num, agent_num = self._get_agent_group(agent)
                self.truck_loaded_uav[agent] = set(f"uav_{group_num}_{x}" for x in range(self.uav_num))
        self.group_assigned = {x: None for x in range(self.group_num)}

        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncation = False
        self.infos = {"cur_time_step": self.time_step, "max_step": self.max_step, "num_customer": self.num_customer}

        self._k_means_cluster()
        observations = self._get_observations()
        return observations, self.infos

    def step(self, action):
        """take a step in the environment
        For trucks, it can choose customer nodes and warehouse as its target. The action should range from [0, num_customer].
        For uavs, it can choose customer nodes and trucks as its target. The action should range from [0, num_customer + truck_num).

        :param action: the index of nodes.
        """
        self.time_step += 1
        self.infos["cur_time_step"] = self.time_step
        if self.time_step > self.max_step:
            self.truncation = True
            observations = self._get_observations()
            rewards = {agent: 0 for agent in self.possible_agents}
            terminations = {agent: True for agent in self.possible_agents}
            return observations, rewards, terminations, self.truncation, self.infos

        rewards = {agent: -1 for agent in self.possible_agents}

        # update agent targets
        # if choose the current target, give a negative reward
        for agent, act in action.items():
            group_num, _ = self._get_agent_group(agent)
            if not self._get_action_mask(agent):
                continue
            elif act < self.num_customer:  # customer
                if self.node_mask[act] != PackageState.WAITING.value:  # the node has been assigned or delivered
                    rewards[agent] += -1
                    continue
                elif np.array_equal(np.array(self.agent_coordinates[agent]), self.nodes_location[act]):
                    rewards[agent] += -1
                    continue
                self.node_mask[act] = PackageState.ASSIGNED.value
                self.agent_target[agent] = act
                if agent.startswith("uav"):  # if uav decides to deliver a new parcel, update the power and capacity
                    self.cur_uav_power[agent] -= self.nodes_weight[act] * self.cur_uav_travel_distance[
                        agent] * self.power_coefficient
                    self.cur_uav_capacity[agent] -= self.nodes_weight[act]
                    group_num, _ = self._get_agent_group(agent)
                    self.truck_loaded_uav[f"truck_{group_num}_0"].discard(agent)  # unregister uav
                    self.agent_status[agent] = UAVState.DELIVERING.value
                else:
                    self.agent_status[agent] = TruckState.MOVING.value
                    self.group_assigned[group_num] = self._get_k_means_cluster(self.agent_coordinates[agent])
            elif agent.startswith("truck") and act == self.num_customer:
                if np.array_equal(np.array(self.agent_coordinates[agent]), self.warehouse):
                    rewards[agent] += -1
                    continue
                self.agent_target[agent] = "warehouse"
                self.group_assigned[group_num] = None
            elif agent.startswith("uav") and act < self.num_customer + self.truck_num:
                group_num, _ = self._get_agent_group(agent)
                if np.array_equal(np.array(self.agent_coordinates[agent]), self.agent_coordinates[f"truck_{group_num}_0"]):
                    rewards[agent] += -1
                    continue
                self.agent_target[agent] = f"truck_{group_num}_{act - self.num_customer}"
            else:
                raise ValueError(f"Unknown action: {act}, agent: {agent}")

        # move to target
        for agent, target in self.agent_target.items():
            if target is None:
                continue
            agent_coordinate = self.agent_coordinates[agent]

            if type(target) is int:
                target_coordinate = self.nodes_location[target]
            elif target == 'warehouse':
                target_coordinate = self.warehouse
            elif target.startswith("truck"):
                target_coordinate = self.agent_coordinates[target]
            else:
                raise ValueError(f"Unknown target: {target}, agent: {agent}")

            distance = np.linalg.norm(np.array(target_coordinate) - np.array(agent_coordinate))
            x_distance = target_coordinate[0] - agent_coordinate[0]
            y_distance = target_coordinate[1] - agent_coordinate[1]

            if agent.startswith("uav"):
                if self.cur_uav_capacity[agent] < 0 or self.cur_uav_power[agent] <= 0:  # dead
                    rewards[agent] += -100
                    if target < self.num_customer:
                        self.node_mask[target] = PackageState.WAITING.value
                    self.cur_uav_capacity[agent] = self.uav_capacity
                    self.cur_uav_power[agent] = self.uav_power
                    self.cur_uav_travel_distance[agent] = 0
                    self.agent_status[agent] = UAVState.IDLE.value
                    self.agent_target[agent] = None
                    group_num, _ = self._get_agent_group(agent)
                    self.agent_coordinates[agent] = self.agent_coordinates[f"truck_{group_num}_0"]
                elif type(target) is str and target.startswith(
                        "truck") and distance <= self.uav_velocity:  # return to truck
                    group_num, _ = self._get_agent_group(agent)
                    self.truck_loaded_uav[f"truck_{group_num}_0"].add(agent)  # register uav
                    self.agent_coordinates[agent] = target_coordinate
                    rewards[agent] += 0 if self.cur_uav_travel_distance[agent] == 0 else (10 + self.cur_uav_travel_distance[agent] * 0.1 + (
                            self.uav_capacity - self.cur_uav_capacity[agent]) * 0.5)  # finish delivery
                    rewards[agent] += 5 * (1 if self.cur_uav_power[agent] < self.low_power_threshold else -1)
                    self.cur_uav_travel_distance[agent] = 0
                    self.cur_uav_power[agent] = self.uav_power
                    self.cur_uav_capacity[agent] = self.uav_capacity
                    self.agent_status[agent] = UAVState.IDLE.value
                    self.agent_target[agent] = None
                    self.truck_loaded_uav[target].add(agent)
                elif distance <= self.uav_velocity:
                    self.agent_coordinates[agent] = target_coordinate
                    self.cur_uav_travel_distance[agent] += distance
                    self.cur_uav_power[agent] -= (self.nodes_weight[
                                                      target] + self.uav_weight) * distance * self.power_coefficient
                    self.agent_status[agent] = UAVState.LANDING.value
                    self.node_mask[target] = PackageState.DELIVERED.value
                    self.agent_target[agent] = None
                else:
                    self.agent_coordinates[agent] = (
                        agent_coordinate[0] + x_distance * self.uav_velocity / distance,
                        agent_coordinate[1] + y_distance * self.uav_velocity / distance)
                    self.cur_uav_travel_distance[agent] += self.uav_velocity

                    cur_uav_load = 0 if type(target) is str and target.startswith("truck") else self.nodes_weight[target]
                    self.cur_uav_power[agent] -= (cur_uav_load + self.uav_weight) * self.uav_velocity * self.power_coefficient
            else:
                group_num, _ = self._get_agent_group(agent)
                for x in range(self.truck_velocity):
                    if agent_coordinate[0] != target_coordinate[0]:
                        direction = 1 if x_distance > 0 else -1
                        self.agent_coordinates[agent] = (agent_coordinate[0] + direction, agent_coordinate[1])
                        for uav in self.truck_loaded_uav[agent]:
                            self.agent_coordinates[uav] = self.agent_coordinates[agent]
                    elif agent_coordinate[1] != target_coordinate[1]:
                        direction = 1 if y_distance > 0 else -1
                        self.agent_coordinates[agent] = (agent_coordinate[0], agent_coordinate[1] + direction)
                        for uav in self.truck_loaded_uav[agent]:
                            self.agent_coordinates[uav] = self.agent_coordinates[agent]
                    else:  # target is reached
                        self.agent_status[agent] = TruckState.LANDING.value
                        self.agent_target[agent] = None
                        if target != 'warehouse':
                            self.node_mask[target] = PackageState.DELIVERED.value
                            for uav in self.truck_loaded_uav[agent]:
                                self.terminations[uav] = False  # open for delivery
                                self.agent_status[uav] = UAVState.IDLE.value
                            rewards[agent] += 1
                        else:
                            self.agent_status[agent] = TruckState.LANDING.value
                            if np.all(self.node_mask == PackageState.DELIVERED.value):
                                for every_agent in self.possible_agents:
                                    rewards[every_agent] += 100
                                self.terminations[agent] = True
                            else:
                                rewards[agent] += -10
                        break

        # Only when all clusters are delivered, all uavs return to the truck and truck is at warehouse, the truck can move.
        for agent in self.possible_agents:
            group_num, _ = self._get_agent_group(agent)
            if agent.startswith("uav") and self.group_assigned[group_num] is None:
                self.terminations[agent] = False
            elif agent.startswith("uav"):
                self.terminations[agent] = self._is_cluster_delivered(self.group_assigned[group_num]) and self._get_action_mask(agent)
            else:  # truck
                self.terminations[agent] = (self._get_action_mask(agent)
                                            and self.agent_coordinates[agent] == self.warehouse
                                            and all(self._is_cluster_delivered(cluster) for cluster in range(self.cluster_number)))

        observations = self._get_observations()
        return observations, rewards, self.terminations, self.truncation, self.infos

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                self.screen_height = self.map_height * self.element_size
                self.screen_width = self.map_width * self.element_size
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                self._load_material()
                pygame.display.set_caption("Delivery Environment")
            self.screen.fill(ColorFamily.WHITE)
            self._draw_grid()

            self._draw_point(0, 0, "Warehouse")
            for i, node in enumerate(self.nodes_location):
                if self.node_mask[i] == PackageState.WAITING.value:
                    self._draw_point(*node, "Customer")
                elif self.node_mask[i] == PackageState.ASSIGNED.value:
                    self._draw_point(*node, "CustomerAssigned")
                else:
                    self._draw_point(*node, "CustomerClosed")
            for agent, coordinate in self.agent_coordinates.items():
                if agent.startswith("uav"):
                    self._draw_point(*coordinate, "UAV")
                else:
                    self._draw_point(*coordinate, "Truck")
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            pass
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _get_observations(self):
        """get the observations of the agents

        :return: the observations of the agents
        """
        observations = {}
        for agent in self.possible_agents:
            group_num, agent_num = self._get_agent_group(agent)

            obs = {
                "action_mask": int(self._get_action_mask(agent)),
                "node_mask": self.node_mask,
                "nodes": self.nodes_location,
                "parcel": self.nodes_weight,
                "coordinate": self.agent_coordinates[agent]
            }
            if agent.startswith("truck"):
                obs["uav_status"] = np.zeros(self.uav_num)
                for x in range(self.uav_num):
                    obs["uav_status"][x] = self.agent_status[f"uav_{group_num}_{x}"]
                observations[agent] = obs
            else:
                obs["truck"] = np.zeros((self.truck_num, 2))
                for x in range(self.truck_num):
                    obs["truck"] = np.array(self.agent_coordinates[f"truck_{group_num}_{x}"])
                obs["power"] = self.cur_uav_power[agent]
                obs["capacity"] = self.cur_uav_capacity[agent]
                obs["travel_distance"] = self.cur_uav_travel_distance[agent]
                observations[agent] = obs
        return observations

    def _get_info(self):
        pass

    def _get_action_mask(self, agent):
        """get the action mask of the agent.

        - One UAV can move when all the trucks are landing, and it is idle.
        - One truck can move when all the UAVs are idle, and it is landing.

        :param agent: the name of the agent
        :return: the action mask of the agent
        """
        status = self.agent_status[agent]
        group_num, agent_num = self._get_agent_group(agent)

        if agent.startswith("truck"):
            if self.group_assigned[group_num] is None:
                return status == TruckState.LANDING.value and all(
                    self.agent_status[f"uav_{group_num}_{i}"] in (UAVState.IDLE.value, UAVState.INIT.value) for i in
                    range(self.uav_num))
            else:
                return (status == TruckState.LANDING.value and all(
                    self.agent_status[f"uav_{group_num}_{i}"] in (UAVState.IDLE.value, UAVState.INIT.value) for i in range(self.uav_num))
                        and self._is_cluster_delivered(self.group_assigned[group_num]))
        else:
            return (status == UAVState.IDLE.value or status == UAVState.LANDING.value) and all(
                self.agent_status[f"truck_{group_num}_{i}"] == TruckState.LANDING.value for i in range(self.truck_num))

    def _generate_nodes(self, distribution="uniform"):
        """generate the nodes in the space
        :return: the coordinates of the nodes
        """
        coordinates = set()
        while len(coordinates) < self.num_customer:
            # normal distribution round to integer
            if distribution == "normal":
                batch = self.RNG.normal(self.means, self.std_dev,
                                        size=(self.num_customer - len(coordinates), 2)).astype(int)
            elif distribution == "uniform":
                batch = self.RNG.uniform(-self.space_width, self.space_width,
                                         size=(self.num_customer - len(coordinates), 2)).astype(int)
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
            # turn into np array
            batch = np.array(batch)
            # filter out the nodes outside the space and the warehouse using 'where' method
            batch_filtered = batch[(-self.space_width <= batch[:, 0])
                                   & (batch[:, 0] <= self.space_width)
                                   & ~((batch[:, 0] == self.warehouse[0]) & (batch[:, 1] == self.warehouse[1]))
                                   & (-self.space_height <= batch[:, 1])
                                   & (batch[:, 1] <= self.space_height)]
            coordinates.update(tuple(row) for row in batch_filtered)

        return np.array(list(coordinates))

    def _generate_parcel_weight(self):
        """generate the weights of the nodes
        :return: the weights of the nodes
        """
        return self.RNG.integers(1, self.uav_capacity // 2, self.num_customer)

    def _k_means_cluster(self):
        """k-means cluster the nodes
        """
        self.kmeans = KMeans(n_clusters=self.cluster_number, random_state=0)
        self.kmeans.fit(self.nodes_location)

        elements = []
        for cluster in range(self.kmeans.n_clusters):
            cluster_elements = self.nodes_location[self.kmeans.labels_ == cluster]
            elements.append(np.array(cluster_elements))
        self.infos["cluster_elements"] = copy(elements)
        self.infos["cluster_centers"] = copy(self.kmeans.cluster_centers_)

        # TODO: decide how to decide a node for trucks

    def _get_k_means_cluster(self, coordinates):
        """get the cluster number of the coordinates

        :param coordinates: the coordinates of the node
        :return: the cluster number of the coordinates
        """
        if self.kmeans is None:
            return None
        new_coordinate = np.array([coordinates])
        return self.kmeans.predict(new_coordinate)[0]

    def _is_cluster_delivered(self, cluster):
        """check if the cluster is delivered

        :param cluster: the cluster number
        :return: if the cluster is delivered
        """
        if cluster is None:
            return False
        delivered_or_assigned = np.logical_or(
            self.node_mask[self.kmeans.labels_ == cluster] == PackageState.DELIVERED.value,
            self.node_mask[self.kmeans.labels_ == cluster] == PackageState.ASSIGNED.value
        )
        return np.all(delivered_or_assigned)

    @staticmethod
    def _get_agent_group(agent):
        """get the group number of the agent

        :param agent: the name of the agent
        :return: the group number and agent number of the agent in that group
        """
        return map(int, re.findall(r"\d+", agent))

    def _load_material(self):
        """load the material from the library
        """
        materials = [
            "UAV.png",
            "Truck.png",
            "Warehouse.png",
            "Customer.png",
            "CustomerClosed.png",
            "CustomerAssigned.png",
        ]

        cur_file_path = os.path.abspath(__file__)
        cur_dir = os.path.dirname(cur_file_path)

        for material in materials:
            material_key = material.split('.')[0]
            if material_key in self.material_library:
                continue
            material_path = os.path.join(cur_dir, "img", material)
            img = pygame.image.load(material_path)
            self.material_library[material_key] = pygame.transform.scale(img, (self.element_size, self.element_size))

        print(self.material_library)

    def _draw_grid(self):
        for x in range(0, self.screen_width, self.element_size):
            pygame.draw.line(self.screen, ColorFamily.BLACK, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.element_size):
            pygame.draw.line(self.screen, ColorFamily.BLACK, (0, y), (self.screen_width, y))

    def _draw_point(self, x, y, material):
        self.screen.blit(self.material_library[material],
                         ((x + self.space_width) * self.element_size, (y + self.space_height) * self.element_size))
