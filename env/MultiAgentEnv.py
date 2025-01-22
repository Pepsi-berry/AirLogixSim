import functools
from copy import copy
from enum import Enum
import re

import numpy as np
from gymnasium.spaces import Discrete, Dict, MultiDiscrete, MultiBinary, Box, Tuple
from pettingzoo import ParallelEnv
from gymnasium.utils import seeding

import pygame


class UAVState(Enum):
    IDLE = 0
    DELIVERING = 1
    LANDING = 2
    RETURNING = 3


class TruckState(Enum):
    MOVING = 0
    LANDING = 1


class DeliveryEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "delivery_env_v1",
    }

    def __init__(self, config: dict):
        """initialize the environment.

        Attributes of config:
            - uav_num: the number of uavs in one group
            - truck_num: the number of trucks in one group
            - group_num: the number of groups
            - uav_velocity: the velocity of uavs
            - truck_velocity: the velocity of trucks
            - uav_capacity: the capacity of uavs
            - uav_power: the power of uavs
            - max_step: the maximum number of steps
            - num_customer: the number of customers
            - render_mode: the mode of rendering
            - space_width: half of the width of the space (2 * space_width + 1 = width of the space)
            - space_height: half of the height of the space (2 * space_height + 1 = height of the space)

        :param config: a dictionary containing the configuration of the environment
        """
        # random number generator
        self.RNG, _ = seeding.np_random()

        # visualization
        self.screen = None

        # hyperparameters
        self.uav_num = config.get("uav_num", 2)
        self.truck_num = config.get("truck_num", 1)
        self.group_num = config.get("group_num", 1)
        self.uav_velocity = config.get("uav_velocity", [12, 29])
        self.truck_velocity = config.get("truck_velocity", 7)
        self.uav_capacity = config.get("uav_capacity", 10)
        self.uav_power = config.get("uav_power", 100)
        self.max_step = config.get("max_step", 10_000)
        self.num_customer = config.get("num_customer", 20)
        self.render_mode = config.get("render_mode", "human")
        self.space_width = config.get("space_width", 10)
        self.space_height = config.get("space_height", 10)

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
        self.agent_coordinates = None
        self.agent_status = None

        self.agent_target = None

        # define the observation space and action space for each agent
        self.observation_spaces = {}
        self.action_spaces = {}
        for agent in self.possible_agents:
            obs = {
                "action_mask": Discrete(2),
                "nodes_mask": MultiBinary(self.num_customer),
                "nodes": Tuple([
                    Box(low=np.array([-self.space_width, -self.space_height]),
                        high=np.array([self.space_width, self.space_height]),
                        dtype=np.float32)
                    ] * self.num_customer),
                "parcel": Tuple([Box(low=0, high=self.uav_capacity, dtype=np.float16)] * self.num_customer),
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
                obs["coordinate"] = Box(
                    low=np.array([-self.space_width, -self.space_height]),
                    high=np.array([self.space_width, self.space_height]),
                    dtype=np.float32
                )
                obs["power"] = Box(0, high=self.uav_power, dtype=np.float32)
                obs["capacity"] = Box(0, high=self.uav_capacity, dtype=np.float32)
                obs["travel_distance"] = Box(0, high=self.space_width * self.space_height, dtype=np.float32),
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
        self.node_mask = np.zeros(self.num_customer)
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
        self.agent_status = {agent: UAVState.IDLE.value if agent.startswith("uav") else TruckState.LANDING.value for agent in self.possible_agents}

        self.agent_target = {agent: None for agent in self.possible_agents}

        observations = self._get_observations()
        return observations, {}

    def step(self, action):
        pass

    def render(self):
        if self.render_mode == "human":
            pass
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
                "nodes_mask": self.node_mask,
                "nodes": self.nodes_location,
                "parcel": self.nodes_weight
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

                obs["coordinate"] = self.agent_coordinates[agent]
                obs["power"] = self.cur_uav_power[agent]
                obs["capacity"] = self.cur_uav_capacity[agent]
                obs["travel_distance"] = self.cur_uav_travel_distance[agent]
                observations[agent] = obs
        return observations

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
            return status == TruckState.LANDING.value and all(self.agent_status[f"uav_{group_num}_{i}"] == UAVState.IDLE.value for i in range(self.uav_num))
        else:
            return status == UAVState.IDLE.value and all(self.agent_status[f"truck_{group_num}_{i}"] == TruckState.LANDING.value for i in range(self.truck_num))

    def _generate_nodes(self):
        """generate the nodes in the space
        :return: the coordinates of the nodes
        """
        coordinates = set()
        while len(coordinates) < self.num_customer:
            # normal distribution round to integer
            batch = self.RNG.normal(self.means, self.std_dev, size=(self.num_customer - len(coordinates), 2)).astype(int)
            batch_filtered = batch[(-self.space_width <= batch[:, 0] <= self.space_width) & (batch[:] != self.warehouse)
                                   & (-self.space_height <= batch[:, 1] <= self.space_height)]
            coordinates.update(tuple(row) for row in batch_filtered)
        return np.array(list(coordinates))

    def _generate_parcel_weight(self):
        """generate the weights of the nodes
        :return: the weights of the nodes
        """
        return self.RNG.integers(1, self.uav_capacity // 2, self.num_customer)

    def _move_to_target(self, agent):
        """move the agent to the target

        :param agent: the name of the agent
        """
        pass

    @staticmethod
    def _get_agent_group(agent):
        """get the group number of the agent

        :param agent: the name of the agent
        :return: the group number and agent number of the agent in that group
        """
        return map(int, re.findall(r"\d+", agent))

