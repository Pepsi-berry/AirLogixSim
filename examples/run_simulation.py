# -*- coding: utf-8 -*-
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
dir_name = os.path.dirname(__file__)

from airlogixsim.airlogixsim_env import AirLogixSimEnv
import numpy as np
import random
import yaml
import sys
from airlogixsim.utils.enums import UAVActionRet, PackageState

def load_config(path):
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        return config

class MultiGroupTaskAllocator:
    """
    升级的多组车辆无人机任务分配算法
    支持多组协同工作，智能集群分配和组间协调
    """
    
    def __init__(self, env):
        self.env = env
        self.group_cluster_assignments = {}  # 记录每组负责的集群
        self.group_priorities = {}  # 组优先级，用于解决冲突
        self.last_actions = {}  # 记录上次动作，避免重复
        self.cluster_completion_status = {}  # 集群完成状态
        
    def allocate_tasks(self, observation, info):
        """
        主要任务分配函数
        """
        action = {}
        
        # 更新集群完成状态
        self._update_cluster_status(observation, info)
        
        # 为每个组分配任务
        for group_num in range(self.env.group_num):
            group_actions = self._allocate_tasks_for_group(group_num, observation, info)
            action.update(group_actions)
            
        return action
    
    def _update_cluster_status(self, observation, info):
        """更新集群完成状态"""
        if 'cluster_mask' in info:
            for cluster_id, node_indices in info['cluster_mask'].items():
                delivered_count = sum(1 for idx in node_indices 
                                    if self.env.node_mask[idx] == PackageState.DELIVERED.value)  # DELIVERED状态
                self.cluster_completion_status[cluster_id] = {
                    'total_nodes': len(node_indices),
                    'delivered_nodes': delivered_count,
                    'completion_rate': delivered_count / len(node_indices) if len(node_indices) > 0 else 1.0
                }
    
    def _allocate_tasks_for_group(self, group_num, observation, info):
        """为特定组分配任务"""
        group_actions = {}
        
        # 获取该组的车辆和无人机
        group_vehicles = [agent for agent in self.env.possible_agents 
                         if agent.startswith(f'vehicle_{group_num}_')]
        group_uavs = [agent for agent in self.env.possible_agents 
                     if agent.startswith(f'UAV_{group_num}_')]
        
        # 车辆任务分配
        for vehicle_id in group_vehicles:
            vehicle_action = self._assign_vehicle_task(vehicle_id, group_num, observation, info)
            if vehicle_action is not None:
                group_actions[vehicle_id] = vehicle_action
        
        # 无人机任务分配
        for uav_id in group_uavs:
            uav_action = self._assign_uav_task(uav_id, group_num, observation, info)
            if uav_action is not None:
                group_actions[uav_id] = uav_action
        
        return group_actions
    
    def _assign_vehicle_task(self, vehicle_id, group_num, observation, info):
        """为车辆分配任务"""
        if observation[vehicle_id]['action_mask'] != 1:
            return None
            
        # 获取集群信息
        if 'center_node' not in info or 'cluster_mask' not in info:
            return None
            
        # 选择最优集群
        best_cluster = self._select_best_cluster_for_group(group_num, info)
        
        if best_cluster is not None:
            # 记录组的集群分配
            self.group_cluster_assignments[group_num] = best_cluster
            return info['center_node'][best_cluster]
        else:
            # 所有集群都完成，返回仓库
            return int(info['num_customer'])
    
    def _assign_uav_task(self, uav_id, group_num, observation, info):
        """为无人机分配任务"""
        if observation[uav_id]['action_mask'] != 1:
            return None
            
        # 获取该组分配的集群
        assigned_cluster = self.group_cluster_assignments.get(group_num)
        if assigned_cluster is None:
            return None
            
        # 智能任务选择策略
        return self._smart_uav_task_selection(uav_id, group_num, assigned_cluster, observation, info)
    
    def _smart_uav_task_selection(self, uav_id, group_num, assigned_cluster, observation, info):
        """智能无人机任务选择"""
        num_tasks = self.env.task_manager.get_num_of_tasks()
        choice_mask = observation[uav_id]['choice_mask']
        
        # 策略1: 优先返回车辆（如果电量不足或容量已满）
        uav_power = observation[uav_id]['power']
        uav_capacity = observation[uav_id]['capacity']
        
        # 检查是否需要返回车辆
        if (uav_power < self.env.uav_power * 0.3 or uav_capacity <= 0):
            for index in range(num_tasks, len(choice_mask)):
                if choice_mask[index] == UAVActionRet.FEASIBLE.value:
                    return index
        
        # 策略2: 选择配送任务
        if 'cluster_mask' in info and assigned_cluster in info['cluster_mask']:
            cluster_nodes = info['cluster_mask'][assigned_cluster]
            
            # 优先级排序：距离近、重量轻、未被其他无人机选择
            feasible_tasks = []
            for node_idx in cluster_nodes:
                if (node_idx < num_tasks and 
                    choice_mask[node_idx] == UAVActionRet.FEASIBLE.value and
                    self.env.node_mask[node_idx] == PackageState.WAITING.value):  # WAITING状态
                    
                    # 计算优先级分数
                    distance = self._calculate_distance(uav_id, node_idx, observation, info)
                    weight = self.env.nodes_weight[node_idx]
                    priority_score = self._calculate_task_priority(distance, weight, node_idx)
                    
                    feasible_tasks.append((node_idx, priority_score))
            
            # 按优先级排序并选择最佳任务
            if feasible_tasks:
                feasible_tasks.sort(key=lambda x: x[1], reverse=True)
                return feasible_tasks[0][0]
        
        # 策略3: 如果没有合适的配送任务，返回车辆
        for index in range(num_tasks, len(choice_mask)):
            if choice_mask[index] == UAVActionRet.FEASIBLE.value:
                return index
        
        return None
    
    def _select_best_cluster_for_group(self, group_num, info):
        """为组选择最佳集群"""
        if 'cluster_mask' not in info or 'cluster_centers' not in info:
            return None
            
        # 获取未分配的集群
        available_clusters = []
        for cluster_id in info['cluster_mask'].keys():
            if not self._is_cluster_assigned_to_other_group(cluster_id, group_num):
                completion_rate = self.cluster_completion_status.get(cluster_id, {}).get('completion_rate', 0)
                if completion_rate < 1.0:  # 未完成的集群
                    available_clusters.append(cluster_id)
        
        if not available_clusters:
            return None
            
        # 选择策略：优先选择节点数量适中、距离合理的集群
        best_cluster = None
        best_score = -1
        
        for cluster_id in available_clusters:
            cluster_nodes = info['cluster_mask'][cluster_id]
            cluster_center = info['cluster_centers'][cluster_id]
            
            # 计算集群评分
            node_count_score = min(len(cluster_nodes) / 5.0, 1.0)  # 节点数量评分
            distance_score = 1.0 / (1.0 + np.linalg.norm(cluster_center))  # 距离评分
            urgency_score = 1.0 - self.cluster_completion_status.get(cluster_id, {}).get('completion_rate', 0)
            
            total_score = node_count_score * 0.4 + distance_score * 0.3 + urgency_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_cluster = cluster_id
                
        return best_cluster
    
    def _is_cluster_assigned_to_other_group(self, cluster_id, current_group):
        """检查集群是否已被其他组分配"""
        for group, assigned_cluster in self.group_cluster_assignments.items():
            if group != current_group and assigned_cluster == cluster_id:
                return True
        return False
    
    def _calculate_distance(self, uav_id, node_idx, observation, info):
        """计算无人机到节点的距离"""
        uav_pos = observation[uav_id]['coordinate']
        node_pos = self.env.nodes_location[node_idx]
        return np.linalg.norm(np.array(uav_pos) - np.array(node_pos))
    
    def _calculate_task_priority(self, distance, weight, node_idx):
        """计算任务优先级"""
        distance_score = 1.0 / (1.0 + distance)  # 距离越近分数越高
        weight_score = 1.0 / (1.0 + weight)      # 重量越轻分数越高
        return distance_score * 0.6 + weight_score * 0.4

# 1. Load the configuration file
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = load_config(config_path)

# 2. Create the environment
# env = AirLogixSimEnv(config, interactive_mode='graphic')
env = AirLogixSimEnv(config, interactive_mode=None)

observation, info = env.reset()

print(info)
print(observation)

# 创建多组任务分配器
task_allocator = MultiGroupTaskAllocator(env)

running = True
step_count = 0
accumulated_simulation_time = 0

while running:
    action = {}
    print(f"Time Step: {info['cur_time_step']}")
    
    # 显示所有智能体的行动掩码
    for agent in env.possible_agents:
        print(f"{agent}: action_mask={observation[agent]['action_mask']}")

    # 使用升级的多组任务分配算法
    action = task_allocator.allocate_tasks(observation, info)
    
    print(f"Assigned Actions: {action}")
    
    step_start_time = time.time()
    observation, reward, termination, truncation, info = env.step(action, training=True)
    step_end_time = time.time()
    step_duration = step_end_time - step_start_time
    accumulated_simulation_time += step_duration
    step_count += 1

    running = not truncation and any([not t for t in termination.values()])

    print("Actions:", action)
    print("Rewards:", reward)
    
    # 显示无人机状态信息
    uav_status = env.getAllUAVsStatus()
    print("UAV Status:")
    for uav_id, status in uav_status.items():
        print(f"  {uav_id}:")
        print(f"    Capacity: {status['capacity']:.1f}")
        print(f"    Power: {status['power']:.1f} ({status['battery_percentage']:.1f}%)")
        print(f"    Travel Distance: {status['travel_distance']:.1f}")
        print(f"    Is Docked: {status['is_docked']}")
        print(f"    Position: ({status['position'][0]:.1f}, {status['position'][1]:.1f})")
    
    # 显示车辆状态信息
    vehicle_status = env.getAllVehiclesStatus()
    print("Vehicle Status:")
    for vehicle_id, status in vehicle_status.items():
        print(f"  {vehicle_id}:")
        print(f"    Position: ({status['position'][0]:.1f}, {status['position'][1]:.1f})")
        print(f"    Speed: {status['speed']:.1f}")
        print(f"    Onboard UAVs: {status['onboard_uavs']}")
    
    # 显示组分配状态
    print("Group Cluster Assignment Status:")
    for group_num, cluster_id in task_allocator.group_cluster_assignments.items():
        print(f"  Group {group_num}: Cluster {cluster_id}")
    
    print("Termination:", termination)
    print(f"Schedule Step {step_count} completed in {step_duration:.2f} seconds")
    print(f"Total simulation time: {accumulated_simulation_time:.2f} seconds")
    print(f"Average step time: {accumulated_simulation_time / (info['cur_time_step'] / 0.1):.2f} seconds")
    print(f"Average FPS: {1 / (accumulated_simulation_time / (info['cur_time_step'] / 0.1)):.2f}")
    print("==" * 20)

    env.render()

print("==" * 20)
print(observation)
print("==" * 20)
input()

env.close()

print('Simulation completed!')