# -*- coding: utf-8 -*-
"""
Multi-Group UAV-Vehicle Task Allocation Simulation
Safe version with proper encoding handling
"""

import sys
import os
import time
import locale

# Set proper encoding for Windows
if sys.platform.startswith('win'):
    # Try to set UTF-8 encoding on Windows
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        # Fallback to system default
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airlogixsim.airlogixsim_env import AirLogixSimEnv
import numpy as np
import random
import yaml
from airlogixsim.utils.enums import UAVActionRet, PackageState

def load_config(path):
    """Load configuration with proper encoding handling"""
    try:
        with open(path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config
    except UnicodeDecodeError:
        # Fallback to system default encoding
        with open(path, 'r', encoding=locale.getpreferredencoding()) as file:
            config = yaml.safe_load(file)
            return config

class MultiGroupTaskAllocator:
    """
    Advanced multi-group UAV-vehicle task allocation algorithm
    Supports multi-group coordination, intelligent cluster assignment and inter-group coordination
    """
    
    def __init__(self, env):
        self.env = env
        self.group_cluster_assignments = {}  # Track cluster assignments for each group
        self.group_priorities = {}  # Group priorities for conflict resolution
        self.last_actions = {}  # Track last actions to avoid repetition
        self.cluster_completion_status = {}  # Cluster completion status
        
    def allocate_tasks(self, observation, info):
        """Main task allocation function"""
        action = {}
        
        # Update cluster completion status
        self._update_cluster_status(observation, info)
        
        # Allocate tasks for each group
        for group_num in range(self.env.group_num):
            group_actions = self._allocate_tasks_for_group(group_num, observation, info)
            action.update(group_actions)
            
        return action
    
    def _update_cluster_status(self, observation, info):
        """Update cluster completion status"""
        if 'cluster_mask' in info:
            for cluster_id, node_indices in info['cluster_mask'].items():
                delivered_count = sum(1 for idx in node_indices 
                                    if self.env.node_mask[idx] == PackageState.DELIVERED.value)
                self.cluster_completion_status[cluster_id] = {
                    'total_nodes': len(node_indices),
                    'delivered_nodes': delivered_count,
                    'completion_rate': delivered_count / len(node_indices) if len(node_indices) > 0 else 1.0
                }
    
    def _allocate_tasks_for_group(self, group_num, observation, info):
        """Allocate tasks for specific group"""
        group_actions = {}
        
        # Get vehicles and UAVs for this group
        group_vehicles = [agent for agent in self.env.possible_agents 
                         if agent.startswith(f'vehicle_{group_num}_')]
        group_uavs = [agent for agent in self.env.possible_agents 
                     if agent.startswith(f'UAV_{group_num}_')]
        
        # Vehicle task assignment
        for vehicle_id in group_vehicles:
            vehicle_action = self._assign_vehicle_task(vehicle_id, group_num, observation, info)
            if vehicle_action is not None:
                group_actions[vehicle_id] = vehicle_action
        
        # UAV task assignment
        for uav_id in group_uavs:
            uav_action = self._assign_uav_task(uav_id, group_num, observation, info)
            if uav_action is not None:
                group_actions[uav_id] = uav_action
        
        return group_actions
    
    def _assign_vehicle_task(self, vehicle_id, group_num, observation, info):
        """Assign task to vehicle"""
        if observation[vehicle_id]['action_mask'] != 1:
            return None
            
        # Get cluster information
        if 'center_node' not in info or 'cluster_mask' not in info:
            return None
            
        # Select optimal cluster
        best_cluster = self._select_best_cluster_for_group(group_num, info)
        
        if best_cluster is not None:
            # Record group's cluster assignment
            self.group_cluster_assignments[group_num] = best_cluster
            return info['center_node'][best_cluster]
        else:
            # All clusters completed, return to warehouse
            return int(info['num_customer'])
    
    def _assign_uav_task(self, uav_id, group_num, observation, info):
        """Assign task to UAV"""
        if observation[uav_id]['action_mask'] != 1:
            return None
            
        # Get assigned cluster for this group
        assigned_cluster = self.group_cluster_assignments.get(group_num)
        if assigned_cluster is None:
            return None
            
        # Smart task selection strategy
        return self._smart_uav_task_selection(uav_id, group_num, assigned_cluster, observation, info)
    
    def _smart_uav_task_selection(self, uav_id, group_num, assigned_cluster, observation, info):
        """Smart UAV task selection"""
        num_tasks = self.env.task_manager.get_num_of_tasks()
        choice_mask = observation[uav_id]['choice_mask']
        
        # Strategy 1: Return to vehicle if low battery or full capacity
        uav_power = observation[uav_id]['power']
        uav_capacity = observation[uav_id]['capacity']
        
        # Check if need to return to vehicle
        if (uav_power < self.env.uav_power * 0.3 or uav_capacity <= 0):
            for index in range(num_tasks, len(choice_mask)):
                if choice_mask[index] == UAVActionRet.FEASIBLE.value:
                    return index
        
        # Strategy 2: Select delivery tasks
        if 'cluster_mask' in info and assigned_cluster in info['cluster_mask']:
            cluster_nodes = info['cluster_mask'][assigned_cluster]
            
            # Priority ranking: close distance, light weight, not selected by other UAVs
            feasible_tasks = []
            for node_idx in cluster_nodes:
                if (node_idx < num_tasks and 
                    choice_mask[node_idx] == UAVActionRet.FEASIBLE.value and
                    self.env.node_mask[node_idx] == PackageState.WAITING.value):
                    
                    # Calculate priority score
                    distance = self._calculate_distance(uav_id, node_idx, observation, info)
                    weight = self.env.nodes_weight[node_idx]
                    priority_score = self._calculate_task_priority(distance, weight, node_idx)
                    
                    feasible_tasks.append((node_idx, priority_score))
            
            # Sort by priority and select best task
            if feasible_tasks:
                feasible_tasks.sort(key=lambda x: x[1], reverse=True)
                return feasible_tasks[0][0]
        
        # Strategy 3: If no suitable delivery tasks, return to vehicle
        for index in range(num_tasks, len(choice_mask)):
            if choice_mask[index] == UAVActionRet.FEASIBLE.value:
                return index
        
        return None
    
    def _select_best_cluster_for_group(self, group_num, info):
        """Select best cluster for group"""
        if 'cluster_mask' not in info or 'cluster_centers' not in info:
            return None
            
        # Get unassigned clusters
        available_clusters = []
        for cluster_id in info['cluster_mask'].keys():
            if not self._is_cluster_assigned_to_other_group(cluster_id, group_num):
                completion_rate = self.cluster_completion_status.get(cluster_id, {}).get('completion_rate', 0)
                if completion_rate < 1.0:  # Uncompleted clusters
                    available_clusters.append(cluster_id)
        
        if not available_clusters:
            return None
            
        # Selection strategy: prioritize moderate node count and reasonable distance
        best_cluster = None
        best_score = -1
        
        for cluster_id in available_clusters:
            cluster_nodes = info['cluster_mask'][cluster_id]
            cluster_center = info['cluster_centers'][cluster_id]
            
            # Calculate cluster score
            node_count_score = min(len(cluster_nodes) / 5.0, 1.0)  # Node count score
            distance_score = 1.0 / (1.0 + np.linalg.norm(cluster_center))  # Distance score
            urgency_score = 1.0 - self.cluster_completion_status.get(cluster_id, {}).get('completion_rate', 0)
            
            total_score = node_count_score * 0.4 + distance_score * 0.3 + urgency_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_cluster = cluster_id
                
        return best_cluster
    
    def _is_cluster_assigned_to_other_group(self, cluster_id, current_group):
        """Check if cluster is assigned to other groups"""
        for group, assigned_cluster in self.group_cluster_assignments.items():
            if group != current_group and assigned_cluster == cluster_id:
                return True
        return False
    
    def _calculate_distance(self, uav_id, node_idx, observation, info):
        """Calculate distance from UAV to node"""
        uav_pos = observation[uav_id]['coordinate']
        node_pos = self.env.nodes_location[node_idx]
        return np.linalg.norm(np.array(uav_pos) - np.array(node_pos))
    
    def _calculate_task_priority(self, distance, weight, node_idx):
        """Calculate task priority"""
        distance_score = 1.0 / (1.0 + distance)  # Closer distance = higher score
        weight_score = 1.0 / (1.0 + weight)      # Lighter weight = higher score
        return distance_score * 0.6 + weight_score * 0.4

def main():
    """Main simulation function"""
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        config = load_config(config_path)
        
        print("Configuration loaded successfully")
        print(f"Groups: {config['traffic']['group_num']}")
        print(f"Vehicles per group: {config['traffic']['vehicle_count']}")
        print(f"UAVs per group: {config['traffic']['UAV_count']}")
        
        # Create environment
        env = AirLogixSimEnv(config, interactive_mode=None)
        observation, info = env.reset()
        
        print("Environment initialized successfully")
        print(f"Total agents: {len(env.possible_agents)}")
        
        # Create multi-group task allocator
        task_allocator = MultiGroupTaskAllocator(env)
        
        running = True
        step_count = 0
        accumulated_simulation_time = 0
        
        while running:
            action = {}
            print(f"\n--- Time Step: {info['cur_time_step']} ---")
            
            # Show action masks for all agents
            active_agents = [agent for agent in env.possible_agents 
                           if observation[agent]['action_mask'] == 1]
            print(f"Active agents: {len(active_agents)}")
            
            # Use upgraded multi-group task allocation algorithm
            action = task_allocator.allocate_tasks(observation, info)
            
            if action:
                print(f"Assigned Actions: {action}")
            else:
                print("No actions assigned this step")
            
            step_start_time = time.time()
            observation, reward, termination, truncation, info = env.step(action, training=True)
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            accumulated_simulation_time += step_duration
            step_count += 1
            
            running = not truncation and any([not t for t in termination.values()])
            
            # Show UAV status
            uav_status = env.getAllUAVsStatus()
            if uav_status:
                print(f"UAV Status Summary: {len(uav_status)} UAVs")
                for uav_id, status in list(uav_status.items())[:2]:  # Show first 2 for brevity
                    print(f"  {uav_id}: Power {status['power']:.1f}, Capacity {status['capacity']:.1f}")
            
            # Show group assignments
            if task_allocator.group_cluster_assignments:
                print("Group Assignments:", 
                      {f"Group{k}": f"Cluster{v}" for k, v in task_allocator.group_cluster_assignments.items()})
            
            print(f"Step {step_count} completed in {step_duration:.3f}s")
            
            # Render environment
            env.render()
            
            # Break if too many steps (safety)
            if step_count > 1000:
                print("Maximum steps reached, stopping simulation")
                break
        
        print("\n" + "="*50)
        print("SIMULATION COMPLETED")
        print(f"Total steps: {step_count}")
        print(f"Total time: {accumulated_simulation_time:.2f} seconds")
        print(f"Average step time: {accumulated_simulation_time/step_count:.3f} seconds")
        print("="*50)
        
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    main() 