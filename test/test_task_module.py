import sys
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import yaml
import time

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airlogixsim.entities.task import Task
from airlogixsim.manager.task_manager import TaskManager

def load_config(config_file="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def test_basic_task_creation(config):
    """Test basic task creation functionality."""
    # Ensure task configuration exists
    if "task" not in config:
        # Add task configuration if not in config file
        config["task"] = {
            "x_range": [0, 1000],
            "y_range": [0, 1000],
            "min_package_weight": 0.5,  # kg
            "max_package_weight": 5.0,  # kg
            "min_delivery_time": 300,  # seconds
            "max_delivery_time": 1800,  # seconds
        }
    
    # Create task manager using the from_config method
    task_manager = TaskManager.from_config(config)
    
    # Create a task with specific parameters
    specific_task = task_manager.create_task(
        destination=(500, 500, 0),
        package_weight=2.5,
        latest_delivery_time=600
    )
    print("Created specific task:")
    print(specific_task)
    
    # Initialize multiple random tasks
    print("\nInitializing 5 random tasks:")
    current_time = 100  # Simulation has been running for 100 seconds
    random_tasks = task_manager.initialize_tasks(5, current_time)
    
    for task_id, task in random_tasks.items():
        print(task)
    
    # Get tasks by status
    pending_tasks = task_manager.get_tasks_by_status("pending")
    print(f"\nNumber of pending tasks: {len(pending_tasks)}")
    
    # Change status of one task
    if len(random_tasks) > 0:
        first_task_id = list(random_tasks.keys())[0]
        first_task = task_manager.get_task(first_task_id)
        first_task.setStatus("assigned")
        print(f"\nChanged status of task {first_task_id} to 'assigned'")
    
    # Get tasks by status again
    pending_tasks = task_manager.get_tasks_by_status("pending")
    assigned_tasks = task_manager.get_tasks_by_status("assigned")
    print(f"Number of pending tasks: {len(pending_tasks)}")
    print(f"Number of assigned tasks: {len(assigned_tasks)}")
    
    return task_manager, random_tasks

def visualize_task_destinations(task_manager):
    """Visualize task destinations on the map."""
    tasks = task_manager.get_all_tasks()
    
    # Extract destination coordinates from all tasks
    destinations = [task.getDestination() for task in tasks.values()]
    print(destinations)
    xs = [dest[0] for dest in destinations]
    print(xs)
    ys = [dest[1] for dest in destinations]
    print(ys)
    weights = [task.getPackageWeight() for task in tasks.values()]
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(xs, ys, c=weights, cmap='viridis', 
                          s=100, alpha=0.7, edgecolors='black')
    plt.colorbar(scatter, label='Package Weight (kg)')
    
    # Set chart title and labels
    plt.title('Task Destination Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Get map boundaries
    x_range = task_manager._x_range
    y_range = task_manager._y_range
    
    # Set axis ranges
    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    
    # Save chart
    plt.savefig('task_destinations.png')
    print("\nSaved task destination distribution chart to test/task_destinations.png")
    
def visualize_delivery_times(task_manager):
    """Visualize latest delivery time distribution."""
    tasks = task_manager.get_all_tasks()
    
    # Extract latest delivery times from all tasks
    delivery_times = [task.getLatestDeliveryTime() for task in tasks.values()]
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(delivery_times, bins=10, color='skyblue', edgecolor='black')
    
    # Set chart title and labels
    plt.title('Latest Delivery Time Distribution')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Tasks')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save chart
    plt.savefig('delivery_times.png')
    print("Saved latest delivery time distribution chart to test/delivery_times.png")

def test_large_scale_task_generation(config):
    """Test large-scale task generation."""
    # Create a copy of the configuration to avoid modifying the original
    large_scale_config = config.copy()
    
    # Set task parameters for large-scale test
    if "task" not in large_scale_config:
        large_scale_config["task"] = {}
        
    large_scale_config["task"]["x_range"] = [0, 2000]
    large_scale_config["task"]["y_range"] = [0, 2000]
    large_scale_config["task"]["min_package_weight"] = 0.1
    large_scale_config["task"]["max_package_weight"] = 10.0
    large_scale_config["task"]["min_delivery_time"] = 300
    large_scale_config["task"]["max_delivery_time"] = 3600
    
    # Create task manager
    task_manager = TaskManager.from_config(large_scale_config)
    
    # Generate a large number of tasks
    num_tasks = 50
    print(f"\nGenerating {num_tasks} tasks:")
    
    # Record start time to measure performance
    start_time = time.time()
    
    tasks = task_manager.initialize_tasks(num_tasks)
    
    end_time = time.time()
    print(f"Generated {num_tasks} tasks in: {end_time - start_time:.4f} seconds")
    
    # Analyze task properties
    weights = [task.getPackageWeight() for task in tasks.values()]
    times = [task.getLatestDeliveryTime() for task in tasks.values()]
    
    print(f"Package weight statistics: min={min(weights):.2f}kg, max={max(weights):.2f}kg, avg={sum(weights)/len(weights):.2f}kg")
    print(f"Latest delivery time statistics: min={min(times):.2f}s, max={max(times):.2f}s, avg={sum(times)/len(times):.2f}s")
    
    return task_manager

def main():
    """Main test function."""
    # Load configuration
    config = load_config()
    
    print("===== Testing Basic Task Creation =====")
    task_manager, tasks = test_basic_task_creation(config)
    
    print("\n===== Visualizing Task Destinations =====")
    visualize_task_destinations(task_manager)
    
    print("\n===== Visualizing Latest Delivery Times =====")
    visualize_delivery_times(task_manager)
    
    print("\n===== Testing Large-Scale Task Generation =====")
    large_scale_task_manager = test_large_scale_task_generation(config)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 