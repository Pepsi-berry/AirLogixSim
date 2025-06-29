import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import argparse

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airlogixsim.airlogixsim_env import AirLogixSimEnv

def load_config(config_path):
    """Load the configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def test_initialization(env):
    """Test if the environment initializes correctly."""
    print("\n=== Testing Environment Initialization ===")
    print(f"Number of UAVs: {env.traffic_manager.getNumberOfUAVs()}")
    print(f"Number of vehicles: {env.traffic_manager.getNumberOfVehicles()}")
    return True

def test_uav_complete_workflow(env):
    """Test the complete UAV workflow: takeoff, flying, hovering, and resuming."""
    print("\n=== Testing Complete UAV Workflow ===")
    
    # Find a docked UAV for testing
    uav_infos = env.traffic_manager.getUAVTrafficInfos()
    docked_uavs = []
    
    for uav_id, uav_info in uav_infos.items():
        if uav_info.get("is_docked", False):
            docked_uavs.append(uav_id)
    
    if not docked_uavs:
        print("No docked UAVs available for testing. Checking if we can dock any...")
        # Try to dock a flying UAV to a vehicle
        flying_uavs = []
        for uav_id, uav_info in uav_infos.items():
            if not uav_info.get("is_docked", False):
                flying_uavs.append(uav_id)
        
        if flying_uavs and env.traffic_manager.getNumberOfVehicles() > 0:
            vehicle_ids = list(env.traffic_manager.getVehicleTrafficInfos().keys())
            if vehicle_ids:
                dock_success = env.traffic_manager.dockUAV(flying_uavs[0], vehicle_ids[0])
                if dock_success:
                    print(f"Successfully docked UAV {flying_uavs[0]} to vehicle {vehicle_ids[0]}")
                    docked_uavs.append(flying_uavs[0])
    
    if not docked_uavs:
        print("Failed to find or create a docked UAV. Cannot proceed with test.")
        return False
    
    # Select the first docked UAV for testing
    test_uav_id = docked_uavs[0]
    print(f"Selected docked UAV {test_uav_id} for testing")
    
    # Get the vehicle it's docked to
    vehicle_id = env.traffic_manager.getDockedVehicleId(test_uav_id)
    print(f"UAV is docked to vehicle {vehicle_id}")
    
    # Data collection for visualization
    timestamps = []
    positions = []
    speeds = []
    hovering_states = []
    current_time = 0
    
    # 1. Take off from vehicle and fly to destination
    print("\n1. TESTING TAKEOFF FROM VEHICLE")
    
    # Set initial target destination (random position in the map)
    x_range = env.traffic_manager._x_range
    y_range = env.traffic_manager._y_range
    z_range = env.traffic_manager._UAV_z_range
    
    # Get current position
    current_pos = env.traffic_manager.getUAVTrafficInfos()[test_uav_id]["position"]
    
    # Set destination some distance away from current position (not too far)
    dest_x = current_pos[0] + np.random.uniform(-200, 200)
    dest_y = current_pos[1] + np.random.uniform(-200, 200)
    dest_z = np.random.uniform(z_range[0], z_range[1])
    
    # Ensure destination is within map bounds
    dest_x = np.clip(dest_x, x_range[0], x_range[1])
    dest_y = np.clip(dest_y, y_range[0], y_range[1])
    
    destination = (dest_x, dest_y, dest_z)
    print(f"Setting destination for flight: {destination}")
    
    # Take off with initial speed
    initial_speed = 20.0
    takeoff_success = env.traffic_manager.takeOffUAV(test_uav_id, initial_speed=initial_speed)
    print(f"Takeoff {'successful' if takeoff_success else 'failed'}")
    
    if not takeoff_success:
        return False
    
    # Collect initial data point
    uav_info = env.traffic_manager.getUAVTrafficInfos()[test_uav_id]
    timestamps.append(current_time)
    positions.append(uav_info["position"])
    speeds.append(uav_info["speed"])
    hovering_states.append(False)
    
    # Set destination
    env.traffic_manager.setUAVDestination(test_uav_id, destination)
    print(f"Flying to destination {destination}")
    
    # 2. Fly for a while and collect data
    print("\n2. FLYING TOWARD DESTINATION")
    flight_duration = 30  # steps
    
    for i in range(flight_duration):
        env.step()
        current_time += 1
        
        # Collect data
        uav_info = env.traffic_manager.getUAVTrafficInfos()[test_uav_id]
        timestamps.append(current_time)
        positions.append(uav_info["position"])
        speeds.append(uav_info["speed"])
        hovering_states.append(False)
        
        # Print progress every few steps
        if i % 5 == 0:
            current_pos = uav_info["position"]
            current_speed = uav_info["speed"]
            
            # Calculate distance to destination
            dx = destination[0] - current_pos[0]
            dy = destination[1] - current_pos[1]
            dz = destination[2] - current_pos[2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            print(f"Step {i+1}: Position={current_pos}, Speed={current_speed}, Distance to destination={distance:.2f}")
        
        time.sleep(0.05)  # Shorter sleep time for faster testing
    
    # 3. Hover
    print("\n3. TESTING HOVER")
    hover_success = env.traffic_manager.hoverUAV(test_uav_id)
    print(f"Hover command {'successful' if hover_success else 'failed'}")
    
    if not hover_success:
        return False
    
    # Verify hovering state
    is_hovering = env.traffic_manager.isUAVHovering(test_uav_id)
    print(f"UAV hovering status: {is_hovering}")
    
    # Run simulation while hovering
    hover_duration = 20  # steps
    
    for i in range(hover_duration):
        env.step()
        current_time += 1
        
        # Collect data
        uav_info = env.traffic_manager.getUAVTrafficInfos()[test_uav_id]
        timestamps.append(current_time)
        positions.append(uav_info["position"])
        speeds.append(uav_info["speed"])
        hovering_states.append(True)
        
        # Print progress occasionally
        if i % 5 == 0:
            current_pos = uav_info["position"]
            current_speed = uav_info["speed"]
            print(f"Hover step {i+1}: Position={current_pos}, Speed={current_speed}")
        
        time.sleep(0.05)
    
    # Get hover position for distance calculation
    hover_position = env.traffic_manager.getUAVTrafficInfos()[test_uav_id]["position"]
    
    # 4. Resume flight
    print("\n4. TESTING RESUME")
    resume_success = env.traffic_manager.resumeUAV(test_uav_id)
    print(f"Resume command {'successful' if resume_success else 'failed'}")
    
    if not resume_success:
        return False
    
    # Set a new destination
    new_dest_x = hover_position[0] + np.random.uniform(-200, 200)
    new_dest_y = hover_position[1] + np.random.uniform(-200, 200)
    new_dest_z = np.random.uniform(z_range[0], z_range[1])
    
    # Ensure destination is within map bounds
    new_dest_x = np.clip(new_dest_x, x_range[0], x_range[1])
    new_dest_y = np.clip(new_dest_y, y_range[0], y_range[1])
    
    new_destination = (new_dest_x, new_dest_y, new_dest_z)
    print(f"Setting new destination after resume: {new_destination}")
    
    dest_set = env.traffic_manager.setUAVDestination(test_uav_id, new_destination)
    print(f"New destination set: {dest_set}")
    
    # Run simulation after resuming
    resume_duration = 30  # steps
    
    for i in range(resume_duration):
        env.step()
        current_time += 1
        
        # Collect data
        uav_info = env.traffic_manager.getUAVTrafficInfos()[test_uav_id]
        timestamps.append(current_time)
        positions.append(uav_info["position"])
        speeds.append(uav_info["speed"])
        hovering_states.append(False)
        
        # Print progress occasionally
        if i % 5 == 0:
            current_pos = uav_info["position"]
            current_speed = uav_info["speed"]
            
            # Calculate distance to new destination
            dx = new_destination[0] - current_pos[0]
            dy = new_destination[1] - current_pos[1]
            dz = new_destination[2] - current_pos[2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            print(f"Resume step {i+1}: Position={current_pos}, Speed={current_speed}, Distance to destination={distance:.2f}")
        
        time.sleep(0.05)
    
    # Check if position changed after resuming (compared to hover position)
    last_position = positions[-1]
    dx = last_position[0] - hover_position[0]
    dy = last_position[1] - hover_position[1]
    dz = last_position[2] - hover_position[2]
    distance_moved = np.sqrt(dx**2 + dy**2 + dz**2)
    
    print(f"Distance moved after resuming: {distance_moved:.2f} meters")
    resume_movement_confirmed = distance_moved > 5.0
    print(f"UAV resumed movement: {resume_movement_confirmed}")
    
    # Create visualizations
    create_visualizations(timestamps, positions, speeds, hovering_states)
    
    # Overall test success if we completed all steps and confirmed movement
    return takeoff_success and hover_success and resume_success and resume_movement_confirmed

def create_visualizations(timestamps, positions, speeds, hovering_states):
    """Create and save visualizations of UAV movement."""
    # Convert data to numpy arrays for easier manipulation
    timestamps = np.array(timestamps)
    positions = np.array(positions)
    speeds = np.array(speeds)
    hovering_states = np.array(hovering_states)
    
    # 1. Plot 3D trajectory
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color points based on hovering state
    flying_indices = ~hovering_states
    hovering_indices = hovering_states
    
    # Plot the trajectory with different colors for flying and hovering
    ax.plot(positions[flying_indices, 0], positions[flying_indices, 1], positions[flying_indices, 2], 
            'b-', label='Flying', linewidth=2)
    ax.scatter(positions[hovering_indices, 0], positions[hovering_indices, 1], positions[hovering_indices, 2], 
              c='red', s=30, label='Hovering')
    
    # Add start and end markers
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              c='green', s=100, marker='^', label='Start (Takeoff)')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              c='black', s=100, marker='x', label='End')
    
    # Set labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('UAV 3D Trajectory')
    ax.legend()
    
    # Save figure
    plt.savefig("uav_3d_trajectory.png")
    print("UAV 3D trajectory visualization saved to 'uav_3d_trajectory.png'")
    
    # 2. Plot speed over time
    plt.figure(figsize=(12, 6))
    
    # Determine indices for each phase
    takeoff_end_idx = np.min(np.where(hovering_states)[0]) if np.any(hovering_states) else len(timestamps) // 3
    hover_end_idx = np.max(np.where(hovering_states)[0]) if np.any(hovering_states) else 2 * len(timestamps) // 3
    
    # Plot with phase coloring
    plt.plot(timestamps[:takeoff_end_idx], speeds[:takeoff_end_idx], 'b-', label='Flying (Phase 1)')
    plt.plot(timestamps[takeoff_end_idx:hover_end_idx+1], speeds[takeoff_end_idx:hover_end_idx+1], 'r-', label='Hovering')
    plt.plot(timestamps[hover_end_idx:], speeds[hover_end_idx:], 'g-', label='Flying (Phase 2)')
    
    # Add phase labels
    plt.axvline(x=timestamps[takeoff_end_idx], color='gray', linestyle='--')
    plt.axvline(x=timestamps[hover_end_idx], color='gray', linestyle='--')
    
    plt.text(timestamps[takeoff_end_idx//2], max(speeds)*0.9, "Initial Flight", 
             horizontalalignment='center', fontsize=12)
    plt.text(timestamps[takeoff_end_idx] + (timestamps[hover_end_idx]-timestamps[takeoff_end_idx])/2, 
             max(speeds)*0.9, "Hovering", horizontalalignment='center', fontsize=12)
    plt.text(timestamps[hover_end_idx] + (timestamps[-1]-timestamps[hover_end_idx])/2, 
             max(speeds)*0.9, "Resumed Flight", horizontalalignment='center', fontsize=12)
    
    # Set labels and title
    plt.xlabel('Simulation Step')
    plt.ylabel('UAV Speed (m/s)')
    plt.title('UAV Speed Over Time')
    plt.grid(True)
    plt.legend()
    
    # Save figure
    plt.savefig("uav_speed_over_time.png")
    print("UAV speed chart saved to 'uav_speed_over_time.png'")
    
    # 3. Plot 2D trajectory with colored segments
    plt.figure(figsize=(12, 10))
    
    # Plot each phase with different colors
    plt.plot(positions[:takeoff_end_idx, 0], positions[:takeoff_end_idx, 1], 'b-', 
             linewidth=2, label='Flying (Phase 1)')
    plt.plot(positions[takeoff_end_idx:hover_end_idx+1, 0], positions[takeoff_end_idx:hover_end_idx+1, 1], 
             'r-', linewidth=2, label='Hovering')
    plt.plot(positions[hover_end_idx:, 0], positions[hover_end_idx:, 1], 
             'g-', linewidth=2, label='Flying (Phase 2)')
    
    # Add start and end markers
    plt.scatter(positions[0, 0], positions[0, 1], 
                c='green', s=100, marker='^', label='Start (Takeoff)')
    plt.scatter(positions[-1, 0], positions[-1, 1], 
                c='black', s=100, marker='x', label='End')
    
    # Set labels and title
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('UAV 2D Trajectory (Top View)')
    plt.grid(True)
    plt.legend()
    
    # Make axes equal to preserve shape
    plt.axis('equal')
    
    # Save figure
    plt.savefig("uav_2d_trajectory.png")
    print("UAV 2D trajectory visualization saved to 'uav_2d_trajectory.png'")

def run_all_tests(use_gui=False):
    """Run all UAV hover functionality tests."""
    # Configuration file path
    config_path = "config.yaml"
    
    # Load configuration
    try:
        config = load_config(config_path)
        print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Increase UAV count for better testing
    if 'traffic' in config:
        config['traffic']['UAV_count'] = max(config['traffic'].get('UAV_count', 10), 10)
    
    # Set SUMO binary based on parameters
    if 'sumo' in config:
        config['sumo']['sumo_binary'] = 'sumo-gui' if use_gui else 'sumo'
    
    # Create environment
    try:
        env = AirLogixSimEnv(config)
        print("Environment created successfully")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    
    # Run tests
    tests = {
        "Environment Initialization": test_initialization,
        "UAV Complete Workflow": test_uav_complete_workflow
    }
    
    results = {}
    
    for test_name, test_func in tests.items():
        print(f"\n{'=' * 40}")
        print(f"Running test: {test_name}")
        print(f"{'=' * 40}")
        
        try:
            result = test_func(env)
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            print(f"Test {test_name}: {status}")
        except Exception as e:
            print(f"Error in test {test_name}: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n\n")
    print("=" * 60)
    print("TEST SUMMARY".center(60))
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        if not result:
            all_passed = False
        print(f"{test_name:.<40}{status:.>20}")
    
    print("=" * 60)
    overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(overall.center(60))
    print("=" * 60)
    
    # Clean up
    env.close()
    print("Environment closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test UAV hover functionality with complete workflow.')
    parser.add_argument('--gui', action='store_true', help='Use SUMO-GUI visualization mode')
    args = parser.parse_args()
    
    run_all_tests(use_gui=args.gui) 