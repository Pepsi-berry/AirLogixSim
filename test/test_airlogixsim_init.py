import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import argparse

# Add the parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airlogixsim.airlogixsim_env import AirLogixSimEnv

def load_config(config_path):
    """Load the configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def test_initialization(env):
    """Test if the environment initializes correctly."""
    print("\n=== Testing Environment Initialization ===")
    print(f"Simulation time: {env.simulation_time}")
    print(f"Max simulation time: {env.max_simulation_time}")
    print(f"Traffic interval: {env.traffic_interval}")
    print(f"Simulation interval: {env.simulation_interval}")
    
    # Check if SUMO connection established (if using SUMO mode)
    if env.config['traffic']['traffic_mode'] == 'SUMO':
        print(f"SUMO connection established: {env.traci_connection is not None}")
    
    # Check traffic manager initialization
    print(f"Traffic manager initialized: {env.traffic_manager is not None}")
    print(f"Traffic mode: {env.traffic_manager._traffic_mode}")
    
    # Verify map boundaries
    print(f"X range: {env.traffic_manager._x_range}")
    print(f"Y range: {env.traffic_manager._y_range}")
    
    return env.traffic_manager is not None

def test_vehicle_creation(env):
    """Test if vehicles are created with the correct count."""
    print("\n=== Testing Vehicle Creation ===")
    vehicle_count = env.config['traffic']['vehicle_count']
    actual_count = len(env.vehicles)
    print(f"Expected vehicle count: {vehicle_count}")
    print(f"Actual vehicle count: {actual_count}")
    
    # Check a sample vehicle's properties
    if len(env.vehicles) > 0:
        vehicle_id = list(env.vehicles.keys())[0]
        vehicle = env.vehicles[vehicle_id]
        print(f"\nSample Vehicle (ID: {vehicle_id}):")
        print(f"Position: {vehicle.getPosition()}")
        print(f"Speed: {vehicle.getSpeed()}")
        print(f"Angle: {vehicle.getAngle()}")
    
    return vehicle_count == actual_count

def test_uav_creation(env):
    """Test if UAVs are created with the correct count."""
    print("\n=== Testing UAV Creation ===")
    uav_count = env.config['traffic']['UAV_count']
    actual_count = len(env.UAVs)
    print(f"Expected UAV count: {uav_count}")
    print(f"Actual UAV count: {actual_count}")
    
    # Check a sample UAV's properties
    if len(env.UAVs) > 0:
        uav_id = list(env.UAVs.keys())[0]
        uav = env.UAVs[uav_id]
        print(f"\nSample UAV (ID: {uav_id}):")
        print(f"Position: {uav.getPosition()}")
        print(f"Speed: {uav.getSpeed()}")
        print(f"Is docked: {uav.isDocked()}")
        if uav.isDocked():
            print(f"Docked to vehicle: {uav.getDockedVehicleId()}")
    
    return uav_count == actual_count

def test_uav_docking(env):
    """Test if UAVs are correctly docked to vehicles initially."""
    print("\n=== Testing UAV Docking ===")
    uav_ids = env.getUAVIds()
    docked_count = 0
    
    for uav_id in uav_ids:
        is_docked = env.isUAVDocked(uav_id)
        if is_docked:
            docked_count += 1
            vehicle_id = env.getDockedVehicleId(uav_id)
            print(f"UAV {uav_id} is docked to vehicle {vehicle_id}")
            
            # Check if UAV position matches vehicle position with z-offset
            uav_pos = env.UAVs[uav_id].getPosition()
            vehicle_pos = env.vehicles[vehicle_id].getPosition()
            z_offset = env.traffic_manager._UAV_docking_z_offset
            
            # Check position match with some tolerance for floating point errors
            pos_match = (
                abs(uav_pos[0] - vehicle_pos[0]) < 0.01 and
                abs(uav_pos[1] - vehicle_pos[1]) < 0.01 and
                abs(uav_pos[2] - (vehicle_pos[2] + z_offset)) < 0.01
            )
            
            print(f"UAV position: {uav_pos}")
            print(f"Vehicle position: {vehicle_pos}")
            print(f"Position match: {pos_match}")
            
            # Check UAV speed is 0 when docked
            print(f"UAV speed (should be 0): {env.UAVs[uav_id].getSpeed()}")
    
    print(f"Total UAVs: {len(uav_ids)}")
    print(f"Docked UAVs: {docked_count}")
    
    # All UAVs should be docked at initialization
    return docked_count == len(uav_ids)

def test_uav_takeoff_and_destination(env):
    """Test UAV takeoff functionality and destination setting."""
    print("\n=== Testing UAV Takeoff and Destination Setting ===")
    
    if len(env.UAVs) == 0:
        print("No UAVs to test takeoff")
        return False
    
    # Choose a UAV to test
    uav_id = list(env.UAVs.keys())[0]
    initial_position = env.UAVs[uav_id].getPosition()
    
    print(f"Testing takeoff for UAV {uav_id}")
    print(f"Initial position: {initial_position}")
    print(f"Initially docked: {env.isUAVDocked(uav_id)}")
    
    # Take off the UAV
    takeoff_success = env.takeOffUAV(uav_id, initial_speed=25.0)
    print(f"Takeoff successful: {takeoff_success}")
    print(f"Is still docked after takeoff: {env.isUAVDocked(uav_id)}")
    print(f"Speed after takeoff: {env.UAVs[uav_id].getSpeed()}")
    
    # Set a destination for the UAV
    x_range = env.traffic_manager._x_range
    y_range = env.traffic_manager._y_range
    z_range = env.traffic_manager._UAV_z_range
    
    destination = (
        (x_range[0] + x_range[1]) / 2,  # Middle of x range
        (y_range[0] + y_range[1]) / 2,  # Middle of y range
        z_range[0] + 10  # Low altitude
    )
    
    set_dest_success = env.setUAVDestination(uav_id, destination)
    print(f"Set destination successful: {set_dest_success}")
    print(f"Destination: {destination}")
    
    # Run a few simulation steps to see UAV movement
    print("\nSimulating UAV movement toward destination...")
    initial_distance = np.linalg.norm(np.array(initial_position) - np.array(destination))
    print(f"Initial distance to destination: {initial_distance:.2f} meters")
    
    # Store positions for visualization
    positions = [initial_position]
    
    # Run simulation for a few steps
    for i in range(5):
        env.step()
        current_pos = env.UAVs[uav_id].getPosition()
        positions.append(current_pos)
        current_distance = np.linalg.norm(np.array(current_pos) - np.array(destination))
        print(f"Step {i+1}: Position {current_pos}, Distance to destination: {current_distance:.2f} meters")
    
    # Check if UAV moved toward the destination
    final_pos = positions[-1]
    final_distance = np.linalg.norm(np.array(final_pos) - np.array(destination))
    moving_toward_dest = final_distance < initial_distance
    
    print(f"Moving toward destination: {moving_toward_dest}")
    
    # Visualize the UAV path in 3D
    visualize_uav_path(positions, destination)
    
    return takeoff_success and set_dest_success and moving_toward_dest

def test_task_creation(env):
    """Test if tasks are created with the correct count."""
    print("\n=== Testing Task Creation ===")
    task_count = env.config['traffic']['num_tasks']
    actual_count = len(env.task_manager.get_all_tasks())
    print(f"Expected task count: {task_count}")
    print(f"Actual task count: {actual_count}")

    return task_count == actual_count



def test_simulation_steps(env):
    """Test if the simulation can run multiple steps without errors."""
    print("\n=== Testing Simulation Steps ===")
    
    # Run 10 simulation steps
    num_steps = 10
    print(f"Running {num_steps} simulation steps...")
    
    try:
        for i in range(num_steps):
            start_time = time.time()
            is_done = env.step()
            end_time = time.time()
            time.sleep(0.1)
            
            print(f"Step {i+1}: Simulation time = {env.simulation_time:.2f}s, Processing time = {(end_time - start_time):.4f}s")
            print(f"  Vehicle count: {len(env.vehicles)}, UAV count: {len(env.UAVs)}")
            
            if is_done:
                print("Simulation completed")
                break
        
        return True
    except Exception as e:
        print(f"Error during simulation steps: {e}")
        return False

def visualize_uav_path(positions, destination=None):
    """Visualize the UAV path in 3D."""
    positions = np.array(positions)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='UAV Path')
    
    # Plot the positions as points
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', marker='o')
    
    # Mark start and end positions
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='green', marker='o', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='red', marker='o', s=100, label='Current')
    
    # Plot destination if provided
    if destination is not None:
        ax.scatter(destination[0], destination[1], destination[2], c='purple', marker='*', s=200, label='Destination')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('UAV Flight Path')
    ax.legend()
    
    plt.savefig('uav_path.png')
    print("UAV path visualization saved to 'uav_path.png'")

def generate_sample_data_if_needed():
    """Generate sample vehicle data if using FILE mode and data doesn't exist."""
    if not os.path.exists("sample_vehicle_data.csv"):
        try:
            from create_sample_vehicle_data import generate_sample_vehicle_data
            generate_sample_vehicle_data("sample_vehicle_data.csv")
        except ImportError:
            print("Warning: create_sample_vehicle_data.py not found, cannot generate sample data")
        except Exception as e:
            print(f"Error generating sample data: {e}")

def run_all_tests(use_sumo=True):
    """Run all tests for the AirLogixSim environment."""
    # Path to config file
    if use_sumo:
        config_path = "config.yaml"
    else:
        config_path = "config_no_sumo.yaml"
        generate_sample_data_if_needed()
    
    # Load configuration
    try:
        config = load_config(config_path)
        print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Create environment
    try:
        env = AirLogixSimEnv(config)
        print("Environment created successfully")
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    
    # Run the tests
    tests = {
        "Initialization": test_initialization,
        "Vehicle Creation": test_vehicle_creation,
        "UAV Creation": test_uav_creation,
        "UAV Docking": test_uav_docking,
        "UAV Takeoff and Destination": test_uav_takeoff_and_destination,
        "Simulation Steps": test_simulation_steps
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
    parser = argparse.ArgumentParser(description='Test the AirLogixSim environment.')
    parser.add_argument('--no-sumo', action='store_true', help='Run without SUMO in FILE mode')
    args = parser.parse_args()
    
    run_all_tests(use_sumo=not args.no_sumo) 