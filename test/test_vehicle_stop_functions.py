import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airlogixsim.airlogixsim_env import AirLogixSimEnv

# Note: stopVehicle and stopNearestVehicle functions now include a stop_offset parameter (default 20.0 meters)
# This offset adds distance to the stopping position to prevent "too close to braking point" SUMO errors

def load_config(config_path):
    """Load the configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def test_initialization(env):
    """Test if the environment initializes correctly."""
    print("\n=== Testing Environment Initialization ===")
    print(f"Simulation time: {env.simulation_time}")
    print(f"Traffic mode: {env.traffic_manager._traffic_mode}")
    
    # Check if SUMO connection established (if using SUMO mode)
    if env.config['traffic']['traffic_mode'] == 'SUMO':
        print(f"SUMO connection established: {env.traci_connection is not None}")
    
    # Check traffic manager initialization
    print(f"Traffic manager initialized: {env.traffic_manager is not None}")
    
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
    
    # Check SUMO vehicle count
    if env.traffic_manager._traffic_mode == 'SUMO':
        sumo_vehicles = env.traffic_manager.getVehicleIDsList()
        print(f"SUMO vehicle count: {len(sumo_vehicles)}")
    
    # Check vehicle position and speed
    if len(env.vehicles) > 0:
        vehicle_id = list(env.vehicles.keys())[0]
        vehicle = env.vehicles[vehicle_id]
        print(f"\nSample Vehicle (ID: {vehicle_id}):")
        print(f"Position: {vehicle.getPosition()}")
        print(f"Speed: {vehicle.getSpeed()}")
        print(f"Angle: {vehicle.getAngle()}")
    
    # Add some simulation steps to let vehicles start moving
    print("\nRunning a few simulation steps to let vehicles start moving...")
    for i in range(5):
        env.step()
        time.sleep(0.1)  # Pause to observe
    
    return actual_count > 0

def test_stop_vehicle(env):
    """Test the functionality to stop a single vehicle."""
    print("\n=== Testing Single Vehicle Stop Functionality ===")
    
    # Get available vehicle list
    vehicle_ids = env.traffic_manager.getVehicleIDsList()
    
    if not vehicle_ids:
        print("No available vehicles, test failed")
        return False
    
    # Select the first vehicle for parking test
    test_vehicle_id = vehicle_ids[0]
    print(f"Selected vehicle {test_vehicle_id} for parking test")
    
    # Get initial vehicle state
    vehicle_info_before = env.traffic_manager.getVehicleInfoByIds([test_vehicle_id])[test_vehicle_id]
    initial_position = vehicle_info_before['position']
    initial_speed = vehicle_info_before['speed']
    print(f"Vehicle state before parking: Position={initial_position}, Speed={initial_speed}")

    # Make the vehicle stop for 10 seconds with a safe braking distance offset
    stop_duration = 10
    stop_offset = 25.0  # Set a slightly larger offset for testing to ensure it works
    print(f"Using stop offset of {stop_offset} meters to ensure safe braking distance")
    stop_success = env.traffic_manager.stopVehicle(test_vehicle_id, stop_duration, parking=True, stop_offset=stop_offset)
    print(f"Stop command execution {'successful' if stop_success else 'failed'}")
    
    if not stop_success:
        return False
    
    # Simulate several steps, observe if the vehicle stops
    print("\nObserving vehicle stop status...")
    positions = []
    speeds = []
    
    for i in range(100):
        env.step()
        vehicle_info = env.traffic_manager.getVehicleInfoByIds([test_vehicle_id])[test_vehicle_id]
        current_position = vehicle_info['position']
        current_speed = vehicle_info['speed']
        positions.append(current_position)
        speeds.append(current_speed)
        print(f"Step {i+1}: Position={current_position}, Speed={current_speed}")
        time.sleep(0.1)  # Pause to observe
    
    # Check if the vehicle has actually stopped (speed close to 0)
    is_stopped = abs(speeds[-1]) < 0.1
    print(f"Vehicle stopped: {is_stopped}")
    
    # Resume vehicle movement early
    resume_success = env.traffic_manager.resumeVehicle(test_vehicle_id)
    print(f"\nResume movement command execution {'successful' if resume_success else 'failed'}")
    
    if not resume_success:
        return is_stopped
    
    # Simulate several steps, observe if the vehicle resumes movement
    print("\nObserving vehicle resume status...")
    resume_speeds = []
    
    for i in range(5):
        env.step()
        vehicle_info = env.traffic_manager.getVehicleInfoByIds([test_vehicle_id])[test_vehicle_id]
        current_position = vehicle_info['position']
        current_speed = vehicle_info['speed']
        resume_speeds.append(current_speed)
        print(f"After resume step {i+1}: Position={current_position}, Speed={current_speed}")
        time.sleep(0.1)  # Pause to observe
    
    # Check if the vehicle resumed movement (speed greater than 0)
    is_resumed = resume_speeds[-1] > 0.1
    print(f"Vehicle resumed movement: {is_resumed}")
    
    # Visualize vehicle speed changes
    visualize_vehicle_speed(speeds + resume_speeds, "vehicle_speed_change.png", "Vehicle Speed Changes During Stop and Resume")
    
    return is_stopped and is_resumed

def test_stop_nearest_vehicle(env):
    """Test the functionality to find and stop the nearest vehicle to a specified position."""
    print("\n=== Testing Stop Nearest Vehicle Functionality ===")
    
    # Get positions of all vehicles in the map
    vehicle_ids = env.traffic_manager.getVehicleIDsList()
    
    if not vehicle_ids:
        print("No available vehicles, test failed")
        return False
    
    # Get the average position of all vehicles as the target position
    all_positions = []
    for vehicle_id in vehicle_ids:
        vehicle_info = env.traffic_manager.getVehicleInfoByIds([vehicle_id])[vehicle_id]
        all_positions.append(vehicle_info['position'])
    
    # Calculate average position, and offset slightly to ensure it's not exactly on a vehicle
    avg_position = np.mean(all_positions, axis=0)
    target_position = (avg_position[0] + 10, avg_position[1] + 10, 0)
    print(f"Target position: {target_position}")
    
    # Try to find and stop the nearest vehicle to the target position
    max_distance = 200.0  # Set a larger search range
    stop_offset = 25.0    # Set a safe braking distance offset
    print(f"Using stop offset of {stop_offset} meters to ensure safe braking distance")
    
    success, result = env.traffic_manager.stopNearestVehicle(
        position=target_position,
        duration=10,
        max_distance=max_distance,
        parking=True,
        stop_offset=stop_offset
    )
    
    if not success:
        print(f"Failed to find or stop vehicle, reason: {result}")
        return False
    
    vehicle_id = result
    print(f"Successfully found and stopped vehicle: {vehicle_id}")
    
    # Get vehicle information before parking
    vehicle_info_before = env.traffic_manager.getVehicleInfoByIds([vehicle_id])[vehicle_id]
    initial_position = vehicle_info_before['position']
    initial_speed = vehicle_info_before['speed']
    print(f"Vehicle state before parking: Position={initial_position}, Speed={initial_speed}")
    
    # Calculate distance between vehicle and target position
    distance_to_target = np.sqrt(
        (initial_position[0] - target_position[0])**2 + 
        (initial_position[1] - target_position[1])**2
    )
    print(f"Distance from vehicle to target position: {distance_to_target} meters")
    
    # Verify that the found vehicle is indeed the nearest one
    is_nearest = True
    for other_id in vehicle_ids:
        if other_id == vehicle_id:
            continue
        
        other_info = env.traffic_manager.getVehicleInfoByIds([other_id])[other_id]
        other_position = other_info['position']
        other_distance = np.sqrt(
            (other_position[0] - target_position[0])**2 + 
            (other_position[1] - target_position[1])**2
        )
        
        if other_distance < distance_to_target:
            print(f"Warning: Vehicle {other_id} is closer to target ({other_distance} < {distance_to_target})")
            is_nearest = False
    
    print(f"Found vehicle is the nearest one: {is_nearest}")
    
    # Simulate several steps, observe if the vehicle stops
    print("\nObserving vehicle stop status...")
    positions = []
    speeds = []
    
    for i in range(5):
        env.step()
        vehicle_info = env.traffic_manager.getVehicleInfoByIds([vehicle_id])[vehicle_id]
        current_position = vehicle_info['position']
        current_speed = vehicle_info['speed']
        positions.append(current_position)
        speeds.append(current_speed)
        print(f"Step {i+1}: Position={current_position}, Speed={current_speed}")
        time.sleep(0.1)  # Pause to observe
    
    # Check if the vehicle has actually stopped (speed close to 0)
    is_stopped = abs(speeds[-1]) < 0.1
    print(f"Vehicle stopped: {is_stopped}")
    
    # Resume vehicle movement early
    resume_success = env.traffic_manager.resumeVehicle(vehicle_id)
    print(f"\nResume movement command execution {'successful' if resume_success else 'failed'}")
    
    # Test passes if distance is less than max_distance, it's the nearest vehicle, and it actually stopped
    return distance_to_target <= max_distance and is_nearest and is_stopped

def visualize_vehicle_speed(speeds, filename, title):
    """Visualize vehicle speed changes."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(speeds)), speeds, 'b-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Simulation Steps')
    plt.ylabel('Vehicle Speed (m/s)')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    print(f"Vehicle speed chart saved to '{filename}'")

def run_all_tests(use_gui=False):
    """Run all vehicle stop functionality tests."""
    # Configuration file path
    config_path = "config.yaml"
    
    # Load configuration
    try:
        config = load_config(config_path)
        print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
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
        "Vehicle Creation": test_vehicle_creation,
        "Stop Single Vehicle": test_stop_vehicle,
        #"Stop Nearest Vehicle": test_stop_nearest_vehicle
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
    parser = argparse.ArgumentParser(description='Test vehicle stop functionality.')
    parser.add_argument('--gui', action='store_true', help='Use SUMO-GUI visualization mode')
    args = parser.parse_args()
    
    run_all_tests(use_gui=args.gui) 