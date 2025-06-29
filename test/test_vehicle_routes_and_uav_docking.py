import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# Add the parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airlogixsim.airlogixsim_env import AirLogixSimEnv

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def plot_trajectory(positions, title):
    """Plot trajectory graph"""
    plt.figure(figsize=(10, 8))
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    # Draw trajectory line
    plt.plot(x_coords, y_coords, 'b-', linewidth=2)
    
    # Mark start and end points
    plt.scatter(x_coords[0], y_coords[0], c='green', marker='o', s=100, label='Start')
    plt.scatter(x_coords[-1], y_coords[-1], c='red', marker='o', s=100, label='End')
    
    # Add direction arrows (one every 10 points)
    arrow_step = min(10, len(positions) - 1)
    if arrow_step > 0:
        for i in range(0, len(positions) - 1, arrow_step):
            dx = x_coords[i+1] - x_coords[i]
            dy = y_coords[i+1] - y_coords[i]
            plt.arrow(x_coords[i], y_coords[i], dx, dy, 
                    head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Save image
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()
    print(f"Trajectory plot saved as {title.replace(' ', '_')}.png")

def test_vehicle_routes(env, simulation_steps=20):
    """Test if vehicles follow their initial routes"""
    print("\n===== Testing Vehicle Routes =====")
    
    # Record trajectories for each vehicle
    vehicle_trajectories = defaultdict(list)
    vehicle_route_ids = {}
    
    # Run simulation and record trajectories
    print(f"Running {simulation_steps} simulation steps to track vehicle trajectories...")
    for step in range(simulation_steps):
        # Record current vehicle positions
        for vehicle_id, vehicle in env.vehicles.items():
            position = vehicle.getPosition()
            vehicle_trajectories[vehicle_id].append(position)
            
            # Record vehicle route ID (only once)
            if vehicle_id not in vehicle_route_ids:
                vehicle_traffic_info = env.traffic_manager._vehicle_infos.get(vehicle_id, {})
                route_id = vehicle_traffic_info.get('routeId', 'unknown')
                vehicle_route_ids[vehicle_id] = route_id
        
        # Single step simulation
        env.step()
        print(f"Completed simulation step {step+1}/{simulation_steps}, simulation time: {env.simulation_time:.2f}s")
    
    # Analyze trajectory data
    valid_trajectories = 0
    for vehicle_id, trajectory in vehicle_trajectories.items():
        if len(trajectory) >= 2:
            valid_trajectories += 1
            
            # Calculate total distance moved
            total_distance = 0
            for i in range(1, len(trajectory)):
                pos1 = np.array(trajectory[i-1])
                pos2 = np.array(trajectory[i])
                distance = np.linalg.norm(pos2 - pos1)
                total_distance += distance
            
            route_id = vehicle_route_ids.get(vehicle_id, 'unknown')
            print(f"Vehicle {vehicle_id} (Route ID: {route_id}):")
            print(f"  Starting position: {trajectory[0]}")
            print(f"  Final position: {trajectory[-1]}")
            print(f"  Trajectory points: {len(trajectory)}")
            print(f"  Total distance moved: {total_distance:.2f} meters")
            
            # Determine if vehicle is moving
            is_moving = total_distance > 0.1  # Set a small threshold to tolerate floating point errors
            print(f"  Vehicle is moving: {is_moving}")
            
            # Plot trajectory for each vehicle
            if len(trajectory) > 2:
                plot_trajectory(trajectory, f"Vehicle_{vehicle_id}_Trajectory")
    
    print(f"\nTracked trajectories for {valid_trajectories} vehicles")
    
    # Test success: at least one vehicle is moving
    test_success = valid_trajectories > 0
    return test_success

def test_uav_docking_accuracy(env, simulation_steps=20):
    """Test if UAVs are correctly docked to vehicles"""
    print("\n===== Testing UAV Docking Accuracy =====")
    
    # Record position differences between vehicles and their docked UAVs
    docking_accuracy_data = []
    
    # Find all docked UAVs and their corresponding vehicles
    docked_uavs = {}
    for uav_id, uav in env.UAVs.items():
        if uav.isDocked():
            vehicle_id = uav.getDockedVehicleId()
            if vehicle_id:
                docked_uavs[uav_id] = vehicle_id
    
    print(f"Found {len(docked_uavs)} UAVs docked to vehicles")
    for uav_id, vehicle_id in docked_uavs.items():
        print(f"UAV {uav_id} is docked to vehicle {vehicle_id}")
    
    # No docked UAVs, test fails
    if not docked_uavs:
        print("No UAVs found docked to vehicles, test failed")
        return False
    
    # Run simulation and record docking accuracy
    print(f"Running {simulation_steps} simulation steps to test docking accuracy...")
    for step in range(simulation_steps):
        for uav_id, vehicle_id in docked_uavs.items():
            # Ensure UAV and vehicle are still in simulation
            if uav_id in env.UAVs and vehicle_id in env.vehicles:
                uav = env.UAVs[uav_id]
                vehicle = env.vehicles[vehicle_id]
                
                # Check if UAV is still docked to vehicle
                if uav.isDocked() and uav.getDockedVehicleId() == vehicle_id:
                    uav_pos = uav.getPosition()
                    vehicle_pos = vehicle.getPosition()
                    z_offset = env.traffic_manager._UAV_docking_z_offset
                    
                    # Calculate XY-plane and Z-axis position differences
                    xy_diff = np.linalg.norm(np.array(uav_pos[:2]) - np.array(vehicle_pos[:2]))
                    z_diff = abs(uav_pos[2] - (vehicle_pos[2] + z_offset))
                    
                    docking_accuracy_data.append({
                        'step': step,
                        'uav_id': uav_id,
                        'vehicle_id': vehicle_id,
                        'xy_diff': xy_diff,
                        'z_diff': z_diff,
                        'uav_pos': uav_pos,
                        'vehicle_pos': vehicle_pos
                    })
        
        # Single step simulation
        env.step()
        print(f"Completed simulation step {step+1}/{simulation_steps}, simulation time: {env.simulation_time:.2f}s")
    
    # Analyze docking accuracy data
    if docking_accuracy_data:
        max_xy_diff = max(data['xy_diff'] for data in docking_accuracy_data)
        max_z_diff = max(data['z_diff'] for data in docking_accuracy_data)
        avg_xy_diff = sum(data['xy_diff'] for data in docking_accuracy_data) / len(docking_accuracy_data)
        avg_z_diff = sum(data['z_diff'] for data in docking_accuracy_data) / len(docking_accuracy_data)
        
        print("\nDocking Accuracy Statistics:")
        print(f"Maximum XY-plane position difference: {max_xy_diff:.6f} meters")
        print(f"Maximum Z-axis position difference: {max_z_diff:.6f} meters")
        print(f"Average XY-plane position difference: {avg_xy_diff:.6f} meters")
        print(f"Average Z-axis position difference: {avg_z_diff:.6f} meters")
        
        # Plot docking accuracy charts
        plt.figure(figsize=(12, 6))
        
        steps = sorted(set(data['step'] for data in docking_accuracy_data))
        
        # Calculate average position differences for each step
        avg_xy_by_step = []
        avg_z_by_step = []
        for s in steps:
            step_data = [d for d in docking_accuracy_data if d['step'] == s]
            avg_xy_by_step.append(sum(d['xy_diff'] for d in step_data) / len(step_data))
            avg_z_by_step.append(sum(d['z_diff'] for d in step_data) / len(step_data))
        
        plt.subplot(1, 2, 1)
        plt.plot(steps, avg_xy_by_step, 'b-', marker='o')
        plt.title('XY-Plane Position Difference')
        plt.xlabel('Simulation Step')
        plt.ylabel('Average Position Difference (meters)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(steps, avg_z_by_step, 'r-', marker='o')
        plt.title('Z-Axis Position Difference')
        plt.xlabel('Simulation Step')
        plt.ylabel('Average Position Difference (meters)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('docking_accuracy.png')
        plt.close()
        print("Docking accuracy chart saved as docking_accuracy.png")
        
        # Test success: maximum position difference should be below threshold
        xy_threshold = 0.01  # Allowable XY-plane position difference threshold
        z_threshold = 0.01   # Allowable Z-axis position difference threshold
        
        test_success = max_xy_diff < xy_threshold and max_z_diff < z_threshold
        if test_success:
            print("Docking accuracy test passed!")
        else:
            print("Docking accuracy test failed: Position difference exceeds threshold")
            if max_xy_diff >= xy_threshold:
                print(f"XY-plane position difference ({max_xy_diff:.6f}) exceeds threshold ({xy_threshold})")
            if max_z_diff >= z_threshold:
                print(f"Z-axis position difference ({max_z_diff:.6f}) exceeds threshold ({z_threshold})")
        
        return test_success
    else:
        print("No docking accuracy data collected, test failed")
        return False

def run_tests(config_path="config.yaml", simulation_steps=20):
    """Run tests"""
    # Load configuration
    try:
        config = load_config(config_path)
        print(f"Loaded configuration: {config_path}")
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
    
    # Run tests
    tests = [
        ("Vehicle Route Test", lambda: test_vehicle_routes(env, simulation_steps)),
        ("UAV Docking Accuracy Test", lambda: test_uav_docking_accuracy(env, simulation_steps))
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 40}")
        print(f"Running test: {test_name}")
        print(f"{'=' * 40}")
        
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            duration = end_time - start_time
            print(f"Test {test_name}: {status} (Duration: {duration:.2f} seconds)")
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Test vehicle routes and UAV docking functionality')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--steps', type=int, default=2000, help='Number of simulation steps')
    
    args = parser.parse_args()
    
    run_tests(args.config, args.steps) 