import os
import pandas as pd
import numpy as np
import random
import string

def generate_sample_vehicle_data(num_vehicles=5, output_file="sample_vehicle_data.csv"):
    """
    Generate sample vehicle data for testing without SUMO.
    
    Args:
        num_vehicles (int): Number of vehicles to generate
        output_file (str): Output CSV file name
    """
    # Check if the file already exists
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping generation.")
        return
    
    # Generate random vehicle IDs
    vehicle_ids = [f"vehicle_{i}" for i in range(num_vehicles)]
    
    # Generate random routes within the map boundaries
    routes = []
    for i in range(num_vehicles):
        # Generate a route with 5-10 waypoints
        num_waypoints = random.randint(5, 10)
        
        # Generate random route waypoints
        route = []
        for j in range(num_waypoints):
            x = random.uniform(0, 1000)  # Assuming map size from config
            y = random.uniform(0, 1000)
            route.append((x, y))
        
        routes.append(route)
    
    # Generate dataframe with timestamps
    data = []
    simulation_time = 0
    simulation_step = 0.5  # 0.5 second intervals
    
    # Generate data for 1000 simulation steps (500 seconds)
    for step in range(1000):
        simulation_time += simulation_step
        
        for idx, vehicle_id in enumerate(vehicle_ids):
            # Calculate the route point index based on simulation time
            route = routes[idx]
            route_idx = min(int(step / 100), len(route) - 1)
            
            # Get position at this time
            position = route[route_idx]
            
            # Add some noise to make it look like the vehicle is moving
            x = position[0] + random.uniform(-5, 5)
            y = position[1] + random.uniform(-5, 5)
            
            # Speed between 0 and 30 m/s
            speed = random.uniform(0, 30)
            
            # Random acceleration between -2 and 2 m/s^2
            acceleration = random.uniform(-2, 2)
            
            # Random angle between 0 and 2π
            angle = random.uniform(0, 2 * np.pi)
            
            # Add to data
            data.append({
                'vehicle_id': vehicle_id,
                'time': simulation_time,
                'position_x': x,
                'position_y': y,
                'position_z': 0,  # Vehicles are at ground level
                'speed': speed,
                'acceleration': acceleration,
                'angle': angle
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Write to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated sample vehicle data with {num_vehicles} vehicles at {output_file}")
    print(f"Contains {len(data)} total positions across {len(vehicle_ids)} vehicles")

if __name__ == "__main__":
    # Generate the sample data
    output_file = os.path.join(os.path.dirname(__file__), "sample_vehicle_data.csv")
    generate_sample_vehicle_data(num_vehicles=5, output_file=output_file) 