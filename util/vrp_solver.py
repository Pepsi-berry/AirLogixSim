from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def solve_vrp(data):
    """Entry point of the program."""
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        routes = []
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            plan_output = []
            while not routing.IsEnd(index):
                plan_output.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            plan_output.append(manager.IndexToNode(index))
            routes.append(plan_output)
        return routes
    else:
        return None


def calc_matrix(env, obs, info):
    nodes = obs["truck_0_0"]["nodes"].tolist()
    centers = list(info['center_node'].values())
    warehouse = env.warehouse
    coords = [warehouse]
    for center in centers:
        coords.append(nodes[center])
    distance_matrix = []
    for i in range(len(coords)):
        distance_matrix.append([])
        for j in range(len(coords)):
            distance_matrix[i].append(abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1]))
    return distance_matrix
