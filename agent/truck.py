class TruckAgent:
    pass


def plan_truck_route(nodes, centers, warehouse):
    coords = [warehouse]
    for center in centers:
        coords.append(nodes[center])
    n = len(coords)
    distance_matrix = []
    for i in range(len(coords)):
        distance_matrix.append([])
        for j in range(len(coords)):
            distance_matrix[i].append(abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1]))

    # memo dictionary: key = (mask, pos), value = minimal cost to finish the tour starting from 'pos'
    memo = {}
    # parent dictionary for reconstructing the optimal path
    parent = {}

    # print(coords)
    # print(distance_matrix)

    def dp(mask, pos):
        # If all cities have been visited, return cost to go back to the starting point (city 0)
        if mask == (1 << n) - 1:
            return distance_matrix[pos][0]

        # Return already computed cost if available
        if (mask, pos) in memo:
            return memo[(mask, pos)]

        min_cost = float('inf')
        # Try going to any city that hasn't been visited yet.
        for city in range(n):
            if mask & (1 << city) == 0:
                new_mask = mask | (1 << city)
                cost = distance_matrix[pos][city] + dp(new_mask, city)
                if cost < min_cost:
                    min_cost = cost
                    parent[(mask, pos)] = city
        memo[(mask, pos)] = min_cost
        return min_cost

    # Start from city 0; the initial mask with only city 0 visited is represented by 1.
    total_cost = dp(1, 0)

    # Reconstruct the optimal path using the parent dictionary.
    mask = 1
    pos = 0
    path = []
    for _ in range(n - 1):
        next_city = parent[(mask, pos)]
        path.append(centers[next_city - 1])
        mask |= (1 << next_city)
        pos = next_city

    return total_cost, path


if __name__ == '__main__':
    ret = plan_truck_route([[3, 1],
                            [-3, -6],
                            [-5, 6],
                            [-4, -1],
                            [6, 8],
                            [5, 6],
                            [0, -8],
                            [-7, 4],
                            [-2, -9],
                            [-2, -3],
                            [-5, 7],
                            [-6, 5],
                            [5, 5],
                            [9, 0],
                            [-9, -3],
                            [1, -5],
                            [0, 9],
                            [-9, 0],
                            [-7, 0],
                            [-2, 5]], [13, 9, 2, 1, 7], [0, 0])
    print(ret)
