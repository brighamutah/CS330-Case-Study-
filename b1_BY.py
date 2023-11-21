import copy
import json
import csv
import math
import heapq
from datetime import datetime, timedelta
import time
import random

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_csv(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['Date/Time'] = datetime.strptime(row['Date/Time'], '%m/%d/%Y %H:%M:%S')
            data.append(row)
    return data

def construct_graph(adjacency_list):
    graph = {}
    for start_node, edges in adjacency_list.items():
        graph[start_node] = {}
        for end_node, attributes in edges.items():
            graph[start_node][end_node] = attributes
    return graph

def find_nearest_node(lat1, lon1, node_data):
    nearest_node = None
    min_distance = float('inf')

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    for node_id, coords in node_data.items():
        lat2 = math.radians(float(coords['lat']))
        lon2 = math.radians(float(coords['lon']))
        if lat1 == lat2 and lon1 == lon2:
            return node_id

        distance = haversine(lat1, lon1, lat2, lon2)
        if distance < min_distance:
            min_distance = distance
            nearest_node = node_id

    return nearest_node

def bs_lat_lon(target, node_list, latlon, nodes):
    left, right = 0, len(node_list)-1
    mid = left + (right - left) // 2
    while left <= right:
        mid = left + (right - left) // 2
        if nodes[node_list[mid]][latlon] == target:
            left = mid
            break
        elif nodes[node_list[mid]][latlon] < target:
            left = mid + 1
        else:
            right = mid - 1

    range = len(node_list) * 0.1

    return node_list[int(mid-range//2):int(mid+range//2)]

def fnn_est(lat1, lon1, n_lat, n_lon, node_data):
    nearest_node = None
    min_distance = float('inf')

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    # Nodes of interest
    noi = list(set(bs_lat_lon(lat1, n_lat, 'lat', node_data) + bs_lat_lon(lon1, n_lon, 'lon', node_data)))

    for node_id in noi:
        lat2 = math.radians(float(node_data[node_id]['lat']))
        lon2 = math.radians(float(node_data[node_id]['lon']))
        if lat1 == lat2 and lon1 == lon2: return node_id

        distance = haversine(lat1, lon1, lat2, lon2) # meters
        if distance < 0.1: # terminate early if distance within 0.1 km or 100 m
            return node_id

        if distance < min_distance:
            min_distance = distance
            nearest_node = node_id

    return nearest_node

def haversine(lat1, lon1, lat2, lon2):
    # DISTANCE IN METERS
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return abs(distance)


def calculate_route_time(start_node, end_node, graph, current_time):
    # Returns time it takes to travel from start to end in hours
    distances = {node: float('infinity') for node in graph}
    distances[start_node] = 0
    pq = [(0, start_node)]

    current_hour = current_time.hour
    day_type = 'weekday' if current_time.weekday() < 5 else 'weekend'

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == end_node:
            return distances[end_node]


        for neighbor, attributes_list in graph[current_node].items():
            for attributes in attributes_list:
                if attributes['day_type'] == day_type and attributes['hour'] == current_hour:
                    distance = current_distance + attributes['time']
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))
                    break

    return float('infinity')

def route_time_a_star(start_node, end_node, graph, current_time, nodes, h_weight):
    # Returns time it takes to travel from start to end in hours
    distances = {node: float('infinity') for node in graph}
    a_star_heuristic = {node: float('infinity') for node in graph}
    distances[start_node] = 0
    a_star_heuristic[start_node] = 0
    pq = [(0, 0, start_node)]

    current_hour = current_time.hour
    day_type = 'weekday' if current_time.weekday() < 5 else 'weekend'

    while pq:
        current_heuristic, current_distance, current_node = heapq.heappop(pq)

        if current_node == end_node:
            return distances[end_node]


        for neighbor, attributes_list in graph[current_node].items():
            for attributes in attributes_list:
                if attributes['day_type'] == day_type and attributes['hour'] == current_hour:
                    curr_lat, curr_lon = float(nodes[current_node]['lat']), float(nodes[current_node]['lon'])
                    n_lat, n_lon = float(nodes[neighbor]['lat']), float(nodes[neighbor]['lon'])
                    distance = current_distance + attributes['time']
                    heuristic = (distance + haversine(curr_lat, curr_lon, n_lat, n_lon)*h_weight)
                    if heuristic < a_star_heuristic[neighbor]:
                        distances[neighbor] = distance
                        a_star_heuristic[neighbor] = heuristic
                        heapq.heappush(pq, (heuristic, distance, neighbor))
                    break

    return float('infinity')

def reinsert_driver(drivers, driver, available, new_loc):
    prob = 0.98
    rand = random.random()
    if rand > prob: return drivers

    left, right = 0, len(drivers) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if drivers[mid]['Date/Time'] == available:
            left = mid
            break
        elif drivers[mid]['Date/Time'] < available:
            left = mid + 1
        else:
            right = mid - 1

    driver['Date/Time'] = available
    driver['Node'] = new_loc
    drivers.insert(left, driver)

    return drivers

def k_centers_clustering(passengers_data, k):
    # Simple k-centers clustering implementation using Haversine distance
    coords = [(float(p['Source Lat']), float(p['Source Lon'])) for p in passengers_data]
    centers = [coords[0]]  # Start with the first coordinate as the initial center
    distances = [float('inf')] * len(coords)

    for _ in range(1, k):
        max_distance_index = max(range(len(coords)), key=lambda i: distances[i])
        centers.append(coords[max_distance_index])

        for i, coord in enumerate(coords):
            distances[i] = min(distances[i], haversine_distance(coord, centers[-1]))

    passenger_clusters = [[] for _ in range(k)]
    for i, coord in enumerate(coords):
        cluster_index = min(range(k), key=lambda j: haversine_distance(coord, centers[j]))
        passenger_clusters[cluster_index].append(passengers_data[i])

    return passenger_clusters

def precompute_travel_times(passenger_clusters, graph, nodes, h_weight):
    # Precompute travel times for each batch of passengers
    precomputed_times = []
    
    for cluster in passenger_clusters:
        cluster_times = []
        for passenger in cluster:
            passenger_pickup_node = passenger['Source Node']
            passenger_dropoff_node = passenger['Dest Node']
            time_to_passenger = route_time_a_star(
                cluster[0]['Node'], passenger_pickup_node, graph, cluster[0]['Date/Time'], nodes, h_weight
            )
            time_to_destination = route_time_a_star(
                passenger_pickup_node, passenger_dropoff_node, graph, cluster[0]['Date/Time'], nodes, h_weight
            )
            cluster_times.append((time_to_passenger, time_to_destination))
        
        precomputed_times.append(cluster_times)
    
    return precomputed_times


def match_and_calculate_metrics(drivers, passengers, graph, nodes, h_weight):
    # USING A* FOR EVERYTHING (part 2)
    wait_times = []
    profit_times = []
    d1_times = []
    driver_profit = [0 for i in range(len(drivers))]
    driver_n_trips = [0 for i in range(len(drivers))]

    drivers.sort(key=lambda x: x['Date/Time'])
    #passengers.sort(key=lambda x: x['Date/Time'])
    avg_runtime = 0
    total_n = 0
    T = {}
    for cluster, cluster_times in zip(passenger_clusters, precomputed_times):
        for passenger, (time_to_passenger, time_to_destination) in zip(cluster, cluster_times):
            start = time.time()
            print(f'Remaining Passengers = {len(passengers)}')
            print(f'Remaining Driver = {len(drivers)}')
            print(f'Est. Remaining Time: {(len(passengers) * avg_runtime) / 60.0: .2f} minutes')
            
            passenger_time = passenger['Date/Time']
            passenger_pickup_node = passenger['Source Node']
            passenger_dropoff_node = passenger['Dest Node']

            # BRUTE FORCE
            driver = list(drivers)[0]
            d_lat, d_lon = float(nodes[driver['Node']]['lat']), float(nodes[driver['Node']]['lon'])
            available_drivers = [(haversine(d_lat, d_lon, float(nodes[passenger_pickup_node]['lat']), float(nodes[passenger_pickup_node]['lon'])), 0)]
            t = max(driver['Date/Time'], passenger_time)

            i = 1
            while driver['Date/Time'] <= passenger_time and i < len(drivers):  # get max 10 earliest available drivers
                driver = list(drivers)[i]
                d_lat, d_lon = float(nodes[driver['Node']]['lat']), float(nodes[driver['Node']]['lon'])
                heapq.heappush(available_drivers, (haversine(d_lat, d_lon, float(nodes[passenger_pickup_node]['lat']), float(nodes[passenger_pickup_node]['lon'])), i))
                i += 1

            closest_drivers = []
            j = 0
            while available_drivers and j < 10:
                _, index = heapq.heappop(available_drivers)
                closest_drivers.append(index)
                j += 1

            min_ttp = float('infinity')
            c_d_i = 0
            if len(closest_drivers) != 1:
                for ind in closest_drivers:
                    d = list(drivers)[ind]
                    d_t, d_node = d['Date/Time'], d['Node']
                    ttp = route_time_a_star(d_node, passenger_pickup_node, graph, max(d_t, passenger_time), nodes, h_weight)
                    if ttp < min_ttp:
                        min_ttp = ttp
                        c_d_i = ind

            closest_driver = drivers.pop(c_d_i)
            driver_node = closest_driver['Node']
            driver_time = closest_driver['Date/Time']

            match_time = max(driver_time, passenger_time)   # datetime object

            match_wait_time = match_time - passenger_time   # datetime object
            match_wait_hours = match_wait_time.total_seconds()/3600.0  # wait time in hours

            wait_time = match_wait_hours + time_to_passenger  # in hours
            profit_time = time_to_destination - time_to_passenger  # in hours
            d1 = wait_time + time_to_destination

            driver_profit[driver['ID']] += profit_time
            driver_n_trips[driver['ID']] += 1

            available_time = match_time + timedelta(hours=(time_to_passenger + time_to_destination))  # datetime object
            drivers = reinsert_driver(drivers, closest_driver, available_time, passenger_dropoff_node)

            wait_times.append(wait_time)
            profit_times.append(profit_time)
            d1_times.append(d1)
            end = time.time()
            avg_runtime = avg_runtime*(total_n-1) + (end-start)
            avg_runtime /= total_n

    average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
    average_profit_time = sum(profit_times) / len(profit_times) if profit_times else 0
    average_d1_time = sum(d1_times) / len(d1_times) if d1_times else 0

    return average_wait_time, average_profit_time, average_d1_time, driver_profit, driver_n_trips

#%%
start = time.time()
adjacency_list = load_json("adjacency.json")
node_data = load_json("node_data.json")
drivers_data = load_csv("drivers.csv")
passengers_data = load_csv("passengers.csv")
end = time.time()
print(f'Data Load Time: {(end - start)/60.0: .3f} minutes')
#%%
# TRUE NEAREST NODE
start = time.time()
for d in drivers_data:
    dlat, dlon = float(d['Source Lat']), float(d['Source Lon'])
    d['Node'] = find_nearest_node(dlat, dlon, node_data)

for p in passengers_data:
    pslat, pslon = float(p['Source Lat']), float(p['Source Lon'])
    pdlat, pdlon = float(p['Dest Lat']), float(p['Dest Lon'])
    p['Source Node'] = find_nearest_node(pslat, pslon, node_data)
    p['Dest Node'] = find_nearest_node(pdlat, pdlon, node_data)
end = time.time()
print(f'Finding Nearest Nodes of all Drivers/Passengers: {(end-start)/60.0: .3f} minutes')
#%%
graph = construct_graph(adjacency_list)
#%%
#IDEA 2: get range of nodes nearest to target by lat and long, then find nn and terminate when within range
n_lat = list(node_data.keys())
n_lon = list(node_data.keys())
n_lat.sort(key=lambda x: node_data[x]['lat']) # NODE IDs of nodes sorted by lat
n_lon.sort(key=lambda x: node_data[x]['lon']) # NODE IDs of nodes sorted by lon
#%%
start = time.time()
id = 0
for d in drivers_data:
    dlat, dlon = float(d['Source Lat']), float(d['Source Lon'])
    d['Node'] = fnn_est(dlat, dlon, n_lat, n_lon, node_data)
    d['ID'] = id
    id += 1

for p in passengers_data:
    pslat, pslon = float(p['Source Lat']), float(p['Source Lon'])
    pdlat, pdlon = float(p['Dest Lat']), float(p['Dest Lon'])
    p['Source Node'] = fnn_est(pslat, pslon, n_lat, n_lon, node_data)
    p['Dest Node'] = fnn_est(pdlat, pdlon, n_lat, n_lon, node_data)
end = time.time()
print(f'Finding Estimated Nearest Nodes of all Drivers/Passengers: {(end-start)/60.0: .3f} minutes')
#%%
# K-centers clustering and precompute travel times
k = 4  
passenger_clusters = k_centers_clustering(passengers_data, k)
precomputed_times = precompute_travel_times(passenger_clusters, graph, node_data, h_weight)


#weight computation 
weights = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
best_runtime = float("infinity")
best_weight = -1
for w in weights:
    start_time = time.time()
    average_wait_time, average_profit_time, avg_d1, driver_profit, driver_trips = (
        match_and_calculate_metrics(copy.deepcopy(drivers_data), copy.deepcopy(passengers_data[:100]), graph, node_data,w))
    end_time = time.time()

    if (end_time - start_time) < best_runtime:
        best_runtime = (end_time - start_time)
        best_weight = w

    print(f"Average Wait Time for Passengers: {average_wait_time} hours")
    print(f"Average Profit Time for Drivers (D2): {average_profit_time} hours")
    print(f'Average D1: {avg_d1} hours')
    print(f"Runtime (excluding loading data): {(end_time - start_time)/60.0} minutes")
print(f'Best weight: {best_weight}')
#%%
import matplotlib.pyplot as plt
plt.figure()
plt.hist(driver_profit)
plt.xlabel('Profit Time (hrs)')
plt.ylabel('Frequency')
plt.show()

plt.figure()
plt.hist(driver_trips)
plt.xlabel('Number of Trips')
plt.ylabel('Frequency')
plt.show()
# #%%
# # Determine best weight for heuristic (Haversine distance) in A*
# weights = [0.001, 0.1, 0.5, 1, 10]
# # Optimizing to minimize passenger wait time
# best_weight = -1
# best_wait_time = float('infinity')
#
# for i, w in enumerate(weights):
#     start_time = time.time()
#     average_wait_time, average_profit_time, average_trip_time, driver_profit = (
#         match_and_calculate_metrics(drivers_data, passengers_data, graph, node_data, w))
#     end_time = time.time()
#     print(f'---------------------------------------------------')
#     print(f'Weight = {w}')
#     print(f"Average Wait Time for Passengers (D1): {average_wait_time} hours")
#     print(f"Average Profit Time for Drivers (D2): {average_profit_time} hours")
#     print(f'Runtime:  {(end_time - start_time) / 60.0} minutes')
#     if average_wait_time < best_wait_time:
#         best_wait_time = average_wait_time
#         best_weight = w
# print(f'Best weight = {best_weight}')
# print(f'Best wait time = {best_wait_time}')
#
