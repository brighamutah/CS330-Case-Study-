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
        if distance < 0.2: # terminate early if distance within 0.2 km or 200 m
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
    return distance


def calculate_route_time(start_node, end_node, graph, current_time, nodes):
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
    distances[start_node] = 0
    pq = [(0, start_node)]

    current_hour = current_time.hour
    day_type = 'weekday' if current_time.weekday() < 5 else 'weekend'

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == end_node:
            return distances[end_node]

        curr_lat, curr_lon = float(nodes[current_node]['lat']), float(nodes[current_node]['lon'])

        for neighbor, attributes_list in graph[current_node].items():
            if neighbor not in nodes.keys(): continue
            n_lat, n_lon = float(nodes[neighbor]['lat']), float(nodes[neighbor]['lon'])
            for attributes in attributes_list:
                if attributes['day_type'] == day_type and attributes['hour'] == current_hour:
                    distance = current_distance + attributes['time'] + haversine(curr_lat, curr_lon, n_lat, n_lon)*h_weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))
                    break

    return float('infinity')

def a_star_est(start_node, end_node, graph, current_time, nodes, h_weight):
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

        curr_lat, curr_lon = float(nodes[current_node]['lat']), float(nodes[current_node]['lon'])

        for neighbor, attributes_list in graph[current_node].items():
            if neighbor not in nodes.keys(): continue
            n_lat, n_lon = float(nodes[neighbor]['lat']), float(nodes[neighbor]['lon'])
            for attributes in attributes_list:
                if attributes['day_type'] == day_type and attributes['hour'] == current_hour:
                    distance = current_distance + attributes['time'] + haversine(curr_lat, curr_lon, n_lat, n_lon)*h_weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))
                    break

    return 100*haversine(nodes[start_node]['lat'], nodes[start_node]['lon'], nodes[end_node]['lat'], nodes[end_node]['lon'])
def reinsert_driver(drivers, driver, available, new_loc):
    prob = 0.9
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

def match_and_calculate_metrics(drivers, passengers, graph, nodes, h_weight):
    # USING A* FOR EVERYTHING (part 2)
    wait_times = []
    total_trip_times = []
    profit_times = []

    drivers.sort(key=lambda x: x['Date/Time'])
    passengers.sort(key=lambda x: x['Date/Time'])

    while drivers and passengers:
        print(f'Remaining Passengers = {len(passengers)}')
        passenger = passengers.pop(0)
        passenger_time = passenger['Date/Time']
        passenger_pickup_node = passenger['Source Node']
        passenger_dropoff_node = passenger['Dest Node']

        # BRUTE FORCE
        driver = drivers.pop(0)
        available_drivers = [driver]
        t = max(driver['Date/Time'], passenger_time)

        i = 1
        while passenger_time >= t and i < 5: # get max 5 earliest available drivers
            driver = drivers.pop(i)
            i+=1
            available_drivers.append(driver)

        min_ttp = float('infinity')
        closest_driver = None
        for d in available_drivers:
            d_t, d_node = d['Date/Time'], d['Node']
            ttp = route_time_a_star(d_node, passenger_pickup_node, graph, max(d_t, passenger_time), nodes, h_weight)
            if ttp < min_ttp:
                min_ttp = ttp
                closest_driver = d

        driver_node = closest_driver['Node']
        driver_time = closest_driver['Date/Time']

        match_time = max(driver_time, passenger_time)   # datetime object

        match_wait_time = match_time - passenger_time   # datetime object
        match_wait_hours = match_wait_time.total_seconds()/3600.0 # wait time in hours

        time_to_passenger = route_time_a_star(driver_node, passenger_pickup_node, graph, match_time, nodes, h_weight) # in hours
        time_to_destination = route_time_a_star(passenger_pickup_node, passenger_dropoff_node, graph, match_time, nodes, h_weight) # in hours

        wait_time = match_wait_hours + time_to_passenger # in hours
        profit_time = time_to_destination - time_to_passenger # in hours

        available_time = match_time + timedelta(hours=(time_to_passenger + time_to_destination)) # datetime object
        drivers = reinsert_driver(drivers, closest_driver, available_time, passenger_dropoff_node)

        wait_times.append(wait_time)
        total_trip_times.append(time_to_destination)
        profit_times.append(profit_time)

    average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
    average_profit_time = sum(profit_times) / len(profit_times) if profit_times else 0
    average_trip_time = sum(total_trip_times)/ len(total_trip_times) if total_trip_times else 0
    return average_wait_time, average_profit_time, average_trip_time

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
# Preprocessed graph for approximation
# IDEA 1: shrink node population by 10 randomly (doesn't work bc the path doesn't necessarily exist)
random.seed(42)
keys = random.sample(list(node_data.keys()), len(node_data)//5)
reduced_nodes = {k: node_data[k] for k in keys}
reduced_graph = {k: graph[k] for k in keys}
print(len(reduced_nodes))
#%%
# ESTIMATED NEAREST NODE
start = time.time()
for d in drivers_data:
    dlat, dlon = float(d['Source Lat']), float(d['Source Lon'])
    d['Node Est'] = find_nearest_node(dlat, dlon, reduced_nodes)

for p in passengers_data:
    pslat, pslon = float(p['Source Lat']), float(p['Source Lon'])
    pdlat, pdlon = float(p['Dest Lat']), float(p['Dest Lon'])
    p['Source Node Est'] = find_nearest_node(pslat, pslon, reduced_nodes)
    p['Dest Node Est'] = find_nearest_node(pdlat, pdlon, reduced_nodes)
end = time.time()
print(f'Finding Estimated Nearest Nodes of all Drivers/Passengers: {(end-start)/60.0: .3f} minutes')
#%%
start_time = time.time()
average_wait_time, average_profit_time, average_trip_time = (
        macm_id1(drivers_data, passengers_data, graph, node_data, 1, reduced_nodes, reduced_graph))
end_time = time.time()
print(f"Average Wait Time for Passengers (D1): {average_wait_time} hours")
print(f"Average Profit Time for Drivers (D2): {average_profit_time} hours")
print(f'Runtime:  {(end_time - start_time) / 60.0} minutes')
#%%
#IDEA 2: get range of nodes nearest to target by lat and long, then find nn and terminate when within range
n_lat = list(node_data.keys())
n_lon = list(node_data.keys())
n_lat.sort(key=lambda x: node_data[x]['lat']) # NODE IDs of nodes sorted by lat
n_lon.sort(key=lambda x: node_data[x]['lon']) # NODE IDs of nodes sorted by lon
#%%
start = time.time()
for d in drivers_data:
    dlat, dlon = float(d['Source Lat']), float(d['Source Lon'])
    d['Node Est'] = fnn_est(dlat, dlon, n_lat, n_lon, node_data)

for p in passengers_data:
    pslat, pslon = float(p['Source Lat']), float(p['Source Lon'])
    pdlat, pdlon = float(p['Dest Lat']), float(p['Dest Lon'])
    p['Source Node Est'] = fnn_est(pslat, pslon, n_lat, n_lon, node_data)
    p['Dest Node Est'] = fnn_est(pdlat, pdlon, n_lat, n_lon, node_data)
end = time.time()
print(f'Finding Estimated Nearest Nodes of all Drivers/Passengers: {(end-start)/60.0: .3f} minutes')
#%%
start_time = time.time()
average_wait_time, average_profit_time, _ = match_and_calculate_metrics(drivers_data, passengers_data, graph, node_data,10)
end_time = time.time()

print(f"Average Wait Time for Passengers (D1): {average_wait_time} hours")
print(f"Average Profit Time for Drivers (D2): {average_profit_time} hours")
print(f"Runtime (excluding loading data): {(end_time - start_time)/60.0} minutes")
#%%
#%%
#match and calculate metrics for idea 1
def macm_id1(drivers, passengers, graph, nodes, h_weight, nodes_r, g_r):
    # USING A* FOR EVERYTHING (part 2)
    wait_times = []
    total_trip_times = []
    profit_times = []

    drivers.sort(key=lambda x: x['Date/Time'])
    passengers.sort(key=lambda x: x['Date/Time'])

    while drivers and passengers:
        passenger = passengers.pop(0)
        passenger_time = passenger['Date/Time']
        passenger_pickup_node = passenger['Source Node']
        passenger_dropoff_node = passenger['Dest Node']
        est_pickup = passenger['Source Node Est']
        est_dropoff = passenger['Dest Node Est']

        # BRUTE FORCE
        driver = drivers.pop(0)
        available_drivers = [driver]
        t = max(driver['Date/Time'], passenger_time)        # earliest time both a driver and a passenger are available

        i = 1
        while passenger_time >= t and i < 5: # can maybe add "AND i < 10
            driver = drivers.pop(i)
            i+=1
            available_drivers.append(driver)

        min_ttp = float('infinity')
        closest_driver = None
        for d in available_drivers:
            d_t, d_node = d['Date/Time'], d['Node Est']
            ttp = a_star_est(d_node, est_pickup, g_r, max(d_t, passenger_time), nodes_r, h_weight)
            if ttp < min_ttp:
                min_ttp = ttp
                closest_driver = d

        driver_node = closest_driver['Node']
        driver_time = closest_driver['Date/Time']

        match_time = max(driver_time, passenger_time)   # datetime object

        match_wait_time = match_time - passenger_time   # datetime object
        match_wait_hours = match_wait_time.total_seconds()/3600.0 # wait time in hours

        time_to_passenger = route_time_a_star(driver_node, passenger_pickup_node, graph, match_time, nodes, h_weight) # in hours
        time_to_destination = route_time_a_star(passenger_pickup_node, passenger_dropoff_node, graph, match_time, nodes, h_weight) # in hours

        wait_time = match_wait_hours + time_to_passenger # in hours
        profit_time = time_to_destination - time_to_passenger # in hours

        available_time = match_time + timedelta(hours=(time_to_passenger + time_to_destination)) # datetime object
        drivers = reinsert_driver(drivers, closest_driver, available_time, passenger_dropoff_node)

        wait_times.append(wait_time)
        total_trip_times.append(time_to_destination)
        profit_times.append(profit_time)

    average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
    average_profit_time = sum(profit_times) / len(profit_times) if profit_times else 0
    average_trip_time = sum(total_trip_times)/ len(total_trip_times) if total_trip_times else 0
    return average_wait_time, average_profit_time, average_trip_time
#%%
# Determine best weight for heuristic (Haversine distance) in A*
weights = [0.001, 0.1, 0.5, 1, 10]
# Optimizing to minimize passenger wait time
best_weight = -1
best_wait_time = float('infinity')

for i, w in enumerate(weights):
    start_time = time.time()
    average_wait_time, average_profit_time, average_trip_time = (
        macm_id1(drivers_data, passengers_data, graph, node_data, w, reduced_nodes, reduced_graph))
    end_time = time.time()
    print(f'---------------------------------------------------')
    print(f'Weight = {w}')
    print(f"Average Wait Time for Passengers (D1): {average_wait_time} hours")
    print(f"Average Profit Time for Drivers (D2): {average_profit_time} hours")
    print(f'Runtime:  {(end_time - start_time) / 60.0} minutes')
    if average_wait_time < best_wait_time:
        best_wait_time = average_wait_time
        best_weight = w
print(f'Best weight = {best_weight}')
print(f'Best wait time = {best_wait_time}')

