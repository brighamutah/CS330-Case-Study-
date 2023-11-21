import json
import csv
import math
import heapq
from datetime import datetime, timedelta
import time
import random
import copy

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
        if lat1 == lat2 and lon1 == lon2: return node_id
        distance = 3963.0 * math.acos((math.sin(lat1) * math.sin(lat2)) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
        if distance < min_distance:
            min_distance = distance
            nearest_node = node_id

    return nearest_node

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

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

def reinsert_driver(drivers, driver, available, new_loc):
    prob = 0.95
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
        if lat1 == lat2 and lon1 == lon2: return node_id, len(noi)

        distance = haversine(lat1, lon1, lat2, lon2) # meters
        if distance < 0.1: # terminate early if distance within 0.1 km or 100 m
            return node_id, len(noi)

        if distance < min_distance:
            min_distance = distance
            nearest_node = node_id

    return nearest_node, len(noi)

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

import heapq

def macm_b2(drivers, passengers, graph, nodes, h_weight):
    # USING A* FOR EVERYTHING (part 2)
    wait_times = []
    profit_times = []
    d1_times = []
    match_times = []
    driver_profit = [0 for i in range(len(drivers))]
    driver_n_trips = [0 for i in range(len(drivers))]

    drivers.sort(key=lambda x: x['Date/Time'])
    passengers.sort(key=lambda x: x['Date/Time'])
    avg_runtime = 0
    total_n = 0
    T = {}
    c = 0.15

    while drivers and passengers:
        start = time.time()
        passenger = passengers.pop(0)
        total_n+=1
        passenger_time = passenger['Date/Time']
        passenger_pickup_node = passenger['Source Node']
        passenger_dropoff_node = passenger['Dest Node']

        source_lat, source_lon = float(nodes[passenger_pickup_node]['lat']), float(nodes[passenger_pickup_node]['lon'])

        # BRUTE FORCE
        driver = list(drivers)[0]
        d_lat, d_lon = float(nodes[driver['Node']]['lat']), float(nodes[driver['Node']]['lon'])
        available_drivers = [(haversine(d_lat, d_lon, source_lat, source_lon), 0)]
        t = max(driver['Date/Time'], passenger_time)

        i = 1
        while driver['Date/Time'] <= passenger_time and i < len(drivers): # get max 10 earliest available drivers
            driver = list(drivers)[i]
            d_id = driver["ID"]
            d_lat, d_lon = float(nodes[driver['Node']]['lat']), float(nodes[driver['Node']]['lon'])
            heapq.heappush(available_drivers, (haversine(d_lat, d_lon, source_lat, source_lon) + c * driver_n_trips[d_id] , i))
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
        match_wait_hours = match_wait_time.total_seconds()/3600.0 # wait time in hours


        time_to_passenger = route_time_a_star(driver_node, passenger_pickup_node, graph, match_time, nodes, h_weight) # in hours
        time_to_destination = route_time_a_star(passenger_pickup_node, passenger_dropoff_node, graph, match_time, nodes, h_weight) # in hours
  
        wait_time = match_wait_hours + time_to_passenger # in hours
        profit_time = time_to_destination - time_to_passenger # in hours
        d1 = wait_time + time_to_destination

        driver_profit[driver['ID']] += profit_time
        driver_n_trips[driver['ID']] += 1

        available_time = match_time + timedelta(hours=(time_to_passenger + time_to_destination)) # datetime object
        drivers = reinsert_driver(drivers, closest_driver, available_time, passenger_dropoff_node)

        wait_times.append(wait_time)
        profit_times.append(profit_time)
        d1_times.append(d1)
        match_times.append(match_wait_hours)
        end = time.time()
        avg_runtime = avg_runtime*(total_n-1) + (end-start)
        avg_runtime /= total_n

    average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
    average_profit_time = sum(profit_times) / len(profit_times) if profit_times else 0
    average_d1_time = sum(d1_times) / len(d1_times) if d1_times else 0
    average_match_time = sum(match_times) / len(match_times) if match_times else 0
    return average_wait_time, average_profit_time, average_d1_time, driver_profit, driver_n_trips, average_match_time


#%%
# Load data and print length of Data Load
start = time.time()
adjacency_list = load_json("adjacency.json")
node_data = load_json("node_data.json")
drivers_data = load_csv("drivers.csv")
passengers_data = load_csv("passengers.csv")
end = time.time()
print(f'Data Load Time: {(end - start)/60.0: .3f} minutes')
#%%
drivers_t4 = copy.deepcopy(drivers_data)
passengers_t4 = copy.deepcopy(passengers_data)

#IDEA 2: get range of nodes nearest to target by lat and long, then find nn and terminate when within range
n_lat = list(node_data.keys())
n_lon = list(node_data.keys())
n_lat.sort(key=lambda x: node_data[x]['lat']) # NODE IDs of nodes sorted by lat
n_lon.sort(key=lambda x: node_data[x]['lon']) # NODE IDs of nodes sorted by lon

start = time.time()
id = 0
for d in drivers_t4:
    dlat, dlon = float(d['Source Lat']), float(d['Source Lon'])
    d['Node'] = fnn_est(dlat, dlon, n_lat, n_lon, node_data)
    d['ID'] = id  
    id += 1
for p in passengers_t4:
    pslat, pslon = float(p['Source Lat']), float(p['Source Lon'])
    pdlat, pdlon = float(p['Dest Lat']), float(p['Dest Lon'])
    p['Source Node'] = fnn_est(pslat, pslon, n_lat, n_lon, node_data)
    p['Dest Node'] = fnn_est(pdlat, pdlon, n_lat, n_lon, node_data)
end = time.time()
print(f'Finding Estimated Nearest Nodes of all Drivers/Passengers: {(end-start)/60.0: .3f} minutes')
#%%
# MACM T4
start_time = time.time()
average_wait_time, average_profit_time, average_d1_time, driver_profit, driver_n_trips, average_match_time = macm_b2(copy.deepcopy(drivers_t4), copy.deepcopy(passengers_t4), graph, node_data,1)
end_time = time.time()

print(f"Average Wait Time for Passengers: {average_wait_time} hours")
print(f"Average D1 Time: {average_d1_time} hours")
print(f"Average Match Time for Passengers: {average_match_time} hours")
print(f"Average Profit Time for Drivers (D2): {average_profit_time} hours")
print(f"Runtime (excluding loading data): {(end_time - start_time)/60.0} minutes")