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

def macm_t1(drivers, passengers, graph, nodes):
    wait_times = []
    profit_times = []

    drivers.sort(key=lambda x: x['Date/Time'])
    passengers.sort(key=lambda x: x['Date/Time'])

    while drivers and passengers:
        driver = drivers.pop(0)
        passenger = passengers.pop(0)

        driver_node = driver['Node']
        passenger_pickup_node = passenger['Source Node']
        passenger_dropoff_node = passenger['Dest Node']

        driver_time = driver['Date/Time']
        passenger_time = passenger['Date/Time']

        match_time = max(driver_time, passenger_time)  # datetime object

        match_wait_time = match_time - passenger_time  # datetime object
        match_wait_hours = match_wait_time.total_seconds() / 3600.0  # wait time in hours

        time_to_passenger = calculate_route_time(driver_node, passenger_pickup_node, graph, match_time)  # in hours
        time_to_destination = calculate_route_time(passenger_pickup_node, passenger_dropoff_node, graph,
                                                   match_time)  # in hours

        wait_time = match_wait_hours + time_to_passenger  # in hours
        profit_time = time_to_destination - time_to_passenger  # in hours

        available_time = match_time + timedelta(hours=(time_to_passenger + time_to_destination))  # datetime object
        drivers = reinsert_driver(drivers, driver, available_time, passenger_dropoff_node)

        wait_times.append(wait_time)
        profit_times.append(profit_time)

    average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
    average_profit_time = sum(profit_times) / len(profit_times) if profit_times else 0

    return average_wait_time, average_profit_time

def macm_t2(drivers, passengers, graph, nodes):
    wait_times = []
    profit_times = []
    drivers.sort(key=lambda x: x['Date/Time'])
    passengers.sort(key=lambda x: x['Date/Time'])

    while drivers and passengers:
        passenger = passengers.pop(0)
        passenger_time = passenger['Date/Time']
        passenger_pickup_node = passenger['Source Node']
        passenger_dropoff_node = passenger['Dest Node']

        source_lat, source_lon = float(nodes[passenger_pickup_node]['lat']), float(nodes[passenger_pickup_node]['lon'])

        # BRUTE FORCE
        driver = drivers.pop(0)
        available_drivers = [driver]
        t = max(driver['Date/Time'], passenger_time)

        i = 1
        while passenger_time >= t and i < len(drivers):
            driver = drivers.pop(i)
            i+=1
            available_drivers.append(driver)

        min_dist = float('infinity')
        closest_driver = None
        for d in available_drivers:
            d_lat, d_lon = float(d['Source Lat']), float(d['Source Lon'])
            dist = haversine(source_lat, source_lon, d_lat, d_lon)
            if dist < min_dist:
                min_dist = dist
                closest_driver = d

        driver_node = closest_driver['Node']
        driver_time = closest_driver['Date/Time']

        match_time = max(driver_time, passenger_time)   # datetime object

        match_wait_time = match_time - passenger_time   # datetime object
        match_wait_hours = match_wait_time.total_seconds()/3600.0 # wait time in hours

        time_to_passenger = calculate_route_time(driver_node, passenger_pickup_node, graph, match_time) # in hours
        time_to_destination = calculate_route_time(passenger_pickup_node, passenger_dropoff_node, graph, match_time) # in hours

        wait_time = match_wait_hours + time_to_passenger # in hours
        profit_time = time_to_destination - time_to_passenger # in hours

        available_time = match_time + timedelta(hours=(time_to_passenger + time_to_destination)) # datetime object
        drivers = reinsert_driver(drivers, closest_driver, available_time, passenger_dropoff_node)

        wait_times.append(wait_time)
        profit_times.append(profit_time)

    average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
    average_profit_time = sum(profit_times) / len(profit_times) if profit_times else 0

    return average_wait_time, average_profit_time

def macm_t3(drivers, passengers, graph, nodes):
    wait_times = []
    profit_times = []

    drivers.sort(key=lambda x: x['Date/Time'])
    passengers.sort(key=lambda x: x['Date/Time'])

    while drivers and passengers:
        passenger = passengers.pop(0)
        passenger_time = passenger['Date/Time']
        passenger_pickup_node = passenger['Source Node']
        passenger_dropoff_node = passenger['Dest Node']

        source_lat, source_lon = float(nodes[passenger_pickup_node]['lat']), float(nodes[passenger_pickup_node]['lon'])

        # BRUTE FORCE
        driver = drivers.pop(0)
        available_drivers = [driver]
        t = max(driver['Date/Time'], passenger_time)

        i = 1
        while passenger_time >= t and i < len(drivers):
            driver = drivers.pop(i)
            i+=1
            available_drivers.append(driver)

        min_ttp = float('infinity')
        closest_driver = None
        for d in available_drivers:
            d_t, d_node, d_lat, d_lon = d['Date/Time'], d['Node'], d['Source Lat'], d['Source Lon']
            ttp = calculate_route_time(d_node, passenger_pickup_node, graph, max(d_t, passenger_time))
            if ttp < min_ttp:
                min_ttp = ttp
                closest_driver = d

        driver_node = closest_driver['Node']
        driver_time = closest_driver['Date/Time']

        match_time = max(driver_time, passenger_time)   # datetime object

        match_wait_time = match_time - passenger_time   # datetime object
        match_wait_hours = match_wait_time.total_seconds()/3600.0 # wait time in hours

        time_to_passenger = calculate_route_time(driver_node, passenger_pickup_node, graph, match_time) # in hours
        time_to_destination = calculate_route_time(passenger_pickup_node, passenger_dropoff_node, graph, match_time) # in hours

        wait_time = match_wait_hours + time_to_passenger # in hours
        profit_time = time_to_destination - time_to_passenger # in hours

        available_time = match_time + timedelta(hours=(time_to_passenger + time_to_destination)) # datetime object
        drivers = reinsert_driver(drivers, closest_driver, available_time, passenger_dropoff_node)

        wait_times.append(wait_time)
        profit_times.append(profit_time)

    average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
    average_profit_time = sum(profit_times) / len(profit_times) if profit_times else 0

    return average_wait_time, average_profit_time