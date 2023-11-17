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

def reinsert_driver(drivers, driver, available):
    prob = 0.01
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
    drivers.insert(left, driver)

    return drivers

def match_and_calculate_metrics(drivers, passengers, graph, node_data):
    wait_times = []
    profit_times = []
    total_distances = [] 

    while drivers and passengers:
        min_distance = float('inf')
        selected_driver = None
        selected_passenger = None
        selected_index = None

        for i, passenger in enumerate(passengers):
            passenger_lat, passenger_lon = float(passenger['Source Lat']), float(passenger['Source Lon'])

            for driver in drivers:
                driver_lat, driver_lon = float(driver['Source Lat']), float(driver['Source Lon'])

                distance = haversine(driver_lat, driver_lon, passenger_lat, passenger_lon)

                if distance < min_distance:
                    min_distance = distance
                    selected_driver = driver
                    selected_passenger = passenger
                    selected_index = i

        if selected_driver and selected_passenger:
            drivers.remove(selected_driver)
            passengers.pop(selected_index)

            driver_node = find_nearest_node(float(selected_driver['Source Lat']), float(selected_driver['Source Lon']), node_data)
            passenger_pickup_node = find_nearest_node(float(selected_passenger['Source Lat']), float(selected_passenger['Source Lon']), node_data)
            passenger_dropoff_node = find_nearest_node(float(selected_passenger['Dest Lat']), float(selected_passenger['Dest Lon']), node_data)

            current_time = selected_driver['Date/Time']
            
            time_to_passenger = calculate_route_time(driver_node, passenger_pickup_node, graph, current_time)
            time_to_destination = calculate_route_time(passenger_pickup_node, passenger_dropoff_node, graph, current_time)

            wait_time = time_to_passenger
            profit_time = time_to_destination - time_to_passenger

            total_distance = min_distance 

            available_time = current_time + timedelta(hours=(time_to_passenger + time_to_destination))
            drivers = reinsert_driver(drivers, selected_driver, available_time)

            wait_times.append(wait_time)
            profit_times.append(profit_time)
            total_distances.append(total_distance)

    average_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
    average_profit_time = sum(profit_times) / len(profit_times) if profit_times else 0
    average_total_distance = sum(total_distances) / len(total_distances) if total_distances else 0

    return average_wait_time, average_profit_time, average_total_distance


adjacency_list = load_json("adjacency.json")
node_data = load_json("node_data.json")
drivers_data = load_csv("drivers.csv")
passengers_data = load_csv("passengers.csv")

graph = construct_graph(adjacency_list)

start_time = time.time()
average_wait_time, average_profit_time, average_total_distance = match_and_calculate_metrics(drivers_data, passengers_data, graph, node_data)
end_time = time.time()

print("Average Wait Time for Passengers (D1):", average_wait_time, "minutes")
print("Average Profit Time for Drivers (D2):", average_profit_time, "minutes")
print("Average Total Distance (D3):", average_total_distance, "kilometers") 
print(f"Runtime (excluding loading data): {end_time - start_time}")
