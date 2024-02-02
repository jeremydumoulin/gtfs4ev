# coding: utf-8

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyproj
import matplotlib.pyplot as plt
import numpy as np
import math
import folium
from folium.plugins import MarkerCluster, MeasureControl

from gtfs4ev.gtfsfeed import GTFSFeed
from gtfs4ev.tripsim import TripSim
from gtfs4ev.trafficsim import TrafficSim
from gtfs4ev.topology import Topology

from gtfs4ev import constants as cst
from gtfs4ev import environment as env
from gtfs4ev import helpers as hlp

"""
Main function
"""
def main():

	############################################
	# GTFS Feed initialization and preprocessing
	############################################

	# Populate the feed with the raw data (do not comment!)
	feed = GTFSFeed("GTFS_Freetown")

	feed.general_feed_info() # General information before data cleaning

	# Check data consistency, and perform cleaning if needed
	feed.check_all()
	if not feed.check_all():
		feed.clean_all() # Data cleaning to get a consistent feed
		feed.check_all() # Re-check data consistency

	# Optionnal. If needed, filter out trips belonging to specific services	
	# For example, this is needed fo Freetown, as service 0001 is for weekends and 0003 for ferrys
	feed.filter_services('service_0001', clean_all = True) 
	feed.filter_services('service_0003', clean_all = True)

	feed.check_all()

	###############################################
	# 1. Display general information about the feed
	###############################################

	feed.general_feed_info()

	######################################
	# 2. Visualize the paratransit network
	######################################

	# """ Visualize a trip using the trip_id """
	# trip_id = '1107D110'

	# # Get the shape of the trip_id
	# shape = feed.get_shape(trip_id) 

	# # Extract coordinates from the LineString
	# coordinates = list(shape.coords)

	# # Create a Folium Map centered at the midpoint of the LineString
	# midpoint = (shape.centroid.y, shape.centroid.x)
	# mymap = folium.Map(location=midpoint, zoom_start=12)

	# # Add the LineString to the map
	# folium.PolyLine(locations=[[coord[1], coord[0]] for coord in shape.coords], color='blue').add_to(mymap)

	# # Save the map to an HTML file
	# mymap.save(f"{env.OUTPUT_PATH}/trip_{trip_id}.html")	

	# """ Add the stops to the path """
	# # Get the stop locations
	# stop_coordinates = feed.get_stop_locations(trip_id) 

	# # Add the Points to the map
	# for point in stop_coordinates:
	#     folium.Marker([point.y, point.x], icon=folium.Icon(color='red')).add_to(mymap)

	# # Save the map to an HTML file
	# mymap.save(f"{env.OUTPUT_PATH}/trip_{trip_id}_withstops.html")

	# """ Map the stop frequencies """
	# df = feed.stop_frequencies()

	# # Create a Folium Map centered around the first point
	# first_point = df['geometry'].iloc[0]
	
	# mymap = folium.Map(location=[first_point.y, first_point.x], zoom_start=12)

	# # Add CircleMarkers for each point with size based on 'count'
	# for index, row in df.iterrows():
	#     point_coords = row['geometry']
	#     count_value = row['count']
	#     stop_id = row['stop_id']

	#     # Create a Popup with the 'count' value
	#     popup_content = f"Count: {count_value} \n Stop id: {stop_id} \n"
	#     popup = folium.Popup(popup_content, max_width=300)

	#     # Get the corresponding color from the color scale
	#     color_scale = folium.LinearColormap(colors=['blue', 'red'], vmin=df['count'].min(), vmax=df['count'].max())

	#     fill_color = color_scale(count_value)

	#     folium.CircleMarker(location=(point_coords.y, point_coords.x),
	#                         radius=count_value,  # Adjust the scale factor as needed
	#                         color='none',
	#                         fill=True,
	#                         fill_color=fill_color,
	#                         fill_opacity=0.6,
	#                         popup=popup).add_to(mymap)

	# color_scale.add_to(mymap)

	# # Add MeasureControl for the scale
	# mymap.add_child(MeasureControl(primary_length_unit='kilometers'))

	# # Save the map to an HTML file
	# mymap.save(f"{env.OUTPUT_PATH}/stop_frequencies.html")

	# """ Visualize all trips """
	# Warning: could take a great amount of time
	# print(feed.trips)

	# mymap = folium.Map(location=(feed.get_shape('trip_0001').centroid.y, feed.get_shape('trip_0001').centroid.x), zoom_start=12, control_scale = True)

	# for index, row in feed.trips.iterrows():
	#     trip_id = row['trip_id']

	#     # Get the shape of the trip_id
	#     shape = feed.get_shape(trip_id) 

	#     # Extract coordinates from the LineString
	#     coordinates = list(shape.coords)

	#     # Create a Folium Map centered at the midpoint of the LineString
	#     midpoint = (shape.centroid.y, shape.centroid.x)		

	#     # Add the LineString to the map
	#     folium.PolyLine(locations=[[coord[1], coord[0]] for coord in shape.coords], color='blue').add_to(mymap)

	#     # Save the map to an HTML file
	#     mymap.save(f"{env.OUTPUT_PATH}/all_trips_Freetown.html")	
	#     print(trip_id)

	############################################
	# 3. Extract global metrics of the GTFS Feed
	############################################

	print(f"Simulation area: {feed.simulation_area_km2()} km2")
	print(feed.trip_statistics())
	print(feed.stop_statistics())

	# Average distance between stops along the trips
	dist = feed.ave_distance_between_stops_all(False)
	weighted_average = np.average(dist['stop_dist_km'], weights=dist['n_stops'])

	print(f"Average distance between stops: {weighted_average} km")

	# ############################################
	# 4. Extract topological information 
	############################################

	# tp = Topology(feed)

	# print(tp.trip_crossovers())
	# print(tp.nearest_point_distance_km())

	# print(feed.trip_crossovers())
	# print(feed.nearest_point_distance_km()) # Warning: takes a long time

	#########################
	# 5. Operationnal metrics  
	#########################

	""" For a single trip """

	# trip_sim = TripSim(feed = feed, trip_id='20121111', ev_consumption = 0.2)

	# print(trip_sim.trip_duration_sec)
	# print(trip_sim.trip_length_km)

	# print(trip_sim.operation_estimates())
	# print(trip_sim.operation_estimates_aggregated())

	""" For the whole system """

	# trips = feed.trips	
	
	# df = pd.DataFrame()

	# for index, row in trips.iterrows():
	# 	trip_id = row['trip_id']
	# 	trip_sim = TripSim(feed = feed, trip_id=trip_id, ev_consumption = 0.4)

	# 	trip_stats = trip_sim.operation_estimates_aggregated()
	# 	df = pd.concat([df, pd.DataFrame([trip_stats])], ignore_index=True)
	
	# print(df)

	# print(df['ave_nbr_vehicles'].sum())
	# print(df['n_trips'].sum())
	# print(df['vkm'].sum())
	# print(df['energy_kWh'].sum())

	# print(df['vkt'].mean())
	# print(df['energy_kWh_per_vehicle'].mean())

	# df.to_csv("output/Nairobi_simulation.csv", index=False)

	##########################################################################################
	# 6. Power/energy/speed profile of a single vehicle along a trip and associated statistics 
	##########################################################################################

	# trip_sim = TripSim(feed = feed, trip_id='20121111', ev_consumption = 0.4)

	# time_values = np.arange(0, trip_sim.trip_duration_sec, 20) 
	# values = [trip_sim.power_profile(t) for t in time_values]
	# # values = [trip_sim.energy_profile(t) for t in time_values]
	# # values = [trip_sim.speed_profile(t) for t in time_values]

	# # Plot the function
	# plt.plot(time_values, values, marker='o')
	# plt.xlabel('Time (seconds)')
	# plt.ylabel('Power')
	# plt.title('Power vs. Time')
	# plt.grid(True)

	# # plt.savefig('power_vs_time.png')
	# plt.show()

	# print(trip_sim.vehicle_statistics())

	###########################################
	# 7. Profile of the vehicle fleet on a trip 
	###########################################

	# trip_sim = TripSim(feed = feed, trip_id='20121111', ev_consumption = 0.4)

	# trip_sim.simulate_vehicle_fleet(start_time = "05:00:00", stop_time = "22:00:00", time_step = 30, transient_state = False)

	# print(trip_sim.trip_profile)

	# # Plot the function
	# plt.plot(trip_sim.trip_profile['t'], trip_sim.trip_profile['power_kW'], marker='o')
	# plt.xlabel('Time (seconds)')
	# plt.ylabel('Power')
	# plt.title('Power vs. Time')
	# plt.grid(True)

	# # plt.savefig('power_vs_time.png')
	# plt.show()

	##############################
	# 8. Profile of a set of trips 
	##############################

	# trips = ['1107D110', '1107D111', '10114110']
	# ev_con = [0.4, 0.4, 0.4]

	# traffic_sim = TrafficSim(feed, trips, ev_con)

	# print(traffic_sim.operation_estimates().sum())
	# df = traffic_sim.profile(start_time = "05:00:00", stop_time = "23:00:00", time_step = 50, transient_state = True)

	# # Plot the function
	# plt.plot(df['t'], df['power_kW'], marker='o')
	# plt.xlabel('Time (seconds)')
	# plt.ylabel('Power')
	# plt.title('Power vs. Time')
	# plt.grid(True)

	# # plt.savefig('power_vs_time.png')
	# plt.show()

	###########################################
	# 9. Profile of a the whole traffic network
	###########################################

	# trips = list(feed.trips['trip_id'])

	# print(trips)
	# ev_con = [0.4] * len(trips)

	# traffic_sim = TrafficSim(feed, trips, ev_con)

	# print(traffic_sim.operation_estimates().sum())
	# df = traffic_sim.profile(start_time = "00:00:00", stop_time = "23:59:59", time_step = 20, transient_state = False)

	# df.to_csv("output/Freetown_profile_20s_0h-23h59m59s.csv", index=False)

	# # Plot the function
	# plt.plot(df['t'], df['power_kW'], marker='o')
	# plt.xlabel('Time (seconds)')
	# plt.ylabel('Power')
	# plt.title('Power vs. Time')
	# plt.grid(True)

	# # plt.savefig('power_vs_time.png')
	# plt.show()
		

if __name__ == "__main__":
	main()