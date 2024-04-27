# coding: utf-8

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, box
from shapely.ops import transform
import pyproj
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
import os
import folium
from folium.plugins import MarkerCluster, MeasureControl, HeatMap
from folium.raster_layers import ImageOverlay
from scipy.interpolate import interp1d
from scipy.ndimage import generic_filter
from scipy.signal import convolve2d
from dotenv import load_dotenv
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.mask import mask
import csv

from gtfs4ev.gtfsfeed import GTFSFeed
from gtfs4ev.tripsim import TripSim
from gtfs4ev.trafficsim import TrafficSim
from gtfs4ev.topology import Topology
from gtfs4ev import helpers as hlp

"""
Main function
"""
def main():

	##########################
	# GTFS Feed initialization 
	##########################

	# Load environment variables (do not comment!)
	load_dotenv() # take environment variables from .env
	INPUT_PATH = str(os.getenv("INPUT_PATH"))
	OUTPUT_PATH = str(os.getenv("OUTPUT_PATH"))

	# Populate the feed with the raw data (do not comment!)
	feed = GTFSFeed("GTFS_Nairobi")

	# feed.general_feed_info() # Info on raw data

	# OPTIONNAL If needed, filter out trips belonging to specific services and/or agencies

	# -> Abidjan: keep only gbaka minibus taxis
	# feed.filter_agency('monbus', clean_all = True) 
	# feed.filter_agency('monbus / Navette', clean_all = True) 
	# feed.filter_agency('Express', clean_all = True) 
	# feed.filter_agency('Wibus', clean_all = True) 
	# feed.filter_agency('STL', clean_all = True) 
	# feed.filter_agency('monbato', clean_all = True) 
	# feed.filter_agency('Woro-woro de Cocody', clean_all = True) 
	# feed.filter_agency('Woro-woro de Treichville', clean_all = True) 
	# feed.filter_agency('Woro-woro de Yopougon', clean_all = True) 
	# feed.filter_agency('Woro-woro d\'Adjamé', clean_all = True) 
	# feed.filter_agency('Aqualines', clean_all = True) 
	# feed.filter_agency('Woro-woro de Port-Bouët', clean_all = True) 
	# feed.filter_agency('Woro-woro d\'Abobo', clean_all = True) 
	# feed.filter_agency('Woro-woro de Koumassi', clean_all = True) 
	# feed.filter_agency('Woro-woro de Marcory', clean_all = True) 
	# feed.filter_agency('Woro-woro d\'Attecoubé', clean_all = True) 
	# feed.filter_agency('Woro-woro de Bingerville', clean_all = True)

	# -> Alexandria: drop buses
	# feed.filter_agency('Bus', clean_all = True)

	# -> Cairo: drop the public formal transport
	# feed.filter_agency('CTA', clean_all = True)
	# feed.filter_agency('CTA_M', clean_all = True)

	# -> Freetown: keep only weekdays and poda-podas
	# feed.filter_services('service_0001', clean_all = True) 
	# feed.filter_services('service_0003', clean_all = True)
	# feed.filter_agency('Freetown_SLRTC_03', clean_all = True)
	# feed.filter_agency('Freetown_Tagrin_Ferry_01', clean_all = True)
	# feed.filter_agency('Freetown_Taxi_Cab_04', clean_all = True)

	# -> Harare: drop the weekends
	# feed.filter_services('service_0001', clean_all = True)

	# -> Kampala: drop buses
	# feed.filter_agency('bus', clean_all = True)

	##########################
	# GTFS Feed pre-processing 
	##########################

	# Check data consistency, and perform cleaning if needed
	feed.check_all()
	if not feed.check_all():
		feed.clean_all() # Data cleaning to get a consistent feed
		feed.check_all() # Re-check data consistency

	# Optionnal: snap trip shapes to OSM road network
	# feed.snap_shapes_to_osm() # Takes a lot of time

	# feed.check_all()	

	###############################################
	# 1. Display general information about the feed
	###############################################

	feed.general_feed_info()
	print(feed.simulation_area_km2())

	######################################
	# 2. Visualize the paratransit network
	######################################

	""" Visualize a trip using the trip_id """
	trip_id = 'TI1576360557'

	# Get the shape of the trip_id
	# shape = feed.get_shape(trip_id) 

	# # Extract coordinates from the LineString
	# coordinates = list(shape.coords)

	# # Create a Folium Map centered at the midpoint of the LineString
	# midpoint = (shape.centroid.y, shape.centroid.x)
	# mymap = folium.Map(location=midpoint, zoom_start=12)

	# # Add the LineString to the map
	# folium.PolyLine(locations=[[coord[1], coord[0]] for coord in shape.coords], color='blue').add_to(mymap)

	# # Save the map to an HTML file
	# mymap.save(f"{OUTPUT_PATH}/trip_{trip_id}.html")	

	# # """ Add the stops to the path """
	# # Get the stop locations
	# stop_coordinates = feed.get_stop_locations(trip_id) 

	# # Add the Points to the map
	# for point in stop_coordinates:
	#     folium.Marker([point.y, point.x], icon=folium.Icon(color='red')).add_to(mymap)

	# # Save the map to an HTML file
	# mymap.save(f"{OUTPUT_PATH}/trip_{trip_id}_withstops.html")

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
	# mymap.save(f"{OUTPUT_PATH}/stop_frequencies.html")

	# """ Visualize all trips """
	# Warning: could take a great amount of time

	# print(feed.trips)

	# mymap = folium.Map(location=(feed.get_shape('BOX_6O32_O (15-00-00)').centroid.y, feed.get_shape('BOX_6O32_O (15-00-00)').centroid.x), zoom_start=12, control_scale = True)

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
	#     mymap.save(f"{OUTPUT_PATH}/all_trips_Nairobi.html")	
	#     print(trip_id)

	############################################
	# 3. Extract global metrics of the GTFS Feed
	############################################

	# print(f"Simulation area: {feed.simulation_area_km2()} km2")
	# print(feed.trip_statistics())
	# print(feed.stop_statistics())

	# # Average distance between stops along the trips
	# dist = feed.ave_distance_between_stops_all(False)
	# weighted_average = np.average(dist['stop_dist_km'], weights=dist['n_stops'])

	# print(f"Average distance between stops: {weighted_average} km")

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

	# df.to_csv("output/Kampala_operational_metrics.csv", index=False)

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

	# # # print(trips)
	# ev_con = [0.40] * len(trips)

	# traffic_sim = TrafficSim(feed, trips, ev_con)

	# print(traffic_sim.operation_estimates().sum())
	# print(traffic_sim.vehicle_statistics().mean())

	# df = traffic_sim.profile(start_time = "00:00:00", stop_time = "23:59:59", time_step = 50, transient_state = False)

	# df.to_csv("output/Kampala_profile_50s_0h-23h59m59s.csv", index = False)

	# # Plot the function
	# plt.plot(df['t'], df['power_kW'], marker='o')
	# plt.xlabel('Time (seconds)')
	# plt.ylabel('Power')
	# plt.title('Power vs. Time')
	# plt.grid(True)

	# # plt.savefig('power_vs_time.png')
	# plt.show()

	##########################################################################################
	# 10. Visualize the activity of the fleet (aver. number of vehicles) in a given time frame
	##########################################################################################

	# Warning: could take a great amount of time

	# start_time = "12:00:00"
	# stop_time = "12:20:00"

	# print(feed.trips)

	# # Create a Folium Map centered at a specific location
	# mymap = folium.Map(location=(feed.get_shape('1107D110').centroid.y, feed.get_shape('1107D110').centroid.x), zoom_start=12, control_scale=True)

	# # Initialize a list to store the trip coordinates and values
	# trip_data = []

	# for index, row in feed.trips.iterrows():
	#     trip_id = row['trip_id']

	#     # Get the shape of the trip_id
	#     shape = feed.get_shape(trip_id) 

	#     # Extract coordinates from the LineString
	#     coordinates = list(shape.coords)

	#     # Extract value associated with the trip
	#     trip_sim = TripSim(feed = feed, trip_id=trip_id, ev_consumption = 0.4)
	#     trip_sim.simulate_vehicle_fleet(start_time, stop_time, 20, transient_state = False)

	#     # operation_estimates = trip_sim.operation_estimates_aggregated()
	#     # value = operation_estimates['ave_nbr_vehicles']

	#     value = trip_sim.trip_profile['n_vehicles'].mean()
	#     print(value)

	#     # Append trip coordinates and value to the trip_data list
	#     trip_data.extend([[coord[1], coord[0], value] for coord in shape.coords])

	# # Create a HeatMap layer with the trip data
	# heatmap = HeatMap(trip_data, radius=10)

	# # Add the HeatMap layer to the map
	# heatmap.add_to(mymap)

	# # Save the map to an HTML file
	# mymap.save(f"{OUTPUT_PATH}/activity_Nairobi_noon.html")

	##################
	# 11. GIS air pollution analysis
	##################

	city = "Nairobi"

	# Pre-processing. Comment if already done, crop the population raster to the bounding box
	##################
	hlp.crop_raster(str(os.getenv("INPUT_PATH")) + "/Population_rasters/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R10_C22.tif", 
		feed.bounding_box(), 
		str(os.getenv("INPUT_PATH")) + f"/Population_rasters/{city}_pop.tif")

	# Compute the local emission index map, based on the cropped population data as a blueprint for the output raster. Comment to avoid recomputing. 
	##################
	# Load the cropped population raster and calculate the total population
	# ref_raster = str(os.getenv("INPUT_PATH")) + f"/Population_rasters/{city}_pop.tif"

	# with rasterio.open(ref_raster) as src:
	#     # Read the raster data
	#     raster_data = src.read(1)  # assuming it's a single band raster
	    
	#     # Calculate the sum of all values
	#     raster_sum = np.sum(raster_data)

	# print("Total population:", raster_sum)

	# trips = list(feed.trips['trip_id']) # Trips to consider
	# ev_con = [0.40] * len(trips)

	# traffic_sim = TrafficSim(feed, trips, ev_con) # Carry out the simulation for all trips
	# op = traffic_sim.operation_estimates() # Get operation estimates

	# vkm_list = op['vkm'].tolist() # VKM of trips
	# linestring_list = [feed.get_shape(row['trip_id']) for index, row in op.iterrows()] # Associated linestrings

	# hlp.local_emission_index(vkm_list, linestring_list, ref_raster, str(os.getenv("OUTPUT_PATH")) + f"/{city}_local_em.tif")

	# Compute the exposure index. Comment to avoid recomputing. 
	##################
	# with rasterio.open(str(os.getenv("OUTPUT_PATH")) + f"/{city}_local_em.tif") as src:
	#     # Read the raster data
	#     raster_data = src.read(1).astype(float) # assuming it's a single band raster
	    
	#     # Define the convolution kernel (e.g., exponential decay)
	#     kernel_size = 21 # 5 to left, 5 to the right + current pixel
	#     decay_factor = 0.64  # NO2 = 0.0064 per meter, so 0,64 per pixel | 0.02 in some other refs
	#     kernel = hlp.exponential_decay_kernel(kernel_size, decay_factor)

	#     # # Define the mask for the 5-pixel (500m) radius
	#     # mask = hlp.mask_within_radius(kernel_size, radius=(kernel_size-1)/2)
	    
	#     # # Apply the mask to the kernel
	#     # kernel = kernel*mask
	    
	#     # Perform the convolution operation
	#     convolved_data = convolve2d(raster_data, kernel, mode='same', boundary='wrap')
	    
	#     # Create a new raster file with the convolved data
	#     profile = src.profile
	#     profile.update(dtype=rasterio.float32)  # Update data type to float32
	#     with rasterio.open(str(os.getenv("OUTPUT_PATH")) + f"/{city}_exposure_index.tif", 'w', **profile) as dst:
	#         dst.write(convolved_data.astype(rasterio.float32), 1)  # assuming it's a single band raster

	# Get histogram of population counts by exposure. 
	##################

	# # Open the population raster file
	# with rasterio.open(str(os.getenv("INPUT_PATH")) + f"/Population_rasters/{city}_pop.tif") as population_src:
	#     # Read the population raster data
	#     population_data = population_src.read(1)  # assuming it's a single band raster
	    
	#     # Get the raster profile to use for writing the histogram
	#     profile = population_src.profile

	# # Open the property raster file
	# with rasterio.open(str(os.getenv("OUTPUT_PATH")) + f"/{city}_exposure_index.tif") as property_src:
	#     # Read the property raster data
	#     exposure_data = property_src.read(1)  # assuming it's a single band raster

	# # Define the property range for the histogram
	# property_min = 1  # Define your minimum property value
	# property_max = np.max(exposure_data)  # Define your maximum property value

	# # Initialize histogram bins
	# num_bins = 10  # Define the number of bins for the histogram
	# bins = np.linspace(property_min, property_max, num_bins + 1)

	# # Create an empty histogram to store the counts
	# histogram_counts = np.zeros(num_bins)

	# # Iterate over each pixel in the property raster
	# for i in range(exposure_data.shape[0]):
	#     for j in range(exposure_data.shape[1]):
	#         # Get the property value at the current pixel
	#         property_value = exposure_data[i, j]
	        
	#         # Check if the property value is within the specified range
	#         if property_min <= property_value <= property_max:
	#             # If it is, increment the corresponding histogram bin based on the population at this pixel
	#             for k in range(len(bins) - 1):
	#                 if bins[k] <= property_value < bins[k + 1]:
	#                     histogram_counts[k] += population_data[i, j]

	# # Print the histogram counts
	# print("Histogram counts:", histogram_counts)

	# # Plot the histogram
	# plt.bar(bins[:-1], histogram_counts, width=np.diff(bins), align='edge')
	# plt.xlabel('Property Range')
	# plt.ylabel('Population Count')
	# plt.title('Population Distribution Histogram by Property Range')
	# plt.grid(True)
	# plt.show()

	# # Calculate the sum of the histogram values
	# total_population = np.sum(histogram_counts)
	# print("Total population exposed:", total_population)

	# # Write histogram counts to CSV
	# with open(str(os.getenv("OUTPUT_PATH")) + f"/{city}_exposure_distribution.csv", "w", newline="") as csvfile:
	#     writer = csv.writer(csvfile)
	#     writer.writerow(["Exposure Index", "Population Count"])
	#     for i in range(num_bins):
	#         writer.writerow([f"{bins[i]} - {bins[i+1]}", histogram_counts[i]])


	# Population-weighted exposure 
	##################

	# # Open the population raster file
	# with rasterio.open(str(os.getenv("INPUT_PATH")) + f"/Population_rasters/{city}_pop.tif") as population_src:
	#     # Read the population raster data
	#     population_data = population_src.read(1)  # assuming it's a single band raster

	# # Open the property raster file
	# with rasterio.open(str(os.getenv("OUTPUT_PATH")) + f"/{city}_exposure_index.tif") as property_src:
	#     # Read the property raster data
	#     exposure_data = property_src.read(1)  # assuming it's a single band raster

	#     popweighted_exposure = exposure_data * population_data

	#     with rasterio.open(str(os.getenv("OUTPUT_PATH")) + f"/{city}_popweighted_exposure.tif", 'w', **profile) as dst:
	#     	dst.write(popweighted_exposure, 1)  # assuming it's a single band raster






	# # Load the population raster 
	# data_path = str(os.getenv("INPUT_PATH")) + "/Population_rasters/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R10_C22.tif"

	# with rasterio.open(data_path) as src:
	#     out_image, out_transform = rasterio.mask.mask(src, [feed.bounding_box()], crop=True)
	#     out_meta = src.meta

	# out_meta.update({"driver": "GTiff",
    #              "height": out_image.shape[1],
    #              "width": out_image.shape[2],
    #              "transform": out_transform})

	# with rasterio.open("masked.tif", "w", **out_meta) as dest:
	#     dest.write(out_image)    
	
	# raster = rasterio.open("masked.tif")

	# if raster.crs != 'EPSG:4326':
	# 	print("EPSG")

	# # Transform into numpy array
	# data = raster.read()[0] # read raster vals into numpy array
	# data_normed = data/data.max() # normalization to help with color gradient

	# # Get the bounds of the reprojected raster
	# bounds = [[raster.bounds.bottom, raster.bounds.left], [raster.bounds.top, raster.bounds.right]]

	# # Initialize a folium map
	# mymap = folium.Map(location=(feed.get_shape('1107D110').centroid.y, feed.get_shape('1107D110').centroid.x), zoom_start=12, control_scale=True)

	# # Define colors
	# colors = [(0, 0, 0, 0), 'yellow', 'orange', 'red']

	# # Define positions for each color
	# positions = [0.0, 0.1, 0.5, 1.0]

	# # Create the colormap
	# cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, colors)))

	# #Add the raster overlay to the map
	# ImageOverlay(
	#     image=data_normed,
	#     bounds=bounds,
	#     opacity=0.8,
	#     zindex=1,
	#     colormap=cmap,
	# ).add_to(mymap)

	# Add the trips

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
	#     mymap.save(f"{OUTPUT_PATH}/Nairobi_population.html")
	#     print(trip_id)




	# with rasterio.open("masked.tif") as src:
	#     raster_bounds = src.bounds
	#     raster_transform = src.transform
	#     raster_crs = src.crs
	#     raster_array = src.read(1)

	# gdf = feed.shapes['geometry']
	# gdf_list = gdf.tolist()



	# # Create a GeoDataFrame from the list of LineString objects
	# crs = {'init': 'epsg:4326'}  # Define the coordinate reference system (CRS) as needed
	# gdf = gpd.GeoDataFrame(geometry=gdf_list, crs=crs)

	# # Export the GeoDataFrame to a Shapefile
	# gdf.to_file("input/shapefile.shp")

	# # Initialize an array to store counts
	# pixel_counts = np.zeros_like(raster_array)

	# # Iterate over each pixel
	# i = 1
	# for row in range(raster_array.shape[0]):
	#     for col in range(raster_array.shape[1]):
	#         # Create polygon representing the pixel
	#         pixel_box = box(raster_bounds.left + col * raster_transform[0], 
	#                         raster_bounds.bottom + row * raster_transform[4], 
	#                         raster_bounds.left + (col + 1) * raster_transform[0], 
	#                         raster_bounds.bottom + (row + 1) * raster_transform[4])

	#         print(f"{i} out of {raster_array.shape[0]*raster_array.shape[1]}")
	#         # print("CRS of LineString objects:", gdf_list[0].crs)
	#         print("CRS of Pixel Box:", pixel_box.crs)
	#         i = i + 1
	# 		# Iterate over each LineString
	#         for line_string in gdf_list:     	
	#             if line_string.intersects(pixel_box):
	#                 print("Hello")
	#                 pixel_counts[row, col] += 1

	# # Define the output raster file path
	# output_raster_file = "output_raster.tif"

	# # Write the pixel_counts array to a new raster file
	# with rasterio.open(output_raster_file, 'w', driver='GTiff', width=pixel_counts.shape[1], height=pixel_counts.shape[0],
	#                    count=1, dtype=pixel_counts.dtype, crs=raster_crs, transform=raster_transform) as dst:
	#     dst.write(pixel_counts, 1)



	# Chemin des fichiers des rasters
	# path_raster1 = "masked.tif"
	# path_raster2 = "input/nairobi_raster_routes.tif"
	# path_output = "exposure.tif"






if __name__ == "__main__":
	main()


