# coding: utf-8

""" 
A python script that reproduces the various energy, environmental,
and economic indicators of the analysis of a list of cities

Usage:
- Check that all the input data is available. This includes: 1) GTFS 
Data folder ; 2) Population data raster. Make sur that the area covered 
by the latter is larger than the area covered by the GTFS data. Use the 
GHG_Pop dataset with EPSG 4326 
- Make sur the GTFS data is preprocessed as you which in the 
Run_preprocess_gtfs.py script
- Define the list of cities to run and set the associated input parameters
- Change the global parameters according to your needs 
- Comment/Uncomment the aspects you want/do not want to run
- Run the python script

Output:
- Pickled operation data
- Power/energy profile of the whole fleet
- Cropped population data
- Air pollution maps
- CSV file with aggregated inputs/outputs
"""

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
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Append parent directory

from gtfs4ev.gtfsfeed import GTFSFeed
from gtfs4ev.tripsim import TripSim
from gtfs4ev.trafficsim import TrafficSim
from gtfs4ev.topology import Topology
from gtfs4ev import helpers as hlp

import preprocess_gtfs as pp

#############################################
# PARAMETERS - MODIFY ACCORDING TO YOUR NEEDS
#############################################

"""
City-specific parameters 
"""

cities = [
	# {
	# 	'name': "Nairobi",
	# 	'gtfs_feed': "GTFS_Nairobi",
	# 	'pop_raster': "Nairobi_GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R10_C22.tif",
	# 	'population': 0, # Will not be used if you decide to calculate it using the pop raster layer
	# 	'demand_per_capita': 400, # Yearly demand per capita (kWh)
	# 	'diesel_price': 1.5, # Diesel price (US$/L)
	# 	'diesel_subsidies': 0.1, # Diesel explicit subsidies (US$/L)
	# 	'electricity_price': 0.15, # Electricity price (US$/kWh)
	# 	'electricity_co2_intensity': 0.368 # Electricity CO2 intensity (kgCO2/kWh)
	# },
	# {
	# 	'name': "Freetown",
	# 	'gtfs_feed': "GTFS_Freetown",
	# 	'pop_raster': "Freetown_GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R9_C17.tif",
	# 	'population': 0, # Will not be used if you decide to calculate it using the pop raster layer
	# 	'demand_per_capita': 400, # Yearly demand per capita (kWh)
	# 	'diesel_price': 1.5, # Diesel price (US$/L)
	# 	'diesel_subsidies': 0.1, # Diesel explicit subsidies (US$/L)
	# 	'electricity_price': 0.15, # Electricity price (US$/kWh)
	# 	'electricity_co2_intensity': 0.368 # Electricity CO2 intensity (kgCO2/kWh)
	# },
	# {
	# 	'name': "Abidjan",
	# 	'gtfs_feed': "GTFS_Abidjan",
	# 	'pop_raster': "Freetown_GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R9_C17.tif",
	# 	'population': 0, # Will not be used if you decide to calculate it using the pop raster layer
	# 	'demand_per_capita': 400, # Yearly demand per capita (kWh)
	# 	'diesel_price': 1.5, # Diesel price (US$/L)
	# 	'diesel_subsidies': 0.1, # Diesel explicit subsidies (US$/L)
	# 	'electricity_price': 0.15, # Electricity price (US$/kWh)
	# 	'electricity_co2_intensity': 0.368 # Electricity CO2 intensity (kgCO2/kWh)
	# },
	# {
	# 	'name': "Accra",
	# 	'gtfs_feed': "GTFS_Accra",
	# 	'pop_raster': "Freetown_GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R9_C17.tif",
	# 	'population': 0, # Will not be used if you decide to calculate it using the pop raster layer
	# 	'demand_per_capita': 400, # Yearly demand per capita (kWh)
	# 	'diesel_price': 1.5, # Diesel price (US$/L)
	# 	'diesel_subsidies': 0.1, # Diesel explicit subsidies (US$/L)
	# 	'electricity_price': 0.15, # Electricity price (US$/kWh)
	# 	'electricity_co2_intensity': 0.368 # Electricity CO2 intensity (kgCO2/kWh)
	# },
	# {
	# 	'name': "Alexandria",
	# 	'gtfs_feed': "GTFS_Alexandria",
	# 	'pop_raster': "Freetown_GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R9_C17.tif",
	# 	'population': 0, # Will not be used if you decide to calculate it using the pop raster layer
	# 	'demand_per_capita': 400, # Yearly demand per capita (kWh)
	# 	'diesel_price': 1.5, # Diesel price (US$/L)
	# 	'diesel_subsidies': 0.1, # Diesel explicit subsidies (US$/L)
	# 	'electricity_price': 0.15, # Electricity price (US$/kWh)
	# 	'electricity_co2_intensity': 0.368 # Electricity CO2 intensity (kgCO2/kWh)
	# },
	# {
	# 	'name': "Bamako",
	# 	'gtfs_feed': "GTFS_Bamako",
	# 	'pop_raster': "Freetown_GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R9_C17.tif",
	# 	'population': 0, # Will not be used if you decide to calculate it using the pop raster layer
	# 	'demand_per_capita': 400, # Yearly demand per capita (kWh)
	# 	'diesel_price': 1.5, # Diesel price (US$/L)
	# 	'diesel_subsidies': 0.1, # Diesel explicit subsidies (US$/L)
	# 	'electricity_price': 0.15, # Electricity price (US$/kWh)
	# 	'electricity_co2_intensity': 0.368 # Electricity CO2 intensity (kgCO2/kWh)
	# },
	# {
	# 	'name': "Cairo",
	# 	'gtfs_feed': "GTFS_Cairo",
	# 	'pop_raster': "Freetown_GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R9_C17.tif",
	# 	'population': 0, # Will not be used if you decide to calculate it using the pop raster layer
	# 	'demand_per_capita': 400, # Yearly demand per capita (kWh)
	# 	'diesel_price': 1.5, # Diesel price (US$/L)
	# 	'diesel_subsidies': 0.1, # Diesel explicit subsidies (US$/L)
	# 	'electricity_price': 0.15, # Electricity price (US$/kWh)
	# 	'electricity_co2_intensity': 0.368 # Electricity CO2 intensity (kgCO2/kWh)
	# },
	# {
	# 	'name': "Harare",
	# 	'gtfs_feed': "GTFS_Harare",
	# 	'pop_raster': "Freetown_GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R9_C17.tif",
	# 	'population': 0, # Will not be used if you decide to calculate it using the pop raster layer
	# 	'demand_per_capita': 400, # Yearly demand per capita (kWh)
	# 	'diesel_price': 1.5, # Diesel price (US$/L)
	# 	'diesel_subsidies': 0.1, # Diesel explicit subsidies (US$/L)
	# 	'electricity_price': 0.15, # Electricity price (US$/kWh)
	# 	'electricity_co2_intensity': 0.368 # Electricity CO2 intensity (kgCO2/kWh)
	# },
	{
		'name': "Kampala",
		'gtfs_feed': "GTFS_Kampala",
		'pop_raster': "Freetown_GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R9_C17.tif",
		'population': 0, # Will not be used if you decide to calculate it using the pop raster layer
		'demand_per_capita': 400, # Yearly demand per capita (kWh)
		'diesel_price': 1.5, # Diesel price (US$/L)
		'diesel_subsidies': 0.1, # Diesel explicit subsidies (US$/L)
		'electricity_price': 0.15, # Electricity price (US$/kWh)
		'electricity_co2_intensity': 0.368 # Electricity CO2 intensity (kgCO2/kWh)
	}
]

"""
Global parameters 
"""

# General
output_folder_name = "res_all_cities"
snap_to_osm_roads = False # Could take a long time. Data is generally already consistent with OSM network
reuse_traffic_output = True # If True, serializes the dataframe with operationnal data in order to avopid recomputing TrafficSimulation
active_working_days = 260 # Number of operating days a year of the minibus taxis
pop_from_raster = True # If True, estimates the number of people using the cropped bbox and population raster
time_step = 100 # Time step in seconds for the power/energy profile

# Energy, economy, environmental implications 
ev_consumption = 0.4 # EV consumption (kWh/km) - Value should not affect the output
charging_efficiency = 0.9 # Loss during charging process

diesel_consumption = 0.1 # Diesel consumption (L/km)
diesel_co2_intensity = 2.7 # Diesel CO2 intensity (kgCO2/L)

decay_rate = 0.0064 # Exponential decay rate (1/m)
buffer_distance = 300 # Buffer ditance in meters
low_threshold = 1  # Minimum treshold to consider population exposed to TRAP (km)
medium_threshold = 200 # Medium threshold for population exposure to TRAP (km)
high_threshold = 600 # High threshold for population exposure to TRAP (km) - Corresponds roughly to an exposure of 0.5 hour to the emission of a 100m road segment (when located at the middle of it) of a busy road 
#(5*3600*(30/60)*100)*exp(-0.0064*50)/1000 (5 vehicles per second)

"""
Environment variables
"""
load_dotenv() # take environment variables from .env

INPUT_PATH = str(os.getenv("INPUT_PATH"))
OUTPUT_PATH = f"{str(os.getenv("OUTPUT_PATH"))}/{output_folder_name}"

################################
# WRITE INPUT PARAMETERS TO FILE
################################

# Define the file path
filename = "_INPUTS.json"

general_parameters = {
    "snap_to_osm_roads": snap_to_osm_roads,
    "reuse_traffic_output": reuse_traffic_output,
    "active_working_days": active_working_days,
    "pop_from_raster": pop_from_raster,
    "time_step": time_step,
    "ev_consumption": ev_consumption,
    "charging_efficiency": charging_efficiency,
    "diesel_consumption": diesel_consumption,
    "diesel_co2_intensity": diesel_co2_intensity,
    "decay_rate": decay_rate,
    "buffer_distance": buffer_distance,
    "low_threshold": low_threshold,
    "medium_threshold": medium_threshold,
    "high_threshold": high_threshold
}

# Combine city data and general_parameters into a single dictionary
final_data = {"data": cities, **general_parameters}

if not os.path.exists(str(OUTPUT_PATH)):
   # Create a new directory because it does not exist
   os.makedirs(str(OUTPUT_PATH))
   print("New directory created to store output")

# Write the list of dictionaries to a JSON file
with open(f"{OUTPUT_PATH}/{filename}", 'w') as json_file:
    json.dump(final_data, json_file, indent=4)

################################
# INIT OUTPUT DATAFRAME & FOLDER
################################

columns = ["City", "Area (km2)", "VKM (km)", "VKT per vehicle (km)", "VKT per trip (km)", "Number of vehicles", "Population", "Baseline demand (kWh/year)", "Additionnal demand (kWh/year)", "Savings per km w/ subsidies ($/km)", "Savings per km w/o subsidies ($/km)", "Average per vehicle savings w/ subsidies (US$/year)", "Average per vehicle savings w/o subsidies (US$/year)", "Diesel emissions (kgCO2/km)", "EV emissions (kgCO2/km)", "Average per vehicle emission reduction (tCO2/year)", "Diesel savings (L/year)", "Total emission reductions (tCO2/year)", "Pop not exposed", "Pop w/ low exposure", "Pop w/ medium exposure", "Pop w/ high exposure"]

out_df = pd.DataFrame(columns=columns)

###########
# MAIN CODE
###########

for city in cities:
	"""
	Message
	"""
	print(f"\n********* RUNNING THE SCRIPT FOR THE CITY OF {city["name"]} ********* \n")

	"""
	Init variables
	"""
	city_name = city["name"]
	gtfs_feed_name = city["gtfs_feed"]
	population_raster_name = city["pop_raster"]
	population = city["population"]
	demand_per_capita = city["demand_per_capita"]
	diesel_price = city["diesel_price"]
	diesel_subsidies = city["diesel_subsidies"]
	electricity_price = city["electricity_price"]
	electricity_co2_intensity = city["electricity_co2_intensity"]

	area_km2 = .0
	vkm = .0
	vkt_per_vehicle = .0
	vkt_per_trip = .0
	n_vehicles = .0
	baseline_demand = .0
	additionnal_demand = .0
	savings_per_km_with = .0
	savings_per_km_without = .0
	per_vehicle_savings_with = .0
	per_vehicle_savings_without = .0
	diesel_emissions = .0
	ev_emissions = .0
	per_vehicle_emission_reduction = .0
	diesel_saving = .0
	tot_emission_reduction = .0
	pop_no_exposure = .0
	pop_low_exposure = .0
	pop_medium_exposure = .0
	pop_high_exposure = .0

	"""
	Step 1. GTFS Feed Initialization & Preprocessing (Do not comment - required for the other steps)
	"""

	# Populate the feed with the raw data 
	feed = GTFSFeed(gtfs_feed_name)

	# Filter the feed according to city-specific rules defined in the preprocessing script
	feed = pp.gtfs_preprocessing(feed, city_name)

	# Clean and check data consistency
	feed.clean_all() # Data cleaning to get a consistent feed
	feed.check_all() # Re-check data consistency

	# If necessary, snap the shapefiles to OSM road network (good to check once, but generally does)
	if snap_to_osm_roads:
		feed.snap_shapes_to_osm() # Takes a lot of time

	# Display general information about the data
	feed.general_feed_info()
	area_km2 = feed.simulation_area_km2()
	print(area_km2)

	"""
	Step 2. Operation estimates & Power Profile (Do not comment - required for the other steps)
	"""

	trips = list(feed.trips['trip_id']) # Trips to consider
	ev_con = [ev_consumption] * len(trips) # List of EV consumption for all trips

	op = pd.DataFrame()
	traffic_sim = None

	# If True, only compute Traffic simulation if it has not already been done before
	if reuse_traffic_output: 
		filename = f"{city_name}_tmp_operation_{ev_consumption}.pkl"

		# Check if a pickle file with same parameters already exists and the timeseries
		if os.path.exists(f"{OUTPUT_PATH}/{filename}") and os.path.exists(f"{OUTPUT_PATH}/{city_name}_powerprofile.csv"):
			print(f"INFO \t Using existing pickle data for operationnal simulation - Make sure it matches with the inputs")
			# Load the df from the file
			op = pd.read_pickle(f"{OUTPUT_PATH}/{filename}")  
		else:
			# Carry out the simulation
			traffic_sim = TrafficSim(feed, trips, ev_con) # Carry out the simulation for all trips
			
			op = traffic_sim.operation_estimates() # Get operation estimates
			
			# Serialize and save the df to a file
			op.to_pickle(f"{OUTPUT_PATH}/{filename}")

			df = traffic_sim.profile(start_time = "00:00:00", stop_time = "23:59:59", time_step = time_step, transient_state = False)
			df.to_csv(f"{OUTPUT_PATH}/{city_name}_powerprofile.csv", index = False)  
	else:
		# Carry out the simulation
		traffic_sim = TrafficSim(feed, trips, ev_con) # Carry out the simulation for all trips
		op = traffic_sim.operation_estimates() # Get operation estimates

		# Timeseries
		df = traffic_sim.profile(start_time = "00:00:00", stop_time = "23:59:59", time_step = time_step, transient_state = False)
		df.to_csv(f"{OUTPUT_PATH}/{city_name}_powerprofile.csv", index = False)

	vkm = op['vkm'].sum()
	vkt_per_vehicle = sum(op['vkt'] * op['ave_nbr_vehicles'])/op['ave_nbr_vehicles'].sum()
	vkt_per_trip = op['vkt'].mean()
	n_vehicles = op['ave_nbr_vehicles'].sum()

	# Get the main metrics needed for TRAP exposure calculation
	vkm_list = op['vkm'].tolist() # VKM of trips
	linestring_list = [feed.get_shape(row['trip_id']) for index, row in op.iterrows()] # Associated linestrings

	"""
	Step 3. GIS Population Data Preprocessing (Do not comment if you want to use the pop raster as a population estimate)
	"""

	pop_raster = f"{OUTPUT_PATH}/{city_name}_tmp_popraster_cropped.tif" # Path to the cropped raster

	# Crop the input raster
	hlp.crop_raster(f"{INPUT_PATH}/{population_raster_name}", 
		feed.bounding_box(), 
		pop_raster)

	# Load the cropped population raster and calculate the total population
	with rasterio.open(pop_raster) as src:
	    # Read the raster data
	    raster_data = src.read(1)  # assuming it's a single band raster
		    
	    # Calculate the sum of all values
	    raster_sum = np.sum(raster_data)

	    if pop_from_raster:
	    	population = raster_sum

	"""
	Step 4. Energy metrics
	"""
	baseline_demand = population * demand_per_capita
	additionnal_demand = op['energy_kWh'].sum() / charging_efficiency * active_working_days

	daily_demand_per_trip = op['energy_kWh'] / charging_efficiency
	daily_demand_per_trip.to_csv(f"{OUTPUT_PATH}/{city_name}_daily_demand_per_trip_kWh.csv", index = False)

	"""
	Step 5. CO2 Emission Savings
	""" 

	diesel_emissions = diesel_consumption * diesel_co2_intensity
	ev_emissions = ev_consumption / charging_efficiency * electricity_co2_intensity
	per_vehicle_emission_reduction = vkt_per_vehicle * active_working_days * (diesel_emissions-ev_emissions) / 1000
	diesel_saving = vkm*diesel_consumption * active_working_days
	tot_emission_reduction = vkm * (diesel_emissions-ev_emissions)/1000 * active_working_days

	"""
	Step 6. Economic savings
	""" 

	savings_per_km_with = (diesel_consumption * diesel_price) - (ev_consumption / charging_efficiency * electricity_price)
	savings_per_km_without = (diesel_consumption * (diesel_price+diesel_subsidies)) - (ev_consumption / charging_efficiency * electricity_price)
	per_vehicle_savings_with = vkt_per_vehicle*active_working_days*savings_per_km_with
	per_vehicle_savings_without = vkt_per_vehicle*active_working_days*savings_per_km_without

	"""
	Step 7. Air pollution exposure - TRAP Exposure index map
	"""

	# SUBSTEP 1 : Compute the emission index map (i.e., traffic volume map, VKM) using the cropped pop raster as a reference layer
	# IF NO CHANGE IN PARAMETERS, COMMENT IF ALREADY DONE

	if not os.path.exists(f"{OUTPUT_PATH}/{city_name}_tmp_local_em.tif"):
		hlp.local_emission_index(vkm_list, linestring_list, pop_raster, f"{OUTPUT_PATH}/{city_name}_tmp_local_em.tif")

	# SUBSTEP 2 : Compute the distance-weigthed emission exposure map 
	# IF NO CHANGE IN PARAMETERS, COMMENT IF ALREADY DONE

	with rasterio.open(f"{OUTPUT_PATH}/{city_name}_tmp_local_em.tif") as src:
	    # Read the raster data
	    raster_data = src.read(1).astype(float) # assuming it's a single band raster
	    
	    # Define the convolution kernel (e.g., exponential decay)
	    kernel_size = int( (buffer_distance / 100)*2 + 1 ) # distance + current pixel
	    #print(kernel_size)
	    decay_factor = decay_rate * 100  # NO2 = 0.0064 per meter, so 0,64 per pixel | 0.02 in some other refs

	    kernel = hlp.exponential_decay_kernel(kernel_size, decay_factor)

	    # Define the mask for the buffer distance
	    mask = hlp.mask_within_radius(kernel_size, radius=(kernel_size-1)/2)
	    
	    # Apply the mask to the kernel
	    kernel = kernel*mask
		    
	    # Distance-weigthing: perform the convolution operation 
	    convolved_data = convolve2d(raster_data, kernel, mode='same', boundary='wrap')
		    
	    # Create a new raster file with the convolved data
	    profile = src.profile
	    profile.update(dtype=rasterio.float32)  # Update data type to float32
	    with rasterio.open(f"{OUTPUT_PATH}/{city_name}_distance_weighted_exposure.tif", 'w', **profile) as dst:
	        dst.write(convolved_data.astype(rasterio.float32), 1)  # assuming it's a single band raster

	"""
	Step 8. Air pollution exposure - TRAP Exposure values population counts
	"""

	# Open the population raster file
	with rasterio.open(pop_raster) as population_src:
	    # Read the population raster data
	    population_data = population_src.read(1)  # assuming it's a single band raster

	# Open the exposure raster file
	with rasterio.open(f"{OUTPUT_PATH}/{city_name}_distance_weighted_exposure.tif") as property_src:
	    # Read the property raster data
	    exposure_data = property_src.read(1)  # assuming it's a single band raster

	# # Define the property range 
	max_exposure = np.max(exposure_data)  # Define your maximum exposure value

	# Initialize exposure bins
	low = 0
	medium = 0
	high = 0
	not_exposed = 0

	# # Iterate over each pixel in the property raster
	for i in range(exposure_data.shape[0]):
	    for j in range(exposure_data.shape[1]):
	        # Get the property value at the current pixel
	        property_value = exposure_data[i, j]
		        
	        # Check if the property value is within the specified range
	        if low_threshold <= property_value < medium_threshold:
	            low += population_data[i, j]
	        elif medium_threshold <= property_value < high_threshold:
	        	medium += population_data[i, j]
	        elif high_threshold <= property_value <= max_exposure:
	        	high += population_data[i, j]
	        else:
	        	not_exposed += population_data[i, j]

	pop_no_exposure = not_exposed
	pop_low_exposure = low
	pop_medium_exposure = medium
	pop_high_exposure = high

	"""
	Step 8. Air pollution exposure - Population-weighted TRAP Exposure Map
	"""

	# Open the population raster file
	with rasterio.open(pop_raster) as population_src:
	    # Read the population raster data
	    population_data = population_src.read(1)  # assuming it's a single band raster

	# Open the exposure raster file
	with rasterio.open(f"{OUTPUT_PATH}/{city_name}_distance_weighted_exposure.tif") as property_src:
	    # Read the property raster data
	    exposure_data = property_src.read(1)  # assuming it's a single band raster
	    popweighted_exposure = exposure_data * population_data

	    with rasterio.open(f"{OUTPUT_PATH}/{city_name}_pop_weighted_exposure.tif", 'w', **profile) as dst:
	    	dst.write(popweighted_exposure, 1)  # assuming it's a single band raster

	"""
	Step 9. Append the output data to the dataframe
	"""
	data_dict = {
	    "City": city_name,
	    "Area (km2)": area_km2,
	    "VKM (km)": vkm,
	    "VKT per vehicle (km)": vkt_per_vehicle,
	    "VKT per trip (km)": vkt_per_trip,
	    "Number of vehicles": n_vehicles,
	    "Population": population,
	    "Baseline demand (kWh/year)": baseline_demand,
	    "Additionnal demand (kWh/year)": additionnal_demand,
	    "Savings per km w/ subsidies ($/km)": savings_per_km_with,
	    "Savings per km w/o subsidies ($/km)": savings_per_km_without,
	    "Average per vehicle savings w/ subsidies (US$/year)": per_vehicle_savings_with ,
	    "Average per vehicle savings w/o subsidies (US$/year)": per_vehicle_savings_without,
	    "Diesel emissions (kgCO2/km)": diesel_emissions,
	    "EV emissions (kgCO2/km)": ev_emissions,
	    "Average per vehicle emission reduction (tCO2/year)": per_vehicle_emission_reduction,
	    "Diesel savings (L/year)": diesel_saving,
	    "Total emission reductions (tCO2/year)": tot_emission_reduction,
	    "Pop not exposed": pop_no_exposure,
	    "Pop w/ low exposure": pop_low_exposure,
	    "Pop w/ medium exposure": pop_medium_exposure,
	    "Pop w/ high exposure": pop_high_exposure
	}

	out_df = pd.concat([out_df, pd.DataFrame([data_dict])], ignore_index=True)

###########
# WRITE OUT
###########

# Write the DataFrame to a CSV file
out_df.to_csv(f"{OUTPUT_PATH}/_OUTPUTS.csv", index=False)