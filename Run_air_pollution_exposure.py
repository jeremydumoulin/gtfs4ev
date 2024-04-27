# coding: utf-8

""" 
A python script that reproduces TRAP (Traffic Related Air Pollution) exposure results 
for a given city by cross-referencing GTFS and GIS population data (a population layer 
covering the area is required).
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

from gtfs4ev.gtfsfeed import GTFSFeed
from gtfs4ev.tripsim import TripSim
from gtfs4ev.trafficsim import TrafficSim
from gtfs4ev.topology import Topology
from gtfs4ev import helpers as hlp

import Run_preprocess_gtfs as pp

"""
Parameters - PLEASE MODIFY ACCORDING TO YOUR NEEDS
"""

city = "Freetown" # Used to name intermediate outputs and do some city-dependent pre-processing

# Input data
population_raster_name = "Freetown_GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R9_C17.tif" # Make sure it is in the input folder as defined in the .env file
gtfs_feed_name = "GTFS_Freetown" # Make sure the GTFS folder is in the input folder

# Parameters related GTFS Feed and traffic simulation
snap_to_osm_roads = False # Could take a long time. Data is generally already consistent with OSM network
ev_consumption = 0.4 # EV consumption (kWh/km) - Value should not affect the output
reuse_traffic_output = True # If True, serializes the dataframe with operationnal data in order to avopid recomputing TrafficSimulation

# Parameters related to TRAP exposure assessments
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
OUTPUT_PATH = str(os.getenv("OUTPUT_PATH")) 

"""
Main code
"""

##########################################
# GTFS Feed Initialization & Preprocessing
##########################################

# Populate the feed with the raw data 
feed = GTFSFeed(gtfs_feed_name)

# Filter the feed according to city-specific rules defined in the preprocessing script
feed = pp.gtfs_preprocessing(feed, city)

# Clean and check data consistency
feed.clean_all() # Data cleaning to get a consistent feed
feed.check_all() # Re-check data consistency

# If necessary, snap the shapefiles to OSM road network (good to check once, but generally does)
if snap_to_osm_roads:
	feed.snap_shapes_to_osm() # Takes a lot of time

# Display general information about the data
feed.general_feed_info()
print(feed.simulation_area_km2())

#############################################################
# Traffic simulation & Operation estimates of the whole fleet
#############################################################

trips = list(feed.trips['trip_id']) # Trips to consider
ev_con = [ev_consumption] * len(trips) # List of EV consumption for all trips

# If True, only compute Traffic simulation if it has not already been done before
if reuse_traffic_output: 
	filename = f"{city}_tmp_air_pollution_exposure_{ev_consumption}_operation.pkl"

	# Check if a pickle file with same parameters already exists
	if os.path.exists(f"{OUTPUT_PATH}/{filename}"):
		print(f"INFO \t Using existing pickle data for operationnal simulation - Make sure it matches with the inputs")
		# Load the df from the file
		op = pd.read_pickle(f"{OUTPUT_PATH}/{filename}")  
	else:
	    # Carry out the simulation
	    traffic_sim = TrafficSim(feed, trips, ev_con) # Carry out the simulation for all trips
	    op = traffic_sim.operation_estimates() # Get operation estimates

	    # Serialize and save the df to a file	    
	    op.to_pickle(f"{OUTPUT_PATH}/{filename}")  
else:
	# Carry out the simulation
	traffic_sim = TrafficSim(feed, trips, ev_con) # Carry out the simulation for all trips
	op = traffic_sim.operation_estimates() # Get operation estimates

# Get the main metrics needed for TRAP exposure calculation
vkm_list = op['vkm'].tolist() # VKM of trips
linestring_list = [feed.get_shape(row['trip_id']) for index, row in op.iterrows()] # Associated linestrings

###################################
# GIS Population Data Preprocessing
###################################

pop_raster = f"{OUTPUT_PATH}/{city}_tmp_popraster_cropped.tif" # Path to the cropped raster

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

    print("Total population in the area:", raster_sum)

#########################
# TRAP Exposure Index Map
#########################

# STEP 1 : Compute the emission index map (i.e., traffic volume map, VKM) using the cropped pop raster as a reference layer
# IF NO CHANGE IN PARAMETERS, COMMENT IF ALREADY DONE

hlp.local_emission_index(vkm_list, linestring_list, pop_raster, f"{OUTPUT_PATH}/{city}_tmp_local_em.tif")

# STEP 2 : Compute the distance-weigthed emission exposure map 
# IF NO CHANGE IN PARAMETERS, COMMENT IF ALREADY DONE

with rasterio.open(f"{OUTPUT_PATH}/{city}_tmp_local_em.tif") as src:
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

    print(kernel)

    with open("test.csv", 'w') as myfile:
	    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	    wr.writerow(kernel)
	    
    # Distance-weigthing: perform the convolution operation 
    convolved_data = convolve2d(raster_data, kernel, mode='same', boundary='wrap')
	    
    # Create a new raster file with the convolved data
    profile = src.profile
    profile.update(dtype=rasterio.float32)  # Update data type to float32
    with rasterio.open(f"{OUTPUT_PATH}/{city}_distance_weighted_exposure.tif", 'w', **profile) as dst:
        dst.write(convolved_data.astype(rasterio.float32), 1)  # assuming it's a single band raster

########################################
# TRAP Exposure values population counts
########################################
# Open the population raster file
with rasterio.open(pop_raster) as population_src:
    # Read the population raster data
    population_data = population_src.read(1)  # assuming it's a single band raster

# Open the exposure raster file
with rasterio.open(f"{OUTPUT_PATH}/{city}_distance_weighted_exposure.tif") as property_src:
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

print(f"Not exposed: {not_exposed} - Low: {low} - Medium: {medium} - High: {high}")
print(f"Total: {not_exposed + low + medium + high}")


#######################################
# Population-weighted TRAP Exposure Map
#######################################

# Open the population raster file
with rasterio.open(pop_raster) as population_src:
    # Read the population raster data
    population_data = population_src.read(1)  # assuming it's a single band raster

# Open the exposure raster file
with rasterio.open(f"{OUTPUT_PATH}/{city}_distance_weighted_exposure.tif") as property_src:
    # Read the property raster data
    exposure_data = property_src.read(1)  # assuming it's a single band raster
    popweighted_exposure = exposure_data * population_data

    with rasterio.open(f"{OUTPUT_PATH}/{city}_pop_weighted_exposure.tif", 'w', **profile) as dst:
    	dst.write(popweighted_exposure, 1)  # assuming it's a single band raster