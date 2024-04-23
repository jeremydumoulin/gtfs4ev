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

"""
Environment variables
"""
load_dotenv() # take environment variables from .env

INPUT_PATH = str(os.getenv("INPUT_PATH"))
OUTPUT_PATH = str(os.getenv("OUTPUT_PATH")) 

"""
Parameters
"""

city = "Nairobi" # Used to name intermediate outputs and do some city-dependent pre-processing

# Input data
population_raster_name = "Nairobi_GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0_R10_C22.tif" # Make sure it is in the input folder as defined in the .env file
gtfs_feed_name = "GTFS_Nairobi" # Make sure the GTFS folder is in the input folder

# Parameters related to TRAP exposure assessments
decay_rate = 0.0
buffer_distance = 300 # Distance in meters

"""
Main code
"""

##########################################
# GTFS Feed Initialization & Preprocessing
##########################################

# Populate the feed with the raw data 
feed = GTFSFeed(gtfs_feed_name)

# Filter the feed according to city-specific rules defined in the preprocessing script