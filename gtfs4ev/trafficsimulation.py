# coding: utf-8

import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import transform
import pyproj

from gtfs4ev import constants as cst
from gtfs4ev import environment as env
from gtfs4ev import helpers as hlp

from gtfs4ev.trafficfeed import TrafficFeed

class TrafficSimulation:  

    """
    ATTRIBUTES
    """

    start_time = ""
    stop_time = ""
    time_step = 10

    feed = "" # TrafficFeed object
    trip_list = [] # List of trip_ids to simulate

    """
    METHODS
    """

    """ Constructor """

    def __init__(self, feed, trip_id, ev_consumption):
        print(f"Initializing a new vehicle: ")

        

    """ Setters """

    
