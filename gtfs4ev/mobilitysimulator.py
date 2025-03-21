# coding: utf-8

import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import LineString, Point, box
from shapely.ops import transform, snap
import pyproj
from pyproj import Geod
import osmnx
from contextlib import redirect_stdout
import folium
from folium.plugins import MarkerCluster
from folium import PolyLine

from gtfs4ev import helpers as hlp

class MobilitySimulator:
    """A class to represent and manage a GTFS feed data.

    Holds and curates the GTFS feed. This class reads a GTFS data folder and provides methods
    to check data consistency, filter and clean data, and extract general transit indicators.

    Note: The calendar_dates.txt file is not considered, so some service exceptions are not handled.
    """

    def __init__(self, gtfs_datafolder):
        print("=========================================")
        print(f"INFO \t Creation of a GTFSManager object.")
        print("=========================================")