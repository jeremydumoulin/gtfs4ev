# coding: utf-8

import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, box

from gtfs4ev import constants as cst
from gtfs4ev import environment as env
from gtfs4ev import helpers as hlp

class TrafficFeed:  

    """
    ATTRIBUTES
    """