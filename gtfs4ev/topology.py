# coding: utf-8

""" 
Topology
A class to calculate some topological metrics of a GTFS Feed.
Pending task: for the moment, contains only very basic features 
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, box, MultiPoint
from shapely.ops import transform, nearest_points
import pyproj

from gtfs4ev.gtfsfeed import GTFSFeed
from gtfs4ev import helpers as hlp

class Topology:  

    """
    ATTRIBUTES
    """
    
    feed = "" # GTFSFeed object

    """
    METHODS
    """

    """ Constructor """

    def __init__(self, feed):
        self.set_feed(feed)

    def set_feed(self, feed):
        """ Setter for feed attribute.
        Checks that the object is of the right type
        """
        try:       
            # Check if the value is an instance of the expected type
            if not isinstance(feed, GTFSFeed):
                raise TypeError(f"Expected an instance of YourCustomType, but got {type(value)}")
        except TypeError:
            print(f"\t Error: Impossible to initiate the traffic feed")
        else:            
            self.feed = feed


    """ Topology information """

    def nearest_point_distance_km(self):
        """ Distance from a stop to the closest neirby other stop
        """ 

        points = self.feed.stops['geometry'].tolist()   

        df = pd.DataFrame(self.feed.stops)

        # Reindex the DataFrame starting from 0
        df = df.reset_index(drop=True)

        nearest_point_values = []
        distance_values = []

        for i, row in df.iterrows():
            print(i)
            point_to_drop = points[i]
            filtered_points = [item for item in points if item != point_to_drop ]

            multipoints = MultiPoint(filtered_points)

            nearest_point = nearest_points(points[i], multipoints)

            nearest_point_values.append(nearest_point)  

            point1 = nearest_point[0]
            point2 = nearest_point[1]

            # Project points into EPSG:3857 (Web Mercator)
            # Define the source and target coordinate reference systems
            web_mercator_projection = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform 

            # Project points into EPSG:3857 (Web Mercator)
            point1_proj = transform(web_mercator_projection, point1)
            point2_proj = transform(web_mercator_projection, point2)

            # Calculate the distance between two points using Shapely's distance method
            distance_km = point1_proj.distance(point2_proj) / 1000  # Convert meters to kilometers

            print(distance_km)
            distance_values.append(distance_km)               

        df['nearest_points'] = nearest_point_values
        df['distance_km'] = distance_values

        return df 

    def trip_crossovers(self):
        """ Number of crossovers in the network
            Todo: eliminate the crossovers on the same route
        """
        shapes = self.feed.shapes['geometry'].tolist()

        # Initialize a counter for crossovers
        crossovers_count = 0  

        # Assess the number of crossovers between all pairs of LineStrings
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                if shapes[i].intersects(shapes[j]):
                    crossovers_count += 1 

        return crossovers_count