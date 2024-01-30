# coding: utf-8

""" 
Some useful generic functions for gtfs4ev classes.
"""

import pandas as pd
from shapely.geometry import LineString, Point
import numpy as np


def check_dataframe(df):
	""" Returns false if a dataframe contains neither NaN nor empty values.	  
    """

	# Check for NaN values in the entire DataFrame
	if df.isna().any().any():
		return False
	# Check for empty cells in the entire DataFrame
	elif df.apply(lambda x: x == '').any().any():
		return False
	else:
		return True

def find_closest_point(line, point):
	""" Among all the points of a line (LineString object), returns the one closest to the point (Point object) coordinates  
    """
    min_distance = float('inf')  # Initialize with a large value
    closest_point = None
    
    for coordinate in line.coords:
        current_point = Point(coordinate)
        distance = point.distance(current_point)
        
        if distance < min_distance:
            min_distance = distance
            closest_point = current_point
    
    return closest_point