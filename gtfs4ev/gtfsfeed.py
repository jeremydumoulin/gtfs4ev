# coding: utf-8

""" 
GTFSFeed. 
Holds the GTFS feed. Is instantiated using a GTFS data folder in the input folder. Provides features for 
checking GTFS data, filtering data (e.g. to keep only services present on certain days), and extracting 
general information about the feed. This class is purely about analyzing and curating data; no modeling 
involved here. 
IMPORTANT: The calendar_dates.txt file is not considered, meaning that some service exception are not 
taken into account.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, box, MultiPoint
from shapely.ops import transform, nearest_points
import pyproj
from pyproj import Geod
from contextlib import redirect_stdout

from gtfs4ev import constants as cst
from gtfs4ev import environment as env
from gtfs4ev import helpers as hlp

class GTFSFeed:

    #######################################
    ############# ATTRIBUTES ##############
    #######################################
    
    datapath = "" # Absolute path to the GTFS datafolder

    # Panda dataframes holding mandatory standard GTFS data
    agency = pd.DataFrame()
    routes = pd.DataFrame()
    stop_times = pd.DataFrame()
    stops = gpd.GeoDataFrame(columns=['stop_id', 'stop_name', 'geometry'], crs="EPSG:4326")
    trips = pd.DataFrame()

    # Dataframes for other required data for gtfs4ev
    calendar = pd.DataFrame()
    frequencies = pd.DataFrame()
    shapes = gpd.GeoDataFrame(columns=['shape_id', 'geometry'], crs="EPSG:4326")

    #######################################
    ############### METHODS ###############
    #######################################
    
    ############# Constructor #############
    ####################################### 

    def __init__(self, gtfs_foldername):
        print(f"INFO \t Initializing a new GTFSFeed object using /{gtfs_foldername} data... ")

        self.set_datapath(gtfs_foldername) 

        # Required data according to the GTFS standard    
        self.set_agency() 
        self.set_trips()       
        self.set_routes()
        self.set_stop_times()
        self.set_stops()

        # Other required data for gtfs4ev
        self.set_calendar()     
        self.set_frequencies()      
        self.set_shapes()
        
        
        print("INFO \t GTFSFeed created. Feed could now be analyzed, cleaned, or filtered.")
        print("\t -")


    ############# Setters #############
    ################################### 

    def set_datapath(self, gtfs_foldername):
        """ Setter for datafolder attribute.
        Checks if the GTFS datafolder exists and contains all the required files
        """
        try:
            abs_path = env.INPUT_PATH / str(gtfs_foldername)
            # Check if required files are present : 1) required files from the GTFS specification 2) required files for gtfs4ev
            files_to_check = ['agency.txt', 'routes.txt', 'stop_times.txt', 'calendar.txt', 'frequencies.txt', 'shapes.txt', 'stops.txt', 'trips.txt']          

            if not os.path.isdir(abs_path):
                raise FileNotFoundError() 

            try:                
                for file_name in files_to_check:
                    file_path = os.path.join(abs_path, file_name)
                    if not os.path.exists(file_path):
                        raise FileNotFoundError()                    
            except Exception:
                print("ERROR \t Some required GTFS files missing. Make sure the following files are in the data folder: 'agency.txt', 'routes.txt', 'stop_times.txt', 'calendar.txt', 'frequencies.txt', 'shapes.txt', 'stops.txt'.")
            finally:
                for file_name in os.listdir(abs_path):
                    file_path = os.path.join(abs_path, file_name)
                    if file_name not in files_to_check:
                        print(f"INFO \t The data folder contains a GTFS file that is not required: '{file_name}'.")

        except FileNotFoundError as e:
            print(f"ERROR \t unable to open the /{gtfs_foldername} folder. Make sure the data folder exists. ")
        except Exception as e:
            print(f"ERROR \t {e}")
        else:            
            self.datapath = abs_path

    def set_agency(self):
        """ Setter for agency attribute.
        Keeps only the four required columns of the GTFS standard
        """        
        file_path = open(self.datapath / "agency.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['agency_id', 'agency_name', 'agency_url', 'agency_timezone']
        column_types = {'agency_id': str, 'agency_name': str, 'agency_url': str, 'agency_timezone': str}
            
        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype=column_types)
        except Exception:
            print("Error: it seems that some of the required columns of the agency.txt file are missing. Please check 'agency_id', 'agency_name', 'agency_url', 'agency_timezone' are present. ")    
        else:
            self.agency = df
            if not hlp.check_dataframe(df):
                print("Warning: empty values or NaN values found in agency.txt. This might cause some issues.")
        
        file_path.close()        

    def set_routes(self):
        """ Setter for routes attribute.
        Keeps only the four required columns of the GTFS standard
        """    
        file_path = open(self.datapath / "routes.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['route_id', 'agency_id', 'route_short_name', 'route_long_name']
        column_types = {'route_id': str, 'agency_id': str, 'route_short_name': str, 'route_long_name': str}            

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype=column_types)       
        except Exception:
            print("ERROR \t It seems that some of the required columns of the routes.txt file are missing. Please check 'route_id', 'agency_id', 'route_short_name', 'route_long_name' are present. ")    
        else:
            self.routes = df
            if not hlp.check_dataframe(df):
                print("ALERT \t Empty values or NaN values found in routes.txt. This might cause some issues.")

        # Extract the 'route_id' column from the first DataFrame
        route_ids = self.routes['route_id']

        # Check if all route_ids from df1 are present in df2
        all_present = route_ids.isin(self.trips['route_id']).all()

        if not all_present:
            print("ALERT \t Some routes are not associated to any trip in the trips.txt file. Consistency check recommended.")
        
        file_path.close()

    def set_stop_times(self):
        """ Setter for stop_times attribute.
        Keeps only the five required columns of the GTFS standard
        """
        file_path = open(self.datapath / "stop_times.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
        column_types = {'trip_id': str, 'arrival_time': str, 'departure_time': str, 'stop_id': str, 'stop_sequence': int}     

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype=column_types)
            df['arrival_time']= pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
            df['departure_time']= pd.to_datetime(df['departure_time'], format='%H:%M:%S')                
        except Exception:
            print("ERROR \t It seems that some of the required columns of the stop_times.txt file are missing. Please check 'trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence' are present. ")    
        else:
            self.stop_times = df
            if not hlp.check_dataframe(df):
                print("ALERT \t Empty values or NaN values found in stop_times.txt. This might cause some issues.")
        
        file_path.close()

    def set_calendar(self):
        """ Setter for calendar attribute.
        """        
        file_path = open(self.datapath / "calendar.txt", "r", encoding = "utf-8")       
        
        column_types = {'service_id': str, 'monday': bool, 'tuesday': bool, 'wednesday': bool, 'thursday': bool, 'friday': bool, 'saturday': bool, 'sunday': bool, 'start_date': str, 'end_date': str}     

        try:          
            df = pd.read_csv(file_path, dtype=column_types)
            df['start_date']= pd.to_datetime(df['start_date'], format='%Y%m%d')
            df['end_date']= pd.to_datetime(df['end_date'], format='%Y%m%d')                
        except Exception:
            print("ERROR \t It seems that some of the required columns of the calendar.txt file are missing. ")    
        else:
            self.calendar = df
            if not hlp.check_dataframe(df):
                print("ALERT \t Empty values or NaN values found in calendar.txt. This might cause some issues.")
        
        file_path.close()

    def set_frequencies(self):
        """ Setter for frequencies attribute.
        Keeps only the four required columns of the GTFS standard
        """
        file_path = open(self.datapath / "frequencies.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['trip_id', 'start_time', 'end_time', 'headway_secs']
        column_types = {'trip_id': str, 'start_time': str, 'end_time': str, 'headway_secs': int}   

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype=column_types)

            # 24:00:00 is not a supported datetime. Replace it if necessary 
            df['start_time'] = df['start_time'].str.replace('24:00:00', '23:59:59')
            df['end_time'] = df['end_time'].str.replace('24:00:00', '23:59:59')

            df['start_time']= pd.to_datetime(df['start_time'], format='%H:%M:%S')
            df['end_time']= pd.to_datetime(df['end_time'], format='%H:%M:%S')              
        except Exception:
            print("ERROR \t It seems that some of the required columns of the frequencies.txt file are missing. ")    
        else:
            self.frequencies = df
            if not hlp.check_dataframe(df):
                print("ALERT \t Empty values or NaN values found in frequencies.txt. This might cause some issues.")

            # Extract the 'stop_id' column from the first DataFrame
            trip_ids = self.trips['trip_id']

            # Check if all stop_ids from df1 are present in df2
            all_present = trip_ids.isin(self.frequencies['trip_id']).all()

            if not all_present:
                print("ALERT \t Some trips are not associated to any frequency in the frequencies.txt file. Consistency check recommended.")
        
        file_path.close()

    def set_trips(self):
        """ Setter for trips attribute.
        Keeps only the four required columns of the GTFS standard
        """
        file_path = open(self.datapath / "trips.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['route_id', 'service_id', 'trip_id', 'shape_id']
        column_types = {'route_id': str, 'service_id': str, 'trip_id': str, 'shape_id': str}            

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype=column_types)               
        except Exception:
            print("ERROR \t It seems that some of the required columns of the trips.txt file are missing. ")    
        else:
            self.trips = df
            if not hlp.check_dataframe(df):
                print("ALERT \t Empty values or NaN values found in trips.txt. This might cause some issues.")
        
        file_path.close()

    def set_shapes(self):
        """ Setter for shapes attribute.
        Transforms the list of latitude and longitude in a linestring 
        """
        file_path = open(self.datapath / "shapes.txt", "r", encoding = "utf-8")       
       
        columns_to_keep = ['shape_id', 'shape_pt_lat', 'shape_pt_lon']                      

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep)
            # Group by shape_id and create LineString for each group
            grouped = df.groupby('shape_id').apply(lambda x: LineString(zip(x['shape_pt_lon'], x['shape_pt_lat'])))
            # Create a GeoDataFrame from the grouped data
            gdf = gpd.GeoDataFrame(geometry=grouped, crs="EPSG:4326").reset_index()                
        except Exception:
            print("ERROR \t Problem in creating the geopanda dataframe. Perhaps some of the required columns of the shapes.txt file are missing.")    
        else:
            self.shapes = gdf
            if not hlp.check_dataframe(df):
                print("ALERT \t Empty values or NaN values found in shapes.txt. This might cause some issues.")

        file_path.close()

    def set_stops(self):
        """ Setter for stops attribute.
        Transforms the list of latitude and longitude in a linestring 
        """
        file_path = open(self.datapath / "stops.txt", "r", encoding = "utf-8")       
    
        columns_to_keep = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']                     

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep)
            # Create Point geometries from latitude and longitude columns
            geometry = [Point(lon, lat) for lon, lat in zip(df['stop_lon'], df['stop_lat'])]
            # Drop latitude and longitude columns
            df = df.drop(['stop_lon', 'stop_lat'], axis=1)
            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        except Exception:
            print("ERROR \t Problem in creating the geopanda dataframe. Perhaps some of the required columns of the stops.txt file are missing.")    
        else:
            self.stops = gdf
            if not hlp.check_dataframe(df):
                print("ALERT \t Empty values or NaN values found in stops.txt. This might cause some issues.")

            # Extract the 'stop_id' column from the first DataFrame
            stop_ids_df1 = self.stops['stop_id']

            # Check if all stop_ids from df1 are present in df2
            all_present = stop_ids_df1.isin(self.stop_times['stop_id']).all()

            if not all_present:
                print("ALERT \t Some stops are not associated to any stop_time in the stop_times.txt file. Consistency check recommended.")
        
        file_path.close()

    ############# GTFS Feed analysis: assessing general transit indicators #############
    #################################################################################### 

    """ Overview of basic global indicators """

    def general_feed_info(self):
        """ Displays some general information about the traffic feed
        """
        print("INFO \t General information about the traffic feed data:")
        print(f"\t \t Trips: {self.trips.shape[0]} - Routes: {self.routes.shape[0]} - Services: {self.calendar.shape[0]}")
        print(f"\t \t Stops: {self.stops.shape[0]} - Stop times: {self.stop_times.shape[0]} - Frequencies: {self.frequencies.shape[0]}")
        print(f"\t \t Agencies: {self.agency.shape[0]} - Shapes: {self.shapes.shape[0]}")
        if self.trips.shape[0]/self.routes.shape[0] == 2.0:
            print("\t \t Note: The number of trips is twice the number of routes, probably meaning that each route is associated with a round trip.")          

        # Group frequencies by id and check if all values in 'row_count' are the same
        group_sizes = self.frequencies.groupby('trip_id').size().reset_index(name='row_count')
        are_all_values_same = group_sizes['row_count'].nunique() == 1

        if are_all_values_same:
            print(f"\t \t Frequency intervals: {group_sizes['row_count'][0]} - Start time: {self.frequencies['start_time'].min()} - End time: {self.frequencies['end_time'].max()}")
        else:
            print("\t \t Note: The number of intervals associated with frequencies is not the same for each trip.")

        print("\t -")

    """ Per trip transit indicators """

    def trip_length_km(self, trip_id, geodesic = True):
        """ Calculates the lenght in km of a trip
        """
        # Get the shape of the corresponding trip and project it into epsg:3857 crs
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')
        linestring = gdf.loc[gdf['trip_id'] == trip_id, 'geometry'].iloc[0]

        if geodesic: 
            geod = Geod(ellps="WGS84")
            distance = geod.geometry_length(linestring) / 1000.0
        else:
            web_mercator_projection = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
            linestring_projection = transform(web_mercator_projection, linestring)
            distance = linestring_projection.length / 1000.0

        return distance

    def trip_duration_sec(self, trip_id):
        """ Calculates the time in sec of a trip
        """
        trip_stop_times = self.stop_times[self.stop_times['trip_id'] == trip_id]
        duration = (trip_stop_times['arrival_time'].iloc[-1] - trip_stop_times['departure_time'].iloc[0]).total_seconds()

        return duration

    def ave_distance_between_stops(self, trip_id, correct_stop_loc = True):
        """ Calculates the average distance between stops in km along a trip
        If needed, the stop location can by clipped to the closest point along the shape of the trip
        """
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')
        linestring = gdf.loc[gdf['trip_id'] == trip_id, 'geometry'].iloc[0]

        web_mercator_projection = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform 

        linestring_web_mercator_projection = transform(web_mercator_projection, linestring)

        # 2. Create a table which contains the stop times for the trip, the stop name and geometry
        filtered_stop_times = self.stop_times[self.stop_times['trip_id'] == trip_id]        
        result_df = pd.merge(filtered_stop_times, self.stops[['stop_id', 'stop_name', 'geometry']], on='stop_id', how='left') # Merge the filtered_df with the stop_points_df on 'stop_id'

        # 3. Add the closest point on the linestring path as the point for the stop
        if correct_stop_loc:
            result_df['closest_point_on_path'] = result_df['geometry'].apply(lambda point: hlp.find_closest_point(linestring, point))
        else:
            result_df['closest_point_on_path'] = result_df['geometry']

        result_df['stop_dist_km'] = .0

        # Define the projection from WGS84 (EPSG:4326) to Web Mercator (EPSG:3857) 
        for i in range(1, len(result_df)):
            current_stop = result_df.iloc[i] 
            previous_stop = result_df.iloc[i - 1]

            start_point = transform(web_mercator_projection, previous_stop['closest_point_on_path'])
            end_point = transform(web_mercator_projection, current_stop['closest_point_on_path'])

            distance_along_line = abs(linestring_web_mercator_projection.project(start_point) - linestring_web_mercator_projection.project(end_point)) 

            result_df.loc[result_df.index == i, 'stop_dist_km'] = distance_along_line

        return result_df['stop_dist_km'].mean() / 1000.0

    def n_stops(self, trip_id):
        """ Number of stops of a trip
        """        
        stop_times = self.stop_times       
        n_stops = stop_times.groupby('trip_id').size().reset_index(name='row_count')

        return n_stops.loc[n_stops['trip_id'] == trip_id, 'row_count'].iloc[0]


    """ Per stop transit indicators """

    def stop_frequencies(self):
        """ Returns the number of times a stop is used by all trips
        """
        stop_times = self.stop_times

        stop_counts = stop_times.groupby('stop_id').size().reset_index(name='count')
        stop_counts = pd.merge(stop_counts, self.stops[['stop_id', 'geometry']], on='stop_id', how='left')
        
        return stop_counts

    """ Global transit indicators """

    def bounding_box(self):
        """ Returns the bounding box for the simulation    
        """
        gdf = self.shapes

        self.min_latitude = gdf['geometry'].apply(lambda line: line.bounds[1]).min()
        self.max_latitude = gdf['geometry'].apply(lambda line: line.bounds[3]).max()
        self.min_longitude = gdf['geometry'].apply(lambda line: line.bounds[0]).min()
        self.max_longitude = gdf['geometry'].apply(lambda line: line.bounds[2]).max()

        return box(self.min_longitude, self.min_latitude, self.max_longitude, self.max_latitude)

    def simulation_area_km2(self):
        """ Calculates the simulation area in km2
        """
        # Create a GeoDataFrame with the bounding box
        gdf_bbox = gpd.GeoDataFrame(geometry=[self.bounding_box()], crs='epsg:4326')

        # Reproject the GeoDataFrame to EPSG:3857
        gdf_bbox_web_mercator = gdf_bbox.to_crs('epsg:3857')

        # Calculate the area in square kilometers
        area_km2 = gdf_bbox_web_mercator['geometry'].area / 1e6  # Convert square meters to square kilometers

        return area_km2[0]  

    def trip_length_km_all(self):
        """ Length of all trips
        """        
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')

        gdf['trip_length_km'] = gdf['trip_id'].apply(self.trip_length_km)

        return gdf       

    def ave_distance_between_stops_all(self, correct_stop_loc = True):
        """ Average distance between stops in km of all trips
        """        
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')

        gdf['stop_dist_km'] = gdf['trip_id'].apply(lambda trip_id: self.ave_distance_between_stops(trip_id, correct_stop_loc=correct_stop_loc))
        gdf['n_stops'] = gdf['trip_id'].apply(self.n_stops)

        return gdf

    def trip_statistics(self):
        """ Calculates some general statitistics regarding the trips
        """   
        number_of_trips = len(self.trips)
        number_of_routes = len(self.routes)
        gdf = self.trip_length_km_all()

        # Calculate the minimum, maximum, average, and standard deviation of the number of rows per trip_id
        statistics = {
            'total_trips': number_of_trips,
            'total_trip_len_km': gdf['trip_length_km'].sum(),
            'ave_trip_len_km': gdf['trip_length_km'].mean(),
            'min_trip_len_km': gdf['trip_length_km'].min(),
            'max_trip_len_km': gdf['trip_length_km'].max(), 
            'trip_to_route_ratio': number_of_trips / number_of_routes
        }
        
        return statistics

    def stop_statistics(self):
        """ Calculates some general statistics regarding the number of stops
        """
        # Add the route information to the stop_times by merging the two DataFrames on 'trip_id'
        stop_times = pd.merge(self.stop_times, self.trips[['trip_id', 'route_id']], on='trip_id', how='left')

        number_of_stops = len(self.stops)
        number_of_trips = len(self.trips)
        number_of_routes = len(self.routes)

        # Calculate the minimum, maximum, average, and standard deviation of the number of rows per trip_id
        statistics = {
            'total_stops': number_of_stops,
            'min_stops_per_trip': stop_times.groupby('trip_id').size().min(),
            'max_stops_per_trip': stop_times.groupby('trip_id').size().max(),
            'std_dev_stops_per_trip': stop_times.groupby('trip_id').size().std(),
            'ave_stops_per_trip': stop_times.groupby('trip_id').size().mean(),
            'ave_stops_per_route': stop_times.groupby('route_id').size().mean(),
            'stops_to_trips_ratio': number_of_stops / number_of_trips,
            'stops_to_routes_ratio': number_of_stops / number_of_routes            
        }
        
        return statistics 

    ############# Data checking and cleaning #############
    ######################################################

    """ Data checking """

    def check_agency(self):
        """ Checking that each agency is associated with at least one route
        """
        problem = False
        agency_ids = self.agency['agency_id']
        
        all_present = agency_ids.isin(self.routes['agency_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some agencies are not associated with any route.")

        return problem 

    def check_shapes(self):
        """ Checking that each shape is associated with at least one trip
        """
        problem = False

        shape_ids = self.shapes['shape_id']
        
        all_present = shape_ids.isin(self.trips['shape_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some shapes are not associated with any trip.")

        return problem

    def check_stops(self):
        """ Checking that each stop is associated with at least one stop time
        """
        problem = False

        stop_ids = self.stops['stop_id']
        
        all_present = stop_ids.isin(self.stop_times['stop_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some stops are not associated with any stop time.")

        return problem

    def check_frequencies(self):
        """ Checking that each frequency is associated with at least one stop trip
        """
        problem = False

        trip_ids = self.frequencies['trip_id']
        
        all_present = trip_ids.isin(self.trips['trip_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some frequencies are not associated with any trip.")

        return problem

    def check_calendar(self):
        """ Checking that each service is associated with at least one stop trip
        """
        problem = False

        service_ids = self.calendar['service_id']
        
        all_present = service_ids.isin(self.trips['service_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some services are not associated with any trip.")

        return problem

    def check_routes(self):
        """ Checking that each route is associated with an agency and trips
        """
        problem = False

        route_ids = self.routes['route_id']
        agency_ids = self.routes['agency_id']
        
        all_present = route_ids.isin(self.trips['route_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some routes are not associated with any trip.")

        all_present = agency_ids.isin(self.agency['agency_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some routes are not associated with any agency.")

        return problem

    def check_stop_times(self):
        """ Checking that each stop time is associated with trips and stops
        """
        problem = False

        stop_ids = self.stop_times['stop_id']
        trip_ids = self.stop_times['trip_id']
        
        all_present = stop_ids.isin(self.stops['stop_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some stop times are not associated with any stop.")

        all_present = trip_ids.isin(self.trips['trip_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some stop times are not associated with any trip.")

        return problem

    def check_trips(self):
        """ Checking that each trips is associated with the needed information: routes, service, frequencies, stop_times, shapes
        """
        problem = False

        route_ids = self.trips['route_id']
        service_ids = self.trips['service_id']
        trip_ids = self.trips['trip_id']
        shape_ids = self.trips['shape_id']
        
        all_present = route_ids.isin(self.routes['route_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some trips are not associated with any route.")

        all_present = service_ids.isin(self.calendar['service_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some trips are not associated with any agency.")

        all_present = trip_ids.isin(self.frequencies['trip_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some trips are not associated with any frequencies. Could cause important issues.")

        all_present = trip_ids.isin(self.stop_times['trip_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some trips are not associated with any stop times.")  

        all_present = shape_ids.isin(self.shapes['shape_id']).all()
        if not all_present:
            problem = True
            print("ALERT \t Data consistency: some trips are not associated with any shape.")

        return problem

    def check_all(self): 
        """ Checking the consistency of all the data
        """
        print("INFO \t Checking data consistency...")

        consistent = False

        agency_status = self.check_agency()
        shapes_status = self.check_shapes()
        stops_status = self.check_stops()
        frequencies_status = self.check_frequencies()
        calendar_status = self.check_calendar()
        routes_status = self.check_routes()
        stop_times_status = self.check_stop_times()
        trips_status = self.check_trips()

        if (not agency_status and not shapes_status and not stops_status and not frequencies_status and not calendar_status and not routes_status and not stop_times_status and not trips_status):
            print("INFO \t No problem found.")
            consistent = True

        return consistent

    """ Quick data fixing: the practical minimum for gtfs4ev. WARNING: Data consistency could be altered. This could be useful in order to simulate all the possbile trips even if some of them do not belong to any route or so. """

    def drop_useless_trips(self):
        """ Delete the trips which are not useable because they are not assiocated to any frequency, stop_times, or shapes
        Warning: This ensures that all the trips could be simulated by gtfs4ev but could alter data consistency
        """
        trips = self.trips
        frequencies = self.frequencies
        stop_times = self.stop_times
        shapes = self.shapes

        # Drop trips not associated to any frequency
        merged_df = trips.merge(frequencies, on='trip_id')        
        cleaned_trips = trips[trips['trip_id'].isin(merged_df['trip_id'])] # Only keep the useful rows

        # Drop trips not associated to any stop_times
        merged_df = cleaned_trips.merge(stop_times, on='trip_id')        
        cleaned_trips_2 = cleaned_trips[cleaned_trips['trip_id'].isin(merged_df['trip_id'])] # Only keep the useful rows

        # Drop trips not associated to any shape
        merged_df = cleaned_trips_2.merge(shapes, on='shape_id')        
        cleaned_trips_3 = cleaned_trips_2[cleaned_trips_2['shape_id'].isin(merged_df['shape_id'])] # Only keep the useful rows

        self.trips = cleaned_trips_3

    def drop_useless_stop_times(self):
        """ Delete the stop_times which are not useable because they are not assiocated to any stop
        Warning: This ensures that all the stop_times could be simulated by gtfs4ev but could alter data consistency
        """        
        stop_times = self.stop_times
        stops = self.stops

        # Drop stop_times not associated to any frequency
        merged_df = stop_times.merge(stops, on='stop_id')        
        cleaned_stop_times = stop_times[stop_times['stop_id'].isin(merged_df['stop_id'])] # Only keep the useful rows

        self.stop_times = cleaned_stop_times

    """ Data cleaning: Extensive data cleansing to ensure consistency of GTFS data. """

    def clean_agency(self):
        """ Deleting each agency not associated with any route
        """
        self.agency = self.agency[self.agency['agency_id'].isin(self.routes['agency_id'])]

    def clean_shapes(self):
        """ Deleting each shape not associated with any trip
        """
        self.shapes = self.shapes[self.shapes['shape_id'].isin(self.trips['shape_id'])]

    def clean_stops(self):
        """ Deleting each stop not associated with any stop_time
        """
        self.stops = self.stops[self.stops['stop_id'].isin(self.stop_times['stop_id'])]

    def clean_frequencies(self):
        """ Deleting each frequency not associated with any trip
        """
        self.frequencies = self.frequencies[self.frequencies['trip_id'].isin(self.trips['trip_id'])]

    def clean_calendar(self):
        """ Deleting each service not associated with any trip
        """
        self.calendar = self.calendar[self.calendar['service_id'].isin(self.trips['service_id'])]

    def clean_routes(self):
        """ Deleting each route not associated with any agency or trip
        """
        self.routes = self.routes[self.routes['agency_id'].isin(self.agency['agency_id'])]
        self.routes = self.routes[self.routes['route_id'].isin(self.trips['route_id'])]

    def clean_stop_times(self):
        """ Deleting each stop_time not associated with any trip or stop
        """
        self.drop_useless_stop_times()
        self.stop_times = self.stop_times[self.stop_times['trip_id'].isin(self.trips['trip_id'])]

    def clean_trips(self):
        """ Deleting each trip not associated with any routes, service, frequencies, stop_times, or shapes
        """
        self.drop_useless_trips()
        self.trips = self.trips[self.trips['route_id'].isin(self.routes['route_id'])]
        self.trips = self.trips[self.trips['service_id'].isin(self.calendar['service_id'])]

    def clean_all(self):
        """ Executing all the cleaning functions. No consistency problems should arise after this function has been run.
        """
        # Clean everything and repeat until consistency is reached 
        with redirect_stdout(None):       
            while not self.check_all():
                self.clean_agency()
                self.clean_shapes()
                self.clean_stops()
                self.clean_frequencies()
                self.clean_calendar()
                self.clean_routes() 
                self.clean_stop_times()
                self.clean_trips()

        print("INFO \t Clean all: data has been cleaned. A consistency check is recommended.")        

    ############# Data filtering #############
    ##########################################

    def filter_services(self, service_id, clean_all = True):
        """ Drop the trips belonging to a specific service, for example to consider only weekdays
        """

        print(f"INFO \t Filtering out all the data from the following service: {service_id}")

        # Drop the trips associated to a specific service       
        self.trips.drop(self.trips[self.trips['service_id'] == service_id].index, inplace=True)

        # Clean all the dataset to be consistent with the trips
        if clean_all:
            with redirect_stdout(None): 
                self.clean_all()

    ############# Various helper methods #############
    ##################################################   

    def get_shape(self, trip_id):
        """ Get the shape of a trip_id as a LineString Object
        """
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')
        linestring = gdf.loc[gdf['trip_id'] == trip_id, 'geometry'].iloc[0]

        return linestring

    def get_stop_locations(self, trip_id):
        """ Get a list of the coordinates of all the stops belonging to a trip 
        """
        gdf = pd.merge(self.stop_times, self.stops[['stop_id', 'geometry']], on='stop_id', how='left')       
        coordinates = gdf.loc[gdf['trip_id'] == trip_id, 'geometry']

        return coordinates.tolist()