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

class TrafficSim:  

    """
    ATTRIBUTES
    """
    
datapath = "" # Absolute path to the GTFS datafolder

    # Panda dataframes holding standard data
    agency = pd.DataFrame()    
    routes = pd.DataFrame()
    stop_times = pd.DataFrame()
    calendar = pd.DataFrame()
    frequencies = pd.DataFrame()
    trips = pd.DataFrame()

    # Geopanda dataframes holding georeferenced data
    shapes = gpd.GeoDataFrame(columns=['shape_id', 'geometry'], crs="EPSG:4326")
    stops = gpd.GeoDataFrame(columns=['stop_id', 'stop_name', 'geometry'], crs="EPSG:4326")

    # Spatial parameters
    min_longitude = .0
    min_latitude = .0
    max_longitude = .0
    max_latitude = .0  

    bounding_box = .0

    """
    METHODS
    """

    """ Constructor """

    def __init__(self, gtfs_foldername):
        print(f"Initializing the TrafficFeed object using the /{gtfs_foldername} data folder: ")

        self.set_datapath(gtfs_foldername)

        print("\t -")

        self.set_agency()
        self.set_routes()
        self.set_stop_times()
        self.set_calendar()
        self.set_frequencies()
        self.set_trips()

        self.set_shapes()
        self.set_stops()

        self.set_min_max_latlon()
        self.set_bounding_box()

        print("\t -")
        print("\t TrafficFeed created. You can display general information using the general_feed_info() method.")

    """ Setters """

    def set_datapath(self, gtfs_foldername):
        """ Setter for datafolder attribute.
        Checks if the GTFS datafolder exists and contains all the required files
        """
        try:
            abs_path = env.INPUT_PATH / str(gtfs_foldername)
            files_to_check = ['agency.txt', 'routes.txt', 'stop_times.txt', 'calendar.txt', 'frequencies.txt', 'shapes.txt', 'stops.txt', 'trips.txt']          

            if not os.path.isdir(abs_path):
                raise FileNotFoundError() 

            try:                
                for file_name in files_to_check:
                    file_path = os.path.join(abs_path, file_name)
                    if not os.path.exists(file_path):
                        raise FileNotFoundError()                    
            except Exception:
                print("\t Error: one of the required GTFS files seems to be missing. Make sure the following files are in the data folder: 'agency.txt', 'routes.txt', 'stop_times.txt', 'calendar.txt', 'frequencies.txt', 'shapes.txt', 'stops.txt'.")
            finally:
                print("\t Folder found. All required .txt files are present.")
                for file_name in os.listdir(abs_path):
                    file_path = os.path.join(abs_path, file_name)
                    if file_name not in files_to_check:
                        print(f"\t Warning: The data folder contains an additional but unused file named '{file_name}'.")

        except FileNotFoundError as e:
            print(f"\t Error: unable to open the /{gtfs_foldername} folder. Make sure the data folder exists. ")
        except Exception as e:
            print(f"\t Error: {e}")
        else:            
            self.datapath = abs_path

    def set_agency(self):
        """ Setter for agency attribute.
        Keeps only the four required columns of the GTFS standard
        """
        print("\t Agency: ", end="", flush=True)
        
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
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_routes(self):
        """ Setter for routes attribute.
        Keeps only the four required columns of the GTFS standard
        """
        print("\t Routes: ", end="", flush=True)
        
        file_path = open(self.datapath / "routes.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['route_id', 'agency_id', 'route_short_name', 'route_long_name']
        column_types = {'route_id': str, 'agency_id': str, 'route_short_name': str, 'route_long_name': str}            

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype=column_types)       
        except Exception:
            print("Error: it seems that some of the required columns of the routes.txt file are missing. Please check 'route_id', 'agency_id', 'route_short_name', 'route_long_name' are present. ")    
        else:
            self.routes = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_stop_times(self):
        """ Setter for stop_times attribute.
        Keeps only the five required columns of the GTFS standard
        """
        print("\t Stop times: ", end="", flush=True)

        file_path = open(self.datapath / "stop_times.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']
        column_types = {'trip_id': str, 'arrival_time': str, 'departure_time': str, 'stop_id': str, 'stop_sequence': int}     

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype=column_types)
            df['arrival_time']= pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
            df['departure_time']= pd.to_datetime(df['departure_time'], format='%H:%M:%S')                
        except Exception:
            print("Error: it seems that some of the required columns of the stop_times.txt file are missing. Please check 'trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence' are present. ")    
        else:
            self.stop_times = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_calendar(self):
        """ Setter for calendar attribute.
        """
        print("\t Calendar: ", end="", flush=True)
        
        file_path = open(self.datapath / "calendar.txt", "r", encoding = "utf-8")       
        
        column_types = {'service_id': str, 'monday': bool, 'tuesday': bool, 'wednesday': bool, 'thursday': bool, 'friday': bool, 'saturday': bool, 'sunday': bool, 'start_date': str, 'end_date': str}     

        try:          
            df = pd.read_csv(file_path, dtype=column_types)
            df['start_date']= pd.to_datetime(df['start_date'], format='%Y%m%d')
            df['end_date']= pd.to_datetime(df['end_date'], format='%Y%m%d')                
        except Exception:
            print("Error: it seems that some of the required columns of the calendar.txt file are missing. ")    
        else:
            self.calendar = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_frequencies(self):
        """ Setter for frequencies attribute.
        Keeps only the four required columns of the GTFS standard
        """
        print("\t Frequencies: ", end="", flush=True)

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
            print("Error: it seems that some of the required columns of the frequencies.txt file are missing. ")    
        else:
            self.frequencies = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_trips(self):
        """ Setter for trips attribute.
        Keeps only the four required columns of the GTFS standard
        """
        print("\t Trips: ", end="", flush=True)

        file_path = open(self.datapath / "trips.txt", "r", encoding = "utf-8")       
        
        columns_to_keep = ['route_id', 'service_id', 'trip_id', 'shape_id']
        column_types = {'route_id': str, 'service_id': str, 'trip_id': str, 'shape_id': str}            

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep, dtype=column_types)               
        except Exception:
            print("Error: it seems that some of the required columns of the trips.txt file are missing. ")    
        else:
            self.trips = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_shapes(self):
        """ Setter for shapes attribute.
        Transforms the list of latitude and longitude in a linestring 
        """
        print("\t Shapes: ", end="", flush=True)

        file_path = open(self.datapath / "shapes.txt", "r", encoding = "utf-8")       
       
        columns_to_keep = ['shape_id', 'shape_pt_lat', 'shape_pt_lon']                      

        try:          
            df = pd.read_csv(file_path, usecols=columns_to_keep)
            # Group by shape_id and create LineString for each group
            grouped = df.groupby('shape_id').apply(lambda x: LineString(zip(x['shape_pt_lon'], x['shape_pt_lat'])))
            # Create a GeoDataFrame from the grouped data
            gdf = gpd.GeoDataFrame(geometry=grouped, crs="EPSG:4326").reset_index()                
        except Exception:
            print("Error: problem in creating the geopanda dataframe. Perhaps some of the required columns of the shapes.txt file are missing.")    
        else:
            self.shapes = gdf
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")
        
        file_path.close()

    def set_stops(self):
        """ Setter for stops attribute.
        Transforms the list of latitude and longitude in a linestring 
        """
        print("\t Stops: ", end="", flush=True)

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
            print("Error: problem in creating the geopanda dataframe. Perhaps some of the required columns of the stops.txt file are missing.")    
        else:
            self.stops = gdf
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found. This might cause some issues.")
            else:
                print("Dataframe created")

            # Extract the 'stop_id' column from the first DataFrame
            stop_ids_df1 = self.stop_times['stop_id']

            # Check if all stop_ids from df1 are present in df2
            all_present = stop_ids_df1.isin(self.stops['stop_id']).all()

            if not all_present:
                print("\t Warning: All stop_ids present in the stops.txt file are not present in the stop_times.txt file.")
        
        file_path.close()


    def set_min_max_latlon(self):
        """ Setter for min_lat, max_lat, min_lon, min_lat attributes.        
        """
        gdf = self.shapes

        # Assuming gdf is your GeoPandas DataFrame
        self.min_latitude = gdf['geometry'].apply(lambda line: line.bounds[1]).min()
        self.max_latitude = gdf['geometry'].apply(lambda line: line.bounds[3]).max()
        self.min_longitude = gdf['geometry'].apply(lambda line: line.bounds[0]).min()
        self.max_longitude = gdf['geometry'].apply(lambda line: line.bounds[2]).max()

    def set_bounding_box(self):
        """ Setter for bounding box shape      
        """
        self.bounding_box = box(self.min_longitude, self.min_latitude, self.max_longitude, self.max_latitude)

    """ Data analysis """    

    def general_feed_info(self):
        """ Displays some general information about the traffic feed
        """
        print("\nGeneral information about the traffic feed data:")
        print(f"\t The transport system comprises {self.trips.shape[0]} trips, belonging to {self.routes.shape[0]} routes and {self.calendar.shape[0]} services")
        if int(self.trips.shape[0]/self.routes.shape[0]) == 2:
            print("\t Note: The number of trips is twice the number of routes, probably meaning that each route is associated with a round trip.")
             
        print(f"\t Agencies: {self.agency.shape[0]} - Stops: {self.stops.shape[0]} associated with {self.stop_times.shape[0]} stop times - Frequencies: {self.frequencies.shape[0]}")
    
        # Group frequencies by id and check if all values in 'row_count' are the same
        group_sizes = self.frequencies.groupby('trip_id').size().reset_index(name='row_count')
        are_all_values_same = group_sizes['row_count'].nunique() == 1

        if are_all_values_same:
            print(f"\t Frequency intervals: {group_sizes['row_count'][0]} - Start time: {self.frequencies['start_time'].min()} - End time {self.frequencies['end_time'].max()}")
        else:
            print("\t Warning: the number of intervals associated with frequencies is not the same for each trip")

    def simulation_area(self):
        """ Calculates the simulation area in km2
        """

        # Create a GeoDataFrame with the bounding box
        gdf_bbox = gpd.GeoDataFrame(geometry=[self.bounding_box], crs='epsg:4326')

        # Reproject the GeoDataFrame to EPSG:3857
        gdf_bbox_web_mercator = gdf_bbox.to_crs('epsg:3857')

        # Calculate the area in square kilometers
        area_km2 = gdf_bbox_web_mercator['geometry'].area / 1e6  # Convert square meters to square kilometers

        return area_km2[0]

    def stop_statistics(self):
        """ Calculates some statitistics regarding the number of stops
        """
        # Add the route information to the stop_times by merging the two DataFrames on 'trip_id'
        stop_times = pd.merge(self.stop_times, self.trips[['trip_id', 'route_id']], on='trip_id', how='left')

        number_of_stops = len(self.stops)
        number_of_trips = len(self.trips)
        number_of_routes = len(self.routes)

        # Calculate the minimum, maximum, averagen, and standard deviation of the number of rows per trip_id
        statistics = {
            'min_stops_per_trip': stop_times.groupby('trip_id').size().min(),
            'max_stops_per_trip': stop_times.groupby('trip_id').size().max(),
            'std_dev_stops_per_trip': stop_times.groupby('trip_id').size().std(),
            'ave_stops_per_trip': stop_times.groupby('trip_id').size().mean(),
            'ave_stops_per_route': stop_times.groupby('route_id').size().mean(),
            'stops_to_trips_ratio': number_of_stops / number_of_trips,
            'stops_to_routes_ratio': number_of_stops / number_of_routes
        }
        
        return statistics   