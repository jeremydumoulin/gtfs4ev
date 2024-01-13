# coding: utf-8

import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon, box, MultiPoint
from shapely.ops import transform, nearest_points
import pyproj

from gtfs4ev import constants as cst
from gtfs4ev import environment as env
from gtfs4ev import helpers as hlp

class GTFSFeed:  

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


    """
    METHODS
    """


    """ Constructor """

    def __init__(self, gtfs_foldername):
        print(f"\nInitializing the TrafficFeed object using the /{gtfs_foldername} data folder: ")

        self.set_datapath(gtfs_foldername)        

        self.set_agency()        
        self.set_stop_times()
        self.set_calendar()
        self.set_trips()
        self.set_routes()
        self.set_frequencies()        

        self.set_shapes()
        self.set_stops()

        print("\t -")
        print("\t TrafficFeed successfully created. You can display general information using the general_feed_info() method.")


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
                print("\t Folder found, including all required .txt files.")
                for file_name in os.listdir(abs_path):
                    file_path = os.path.join(abs_path, file_name)
                    if file_name not in files_to_check:
                        print(f"\t Note: The data folder contains an additional but unused file named '{file_name}'.")

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
            print("Error: it seems that some of the required columns of the routes.txt file are missing. Please check 'route_id', 'agency_id', 'route_short_name', 'route_long_name' are present. ")    
        else:
            self.routes = df
            if not hlp.check_dataframe(df):
                print("Warning: empty values or NaN values found in routes.txt. This might cause some issues.")

        # Extract the 'route_id' column from the first DataFrame
        route_ids = self.routes['route_id']

        # Check if all route_ids from df1 are present in df2
        all_present = route_ids.isin(self.trips['route_id']).all()

        if not all_present:
            print("\t Warning: Some routes are not associated to any trip in the trips.txt file. The clean_routes() function could be used to solve this issue.")
        
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
            print("Error: it seems that some of the required columns of the stop_times.txt file are missing. Please check 'trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence' are present. ")    
        else:
            self.stop_times = df
            if not hlp.check_dataframe(df):
                print("Warning: empty values or NaN values found in stop_times.txt. This might cause some issues.")
        
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
            print("Error: it seems that some of the required columns of the calendar.txt file are missing. ")    
        else:
            self.calendar = df
            if not hlp.check_dataframe(df):
                print("Warning: empty values or NaN values found in calendar.txt. This might cause some issues.")
        
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
            print("Error: it seems that some of the required columns of the frequencies.txt file are missing. ")    
        else:
            self.frequencies = df
            if not hlp.check_dataframe(df):
                print("Warning: empty values or NaN values found in frequencies.txt. This might cause some issues.")

            # Extract the 'stop_id' column from the first DataFrame
            trip_ids = self.trips['trip_id']

            # Check if all stop_ids from df1 are present in df2
            all_present = trip_ids.isin(self.frequencies['trip_id']).all()

            if not all_present:
                print("\t Warning: Some trips are not associated to any frequency in the frequencies.txt file. The clean_trips() function could be used to solve this issue.")
        
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
            print("Error: it seems that some of the required columns of the trips.txt file are missing. ")    
        else:
            self.trips = df
            if not hlp.check_dataframe(df):
                print("Dataframe created - Warning: empty values or NaN values found in trips.txt. This might cause some issues.")
        
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
            print("Error: problem in creating the geopanda dataframe. Perhaps some of the required columns of the shapes.txt file are missing.")    
        else:
            self.shapes = gdf
            if not hlp.check_dataframe(df):
                print(" Warning: empty values or NaN values found in shapes.txt. This might cause some issues.")

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
            print("Error: problem in creating the geopanda dataframe. Perhaps some of the required columns of the stops.txt file are missing.")    
        else:
            self.stops = gdf
            if not hlp.check_dataframe(df):
                print("Warning: empty values or NaN values found in stops.txt. This might cause some issues.")

            # Extract the 'stop_id' column from the first DataFrame
            stop_ids_df1 = self.stops['stop_id']

            # Check if all stop_ids from df1 are present in df2
            all_present = stop_ids_df1.isin(self.stop_times['stop_id']).all()

            if not all_present:
                print("\t Warning: Some stops are not associated to any stop_time in the stop_times.txt file. The clean_stops() function could be used to solve this issue")
        
        file_path.close()


    """ Helpers """    

    def get_shape(self, trip_id):
        # Get the shape of the corresponding trip and project it into epsg:3857 crs
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')
        linestring = gdf.loc[gdf['trip_id'] == trip_id, 'geometry'].iloc[0]

        return linestring

    def get_stop_locations(self, trip_id):
        gdf = pd.merge(self.stop_times, self.stops[['stop_id', 'geometry']], on='stop_id', how='left')       
        coordinates = gdf.loc[gdf['trip_id'] == trip_id, 'geometry']

        return coordinates.tolist()


    """ Data check-up """

    def data_check(self):
        """ Checking the consistency of trips, routes, and stops  
        """
        print("\nChecking the gtfs data consistency:")

        problem_found = False

        # Trips

        # Extract the 'stop_id' column from the first DataFrame
        trip_ids = self.trips['trip_id']
        # Check if all stop_ids from df1 are present in df2
        all_present = trip_ids.isin(self.frequencies['trip_id']).all()
        if not all_present:
            problem_found = True
            print("\t Warning: Some trips are not associated to any frequency in the frequencies.txt file. The clean_trips() function could be used to solve this issue.")

        # Routes 

        # Extract the 'route_id' column from the first DataFrame
        route_ids = self.routes['route_id']
        # Check if all route_ids from df1 are present in df2
        all_present = route_ids.isin(self.trips['route_id']).all()
        if not all_present:
            problem_found = True
            print("\t Warning: Some routes are not associated to any trip in the trips.txt file. The clean_routes() function could be used to solve this issue.")

        # Stops

        # Extract the 'stop_id' column from the first DataFrame
        stop_ids_df1 = self.stops['stop_id']
        # Check if all stop_ids from df1 are present in df2
        all_present = stop_ids_df1.isin(self.stop_times['stop_id']).all()
        if not all_present:
            problem_found = True
            print("\t Warning: Some stops are not associated to any stop_time in the stop_times.txt file. The clean_stops() function could be used to solve this issue")

        print("\t -")    
        if problem_found:
            print("\t Problems found. If multiple warnings, the clean_all() method could be used.")
        else:
            print("\t No problems found.")        

        return problem_found


    """ Data cleaning """

    def clean_trips(self):
        """ Deleting the trips which are not associated to a frequency 
        """
        trips = self.trips
        frequencies = self.frequencies

        # Merge the DataFrames based on 'trip_id'
        merged_df = trips.merge(frequencies, on='trip_id')

        # Only keep the rows from the original df1 that have a corresponding trip_id in df2
        cleaned_trips = trips[trips['trip_id'].isin(merged_df['trip_id'])]

        self.trips = cleaned_trips

    def clean_stops(self):
        """ Deleting the stops which are not associated to any stop_times
        """
        stops = self.stops
        stop_times = self.stop_times

        cleaned_stops = stops[stops['stop_id'].isin(stop_times['stop_id'])]

        self.stops = cleaned_stops

    def clean_routes(self):
        """ Deleting the routes which are not associated to any trip
        """        
        routes = self.routes
        trips = self.trips

        cleaned_routes = routes[routes['route_id'].isin(trips['route_id'])]

        self.routes = cleaned_routes

    def clean_all(self):
        """ Executing all the cleaning functions 
        """
        self.clean_trips()
        self.clean_stops()
        self.clean_routes()


    """ General information about the feed """    

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


    """ Per trip metrics """

    def trip_length_km(self, trip_id):
        """ Calculates the lenght in km of a trip
        """
        # Get the shape of the corresponding trip and project it into epsg:3857 crs
        gdf = pd.merge(self.trips, self.shapes[['shape_id', 'geometry']], on='shape_id', how='left')
        linestring = gdf.loc[gdf['trip_id'] == trip_id, 'geometry'].iloc[0]

        web_mercator_projection = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform 

        linestring_projection = transform(web_mercator_projection, linestring)

        return linestring_projection.length / 1000.0

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
        """ Number of stops for each trip
        """        
        stop_times = self.stop_times       
        n_stops = stop_times.groupby('trip_id').size().reset_index(name='row_count')

        return n_stops.loc[n_stops['trip_id'] == trip_id, 'row_count'].iloc[0]


    """ Per stop metrics """

    def stop_frequencies(self):
        """ Returns the number of times a stop is used by all trips
        """
        stop_times = self.stop_times

        stop_counts = stop_times.groupby('stop_id').size().reset_index(name='count')
        stop_counts = pd.merge(stop_counts, self.stops[['stop_id', 'geometry']], on='stop_id', how='left')
        
        return stop_counts


    """ Global metrics assessment """

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

    """ Topology information """

    def nearest_point_distance_km(self):
        """ Distance to the closest stop
        """ 

        points = self.stops['geometry'].tolist()   

        df = pd.DataFrame(self.stops)

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
        """ Number of crossovers
        """
        shapes = self.shapes['geometry'].tolist()

        # Initialize a counter for crossovers
        crossovers_count = 0  

        # Assess the number of crossovers between all pairs of LineStrings
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                if shapes[i].intersects(shapes[j]):
                    crossovers_count += 1 

        return crossovers_count


    """ Filter functions """

    def filter_services(self, service_id):
        """ Number of crossovers
        """
        self.trips.drop(self.trips[self.trips['service_id'] == service_id].index, inplace=True)

        