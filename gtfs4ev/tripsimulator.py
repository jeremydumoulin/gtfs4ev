# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from shapely.ops import transform
from shapely.ops import substring
import pyproj
from pyproj import Geod
from shapely.geometry import LineString, Point
import folium
from folium.plugins import TimestampedGeoJson

from gtfs4ev.gtfsmanager import GTFSManager
from gtfs4ev import helpers as hlp

class TripSimulator:
    """Simulates the operation of a bus trip based on GTFS data."""

    def __init__(self, gtfs_manager: GTFSManager, trip_id: str):
        """
        Initializes the TripSimulator with GTFS data and a specific trip.

        Parameters:
        - gtfs_manager (GTFSManager): An instance of GTFSManager holding GTFS data.
        - trip_id (str): The trip ID to be simulated.
        """
        self.gtfs_manager = gtfs_manager
        self.trip_id = trip_id

        self._fleet_operation = None        

    # Properties and Setters

    @property
    def gtfs_manager(self) -> GTFSManager:
        return self._gtfs_manager

    @gtfs_manager.setter
    def gtfs_manager(self, value):
        if not isinstance(value, object):  # Ideally, check against GTFSManager class
            raise ValueError("gtfs_manager must be a valid GTFSManager instance.")
        self._gtfs_manager = value

    @property
    def trip_id(self) -> str:
        return self._trip_id

    @trip_id.setter
    def trip_id(self, value):
        if self.gtfs_manager is None:
            raise ValueError("gtfs_manager must be set before setting trip_id.")
        if value not in self.gtfs_manager.trips['trip_id'].values:
            raise ValueError(f"Trip ID {value} not found in GTFS data.")
        self._trip_id = value

    @property
    def fleet_operation(self) -> pd.DataFrame:
        """pd.DataFrame: The fleet operation data."""
        return self._fleet_operation

    # Number of vehicles in operation

    def trip_duration_sec(self) -> int:
        """Returns the duration of the trip in seconds."""
        trip_stop_times = self.gtfs_manager.stop_times[self.gtfs_manager.stop_times['trip_id'] == self.trip_id]
        duration = (trip_stop_times['departure_time'].iloc[-1] - trip_stop_times['arrival_time'].iloc[0]).total_seconds()

        return duration

    def max_vehicles_in_operation(self) -> int:
        """
        Estimates the number of buses required to maintain the highest frequency in a given trip.

        Returns:
        - int: Estimated number of buses needed.
        """
        # Get the frequencies for this trip
        trip_frequencies = self.gtfs_manager.frequencies[self.gtfs_manager.frequencies['trip_id'] == self.trip_id]

        if trip_frequencies.empty:
            raise ValueError(f"No frequency data available for trip ID {self.trip_id}.")

        # Find the highest frequency (lowest headway)
        min_headway = trip_frequencies['headway_secs'].min()

        # Compute the number of vehicles needed
        max_num_vehicles = self.trip_duration_sec() / min_headway

        return max(1, round(max_num_vehicles))

    # Operation of the vehicles along the trip

    def _precompute_static_data(self):
        """
        Precomputes static data for a given trip, including stop times, stop geometries,
        and distances along the trip shape.
        """

        # Extract stop times for the current trip
        stop_times = self.gtfs_manager.stop_times[self.gtfs_manager.stop_times['trip_id'] == self.trip_id].copy()
        
        # Sort stop times in order of arrival time
        stop_times.sort_values(by='arrival_time', inplace=True)

        # Define the reference (first) stop time to compute offsets
        first_stop_time = stop_times.iloc[0]['arrival_time']

        # Compute arrival and departure offsets in seconds from the first stop
        stop_times['arrival_offset'] = (stop_times['arrival_time'] - first_stop_time).dt.total_seconds()
        stop_times['departure_offset'] = (stop_times['departure_time'] - first_stop_time).dt.total_seconds()

        # Store the updated stop times dataframe
        self.stop_times = stop_times

        # Get the shape of the trip (a LineString representing the trip path)
        self.trip_linestring = self.gtfs_manager.get_shape(self.trip_id)

        # Store stop geometries as a dictionary {stop_id: geometry}
        self.stop_geometries = {
            row['stop_id']: self.gtfs_manager.stops.loc[self.gtfs_manager.stops['stop_id'] == row['stop_id'], 'geometry'].iloc[0]
            for _, row in stop_times.iterrows()
        }

        # Compute the distances along the trip linestring for each stop
        self.stop_distances = {
            stop_id: self.trip_linestring.project(hlp.find_closest_point(self.trip_linestring, geom))
            for stop_id, geom in self.stop_geometries.items()
        }

        # Compute average speeds between stops
        stop_ids = stop_times['stop_id'].values
        distances = np.array([self.stop_distances[sid] for sid in stop_ids])
        arrival = stop_times['arrival_offset'].values
        departure = stop_times['departure_offset'].values
        
        self.avg_speeds = np.zeros(len(stop_ids) - 1)
        
        for i in range(len(stop_ids) - 1):
            time_diff = arrival[i + 1] - departure[i]  # Time between stops in seconds
            if time_diff > 0:
                distance_km = hlp.length_km(substring(self.trip_linestring, distances[i], distances[i + 1]))
                self.avg_speeds[i] = distance_km / (time_diff / 3600)  # Convert to km/h
            else:
                self.avg_speeds[i] = 0  # Avoid division by zero

    def get_vehicle_status_vectorized(self, durations, time_step: int):
        """
        Determines the status, location, traveled distance, and speed of vehicles.
        """
        durations = np.array(durations)
        n = durations.shape[0]
        
        arrival = self.stop_times['arrival_offset'].values  
        departure = self.stop_times['departure_offset'].values  
        stop_ids = self.stop_times['stop_id'].values  
        distances = np.array([self.stop_distances[sid] for sid in stop_ids])  
        
        statuses = np.empty(n, dtype=object)
        current_locations = [None] * n
        traveled_distances = np.zeros(n)
        speeds = np.zeros(n)  # Stores speeds in km/h
        
        valid_mask = (durations >= 0) & (durations <= departure[-1])
        
        statuses[~valid_mask] = None
        for idx in np.where(~valid_mask)[0]:
            current_locations[idx] = None
            traveled_distances[idx] = 0
            speeds[idx] = 0
        
        valid_idx = np.where(valid_mask)[0]
        valid_durations = durations[valid_mask]
        
        at_stop_matrix = (valid_durations[:, None] >= arrival[None, :]) & (valid_durations[:, None] <= departure[None, :])
        has_stop = at_stop_matrix.any(axis=1)
        stop_indices = np.argmax(at_stop_matrix, axis=1)
        search_idx = np.searchsorted(departure, valid_durations, side='left')
        
        prev_geom = None  
        
        for j, d in enumerate(valid_durations):
            global_idx = valid_idx[j]  
            
            if has_stop[j]:  
                s_idx = stop_indices[j]  
                stop_id = stop_ids[s_idx]  
                statuses[global_idx] = "at_terminal" if s_idx in [0, len(stop_ids) - 1] else "at_stop"
                curr_geom = self.stop_geometries[stop_id]  
                speeds[global_idx] = 0
            else:  
                i = search_idx[j]  
                
                if i == 0 or i >= len(departure):
                    statuses[global_idx] = None
                    current_locations[global_idx] = None
                    traveled_distances[global_idx] = 0
                    speeds[global_idx] = 0
                    prev_geom = None
                    continue
                
                statuses[global_idx] = "travelling"
                frac = (d - departure[i - 1]) / (arrival[i] - departure[i - 1])
                current_distance = distances[i - 1] + frac * (distances[i] - distances[i - 1])
                curr_geom = self.trip_linestring.interpolate(current_distance)
                speeds[global_idx] = self.avg_speeds[i - 1]
            
            current_locations[global_idx] = curr_geom  
            
            if prev_geom is not None:
                start_distance = self.trip_linestring.project(prev_geom)
                end_distance = self.trip_linestring.project(curr_geom)
                
                if start_distance > end_distance:  
                    sub_linestring1 = substring(self.trip_linestring, start_distance, self.trip_linestring.length)
                    sub_linestring2 = substring(self.trip_linestring, 0, end_distance)
                    traveled_distances[global_idx] = hlp.length_km(sub_linestring1) + hlp.length_km(sub_linestring2)
                elif start_distance != end_distance:
                    sub_linestring = substring(self.trip_linestring, start_distance, end_distance)
                    traveled_distances[global_idx] = hlp.length_km(sub_linestring)
                else:
                    traveled_distances[global_idx] = 0
            
            prev_geom = curr_geom  
        
        return statuses, current_locations, traveled_distances, speeds

    def simulate_fleet_operation(self, time_step: int) -> pd.DataFrame:
        """
        Optimized simulation of fleet operation using vectorized vehicle status computation.
        """
        self._precompute_static_data()

        trip_duration = self.trip_duration_sec()
        num_vehicles = self.max_vehicles_in_operation()
        # Generate the time range for simulation
        time_range = pd.date_range("00:00:00", "23:59:59", freq=f"{time_step}S").time
        fleet_data = np.empty((num_vehicles, len(time_range)), dtype=object)
        fleet_data.fill(None)
        trip_frequencies = self.gtfs_manager.frequencies[self.gtfs_manager.frequencies['trip_id'] == self.trip_id]

        print(f"INFO \t Simulating fleet operation for trip \"{self.trip_id}\" with {len(trip_frequencies)} frequency intervals...")

        for _, row in trip_frequencies.iterrows():
            start_time, end_time = row['start_time'].time(), row['end_time'].time()
            headway = row['headway_secs']
            vehicles_in_operation = max(1, round(trip_duration / headway))
            active_vehicles = np.arange(min(num_vehicles, vehicles_in_operation))

            # Efficiently determine time indices for the current frequency interval.
            time_range_sec = np.array([t.hour * 3600 + t.minute * 60 + t.second for t in time_range])
            start_sec = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
            end_sec = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
            time_indices = np.where((time_range_sec >= start_sec) & (time_range_sec <= end_sec))[0]

            # Compute start offsets for vehicles.
            vehicle_start_offsets = active_vehicles * headway

            # For each active vehicle, compute statuses using the vectorized method.
            for veh, offset in zip(active_vehicles, vehicle_start_offsets):
                # Compute duration from start for each time index.
                durations = ((time_indices * time_step) - offset) % trip_duration
                # Compute current status for these durations.
                statuses, locations, distances, speeds = self.get_vehicle_status_vectorized(np.maximum(durations, 0), time_step)
                # Package the results in a dictionary per time step.
                vehicle_statuses = [
                    {"status": s, "current_location": loc, "travelled_distance_km": dist, "speed_km_per_h": speed}
                    for s, loc, dist, speed in zip(statuses, locations, distances, speeds)
                ]
                fleet_data[veh, time_indices] = vehicle_statuses

        self._fleet_operation = pd.DataFrame(fleet_data, index=range(num_vehicles), columns=time_range)

        print("\t Fleet simulation completed.")
        
        return self._fleet_operation

    # Export and visualisation

    def create_map_with_slider(self):
        """
        Generates an interactive folium map with a time slider using the simulated fleet operation data.

        This function only works if the time step is exactly 2 minutes.
        """
        if self._fleet_operation is None:
            raise ValueError("Fleet operation data is not available. Run simulate_fleet_operation() first.")

        # Transform the fleet operation DataFrame into a long-format DataFrame
        data = []
        for vehicle_id in self._fleet_operation.index:
            for time, status in self._fleet_operation.loc[vehicle_id].items():
                if not isinstance(status, dict) or status.get("current_location") is None:
                    continue  # Skip invalid data
                
                data.append({
                    "vehicle_id": vehicle_id,
                    "time": datetime.combine(datetime.today(), time),
                    "latitude": status["current_location"].y,
                    "longitude": status["current_location"].x
                })

        # Convert to DataFrame and check validity
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("No valid location data to plot.")

        # Sort and clean data
        df.sort_values(by=["time", "vehicle_id"], inplace=True)
        df.drop_duplicates(subset=["time", "vehicle_id"], inplace=True)

        # Calculate time step in minutes
        unique_times = df["time"].drop_duplicates().sort_values().reset_index(drop=True)
        if len(unique_times) < 2:
            raise ValueError("Not enough timestamps to determine time step.")

        time_diffs = unique_times.diff().dropna()
        min_step = int(min(time_diffs).total_seconds() // 60)  # in minutes

        if min_step != 2:
            raise ValueError("Invalid time step. This function only works with a 2-minute time step.")

        # Build ISO 8601 strings
        period_str = "PT2M"
        duration_str = "PT1M"  # Avoid PT0M

        # Create base map
        first_position = df.iloc[0]
        m = folium.Map(location=[first_position['latitude'], first_position['longitude']], zoom_start=12)

        # Add trip shape (polyline) to map
        trip_linestring = self.gtfs_manager.get_shape(self.trip_id)

        if trip_linestring and not trip_linestring.is_empty:
            if hasattr(trip_linestring, "coords"):
                shape_coords = [(lat, lon) for lon, lat in trip_linestring.coords]  # Folium expects (lat, lon)
                folium.PolyLine(
                    shape_coords,
                    color="black",
                    weight=4,
                    opacity=0.7,
                    tooltip="Trip Shape"
                ).add_to(m)

        # Organize data into time-based FeatureCollections
        time_grouped = df.groupby("time")
        features = []

        for timestamp, group in time_grouped:
            vehicle_points = []

            # Detect overlapping coordinates
            grouped_locations = group.groupby(["latitude", "longitude"])
            
            for (lat, lon), sub_group in grouped_locations:
                num_vehicles = len(sub_group)

                if num_vehicles > 1:
                    # Apply small jitter (random offset) to each overlapping point
                    jitter_offsets = np.linspace(-0.0001, 0.0001, num_vehicles)  # Small lat/lon shifts
                    
                    for (idx, (offset, row)) in enumerate(zip(jitter_offsets, sub_group.itertuples())):
                        vehicle_points.append({
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [lon + offset, lat + offset]  # Apply small shift
                            },
                            "properties": {
                                "time": timestamp.isoformat(),
                                "popup": f"Vehicle {row.vehicle_id}\nTime: {timestamp.strftime('%H:%M:%S')}"
                            }
                        })
                else:
                    # No duplicates, keep original position
                    row = sub_group.iloc[0]
                    vehicle_points.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        },
                        "properties": {
                            "time": timestamp.isoformat(),
                            "popup": f"Vehicle {row.vehicle_id}\nTime: {timestamp.strftime('%H:%M:%S')}"
                        }
                    })

            features.append({
                "type": "FeatureCollection",
                "features": vehicle_points
            })

        # Add dynamic time slider
        TimestampedGeoJson({
            "type": "FeatureCollection",
            "features": features
        },
        period=period_str,
        duration=duration_str,
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options='YYYY-MM-DD HH:mm:ss',
        time_slider_drag_update=True).add_to(m)

        return m

