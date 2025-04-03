# coding: utf-8

import pandas as pd
import sys
import multiprocessing as mp

from gtfs4ev.gtfsmanager import GTFSManager
from gtfs4ev.tripsimulator import TripSimulator
from gtfs4ev import helpers as hlp

class FleetSimulator:
    """
    A class to simulate a fleet of vehicles based on GTFS data.
    
    Attributes:
        gtfs_manager (GTFSManager): The GTFS manager handling transit data.
        trip_ids (list[str]): List of trip IDs to simulate.
    """
    
    def __init__(self, gtfs_manager: GTFSManager, trip_ids: str = None):
        """
        Initializes the FleetSimulator instance.

        Args:
            gtfs_manager (GTFSManager): The GTFS manager handling transit data.
            trip_ids (list[str], optional): The list of trip IDs to simulate. Defaults to all trips.
        """
        print("=========================================")
        print(f"INFO \t Creation of a FleetSimulator object.")
        print("=========================================")

        self.gtfs_manager = gtfs_manager
        self.trip_ids = trip_ids

        self._fleet_operation = None

        print("INFO \t Successful initialization of the FleetSimulator. The fleet operation can now be simulated. ")
    
    # Properties and Setters

    @property
    def gtfs_manager(self) -> GTFSManager:
        return self._gtfs_manager

    @gtfs_manager.setter
    def gtfs_manager(self, value):
        if not isinstance(value, object):  # Ideally, check against GTFSManager class
            raise ValueError("ERROR \t gtfs_manager must be a valid GTFSManager instance.")
        self._gtfs_manager = value
    
    @property
    def trip_ids(self):
        """Gets the list of trip IDs."""
        return self._trip_ids
    
    @trip_ids.setter
    def trip_ids(self, value):
        """Sets the list of trip IDs, ensuring they exist in the GTFS feed."""
        available_trips = set(self.gtfs_manager.trips["trip_id"].unique())
        
        if value is None:
            self._trip_ids = list(available_trips)
        else:
            invalid_trips = [trip for trip in value if trip not in available_trips]
            if invalid_trips:
                raise ValueError(f"ERROR \t Some trip IDs do not exist in the GTFS feed: {invalid_trips}")
            self._trip_ids = value

    @property
    def fleet_operation(self):
        """Gets the pd dataframe with fleet operation."""
        return self._fleet_operation
        
    def compute_fleet_operation(self, use_multiprocessing = True):
        """
        Runs the simulation for the selected trips and accumulates the results in a big dataframe.
        Uses parallel processing to speed up computation if 'use_multiprocessing' is True.
        """
        num_trips = len(self.trip_ids)
        print(f"INFO \t Computing fleet operation of {num_trips} trips...")

        # If use_multiprocessing is True, perform the computation in parallel
        if use_multiprocessing:
            # Create a shared manager to track the progress
            with mp.Manager() as manager:
                progress_counter = manager.Value('i', 0)  # Shared counter to track progress

                # Create multiprocessing pool and apply function
                with mp.Pool(mp.cpu_count()) as pool:
                    results = pool.starmap(
                        process_trip, 
                        [(trip_id, self.gtfs_manager, progress_counter, num_trips) for trip_id in self.trip_ids]
                    )
        else:
            # If no multiprocessing, compute trips sequentially
            results = []
            counter = 1
            for trip_id in self.trip_ids:
                tripsim = TripSimulator(gtfs_manager=self.gtfs_manager, trip_id=trip_id)
                tripsim.compute_fleet_operation()
                result = pd.DataFrame(tripsim._fleet_operation)

                sys.stdout.write(f"\r \t Progress: {counter}/{num_trips} trips.")
                sys.stdout.flush()

                counter += 1

                results.append(result)

        # Step 3: Merge all results **at once**
        self._fleet_operation = pd.concat(results, ignore_index=True)

        print("\n \t Fleet operation computation completed.")

def process_trip(trip_id, gtfs_manager, progress_counter, num_trips):
    """
    Function to process a single trip. This runs in parallel or sequentially.
    """
    # Process the trip (simulation)
    tripsim = TripSimulator(gtfs_manager=gtfs_manager, trip_id=trip_id)
    tripsim.compute_fleet_operation()
    result = pd.DataFrame(tripsim._fleet_operation)
    
    if progress_counter is not None:
        # Update progress
        progress_counter.value += 1  # Increment the shared counter directly
        # Print progress
        sys.stdout.write(f"\r \t Progress: {progress_counter.value}/{num_trips} trips.")
        sys.stdout.flush()

    return result