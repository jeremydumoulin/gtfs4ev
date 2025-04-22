# coding: utf-8

import numpy as np
import pandas as pd
import random
from datetime import date, datetime, timedelta

from gtfs4ev.fleetsimulator import FleetSimulator

class ChargingSimulator:
    """A class to simulate the scenario-based charging of the EV fleet and the related implications."""

    def __init__(self, fleet_sim: FleetSimulator, energy_consumption_kWh_per_km: float, charging_efficiency: float = 0.9, charging_powers_kW: dict = None):
        """
        Initializes the Charging Simulator.

        Args:
            fleet_sim: An instance of a FleetSimulator class containing vehicle operations.
            vehicle_properties (dict): A dictionary of vehicle properties (e.g., energy consumption, battery capacity).
            charging_powers_kw (dict): A dictionary of vehicle_id: charging power in kW.
            charging_efficiency (float): Charging efficiency as a value between 0 and 1 (default is 0.0).
        """
        print("=========================================")
        print(f"INFO \t Creation of a ChargingSimulator object.")
        print("=========================================")

        self.fleet_sim = fleet_sim
        self.energy_consumption_kWh_per_km = energy_consumption_kWh_per_km
        self.charging_efficiency = charging_efficiency
        self.charging_powers_kW = charging_powers_kW or {}

        print("INFO \t Successful initialization of the ChargingSimulator. ")

    # --- fleet_sim ---
    @property
    def fleet_sim(self):
        return self._fleet_sim

    @fleet_sim.setter
    def fleet_sim(self, value):
        if value.fleet_operation is None:
            raise ValueError("fleet_sim.fleet_operation cannot be None. You must simulate the fleet opeation first.")
        self._fleet_sim = value

    # --- vehicle_properties ---


    def compute_charging_schedule(self, charging_strategy: str, **kwargs):
        """
        Reconstructs travel sequences for all vehicles.
        Then, for every vehicle, applies a charging scenario based on the provided charging function.

        Parameters:
        - charging_function (callable): A function that takes a travel sequence and additional parameters, and returns the charging sequence.
        - *charging_args (tuple): Additional arguments to be passed to the charging function.

        Returns:
        - DataFrame with vehicle_id, and a list of charging events as dictionaries.
        """
        print("INFO \t Computing the charging schedule for all vehicles... ")

        fleet_operation = self.fleet_sim.fleet_operation
        trip_travel_sequences = self.fleet_sim.trip_travel_sequences

        trip_base_sequences = {}
        for _, event in trip_travel_sequences.iterrows():
            trip_id = event["trip_id"]
            trip_base_sequences.setdefault(trip_id, []).append(event)

        vehicle_charging_sequences = []

        for _, vehicle_record in fleet_operation.iterrows():
            vehicle_id = vehicle_record["vehicle_id"]
            trip_id = vehicle_record.get("trip_id", None)
            vehicle_sequences = vehicle_record["travel_sequences"]
            base_seq = trip_base_sequences.get(trip_id, [])

            base_durations = [event["duration"] for event in base_seq]
            base_distances = [event.get("distance", 0.0) for event in base_seq]
            base_statuses = [event["status"] for event in base_seq]
            base_locations = [event.get("location", None) for event in base_seq]
            base_stop_ids = [event.get("stop_id", None) for event in base_seq]
            base_cumulative = np.concatenate(([0], np.cumsum(base_durations)))

            travel_sequence = []

            for seq in vehicle_sequences:
                start_time = datetime.strptime(seq["start_time"], "%H:%M:%S")
                end_time = datetime.strptime(seq["end_time"], "%H:%M:%S")
                if end_time < start_time:
                    end_time += timedelta(days=1)
                duration_sec = int((end_time - start_time).total_seconds())

                if not seq.get("operating", True):
                    travel_sequence.append({
                        "start_time": start_time.strftime("%H:%M:%S"),
                        "end_time": end_time.strftime("%H:%M:%S"),
                        "status": "not_operating",
                        "duration_h": duration_sec / 3600,
                        "distance_km": 0.0,
                        "stop_id": None
                    })
                    continue

                for i, event in enumerate(base_seq):
                    event_start_offset = base_cumulative[i]
                    event_end_offset = base_cumulative[i + 1]

                    if event_start_offset >= duration_sec:
                        break

                    effective_start = start_time + timedelta(seconds=event_start_offset)
                    effective_end = start_time + timedelta(seconds=min(event_end_offset, duration_sec))
                    event_duration = (effective_end - effective_start).total_seconds()
                    base_duration = base_durations[i]

                    base_distance = base_distances[i]
                    if base_duration > 0:
                        distance_km = base_distance * (event_duration / base_duration)
                    else:
                        distance_km = 0.0

                    travel_sequence.append({
                        "start_time": effective_start.strftime("%H:%M:%S"),
                        "end_time": effective_end.strftime("%H:%M:%S"),
                        "status": base_statuses[i],
                        "duration_h": event_duration / 3600,
                        "distance_km": distance_km,
                        "stop_id": base_stop_ids[i]
                    })

            # Calculate total charging need based on the travel distance, energy consumption, and charging efficiency
            charging_need_kWh = sum(event["distance_km"] * self.energy_consumption_kWh_per_km for event in travel_sequence)

            charging_events = []       # List to store accepted charging events
            remaining_need = charging_need_kWh
            added_intervals = []       # Tracks time intervals of already accepted charging events

            # Apply each charging strategy in order of priority until the energy need is fulfilled
            for strategy in charging_strategy:
                if remaining_need <= 0:
                    break  # Stop if no more energy is needed

                # Generate candidate charging events based on the current strategy
                events = self.get_charging_sequence(
                    travel_sequence, strategy, remaining_need, **kwargs
                )

                for event in events:
                    # Convert time strings to datetime objects for comparison
                    event_start_time = datetime.strptime(event["start_time"], "%H:%M:%S")
                    event_end_time = datetime.strptime(event["end_time"], "%H:%M:%S")

                    # Check if this event overlaps with any previously accepted event
                    overlaps = any(
                        not (event_end_time <= existing_start or event_start_time >= existing_end)
                        for (existing_start, existing_end) in added_intervals
                    )

                    if overlaps:
                        continue  # Skip overlapping events to preserve higher-priority decisions

                    # Accept this event: add it to the charging list and register its time interval
                    charging_events.append(event)
                    added_intervals.append((event_start_time, event_end_time))

                # Update the remaining energy need based on accepted charging events so far
                charged_energy = sum(e["energy_charged_kWh"] for e in charging_events)
                remaining_need = charging_need_kWh - charged_energy

            # Ensure charging events are returned in chronological order
            charging_events.sort(key=lambda e: e["start_time"])

            min_capacity = self.find_minimum_battery_capacity(travel_sequence, charging_events)

            vehicle_charging_sequences.append({
                "vehicle_id": vehicle_id,
                "charging_sequence": charging_events,
                "charging_need_kWh": charging_need_kWh,
                "remaining_need_kWh": remaining_need,
                "min_capacity_kWh": min_capacity
            })

        self._charging_schedule = pd.DataFrame(vehicle_charging_sequences)

    def get_charging_sequence(self, travel_sequence, charging_strategy, charging_need_kWh, charge_probability, depot_travel_time_min):
        """
        Simulates charging events based on the travel sequence and chosen strategy.
        Includes probabilistic arrival/departure offsets for depot time.
        """
        charging_events = []
        # Apply random delays to consider travel time to depot
        delay = timedelta(minutes=random.randint(depot_travel_time_min[0], depot_travel_time_min[1]))

        if charging_strategy == "terminal":
            remaining_need = charging_need_kWh 

            for event in travel_sequence:
                if remaining_need <= 0:
                    break  # charging need fulfilled

                if event["status"] == "at_terminal":
                    if random.random() < charge_probability:
                        power_rate = self.get_random_charging_power("terminal")
                        duration_h = event["duration_h"]
                        energy = duration_h * power_rate
                        charging_events.append({
                            "start_time": event["start_time"],
                            "end_time": event["end_time"],
                            "location": "terminal",
                            "stop_id": event["stop_id"],
                            "power": power_rate,
                            "energy_charged_kWh": energy
                        })
                        remaining_need -= energy

        if charging_strategy == "depot_day":
            remaining_need = charging_need_kWh            

            for idx, event in enumerate(travel_sequence):             
                
                # Skip first and last events
                if idx == 0 or idx == len(travel_sequence) - 1:
                    continue

                if event["status"] == "not_operating" and remaining_need > 0:
                    power_rate = self.get_random_charging_power("depot")

                    start_dt = datetime.strptime(event["start_time"], "%H:%M:%S")
                    end_dt = datetime.strptime(event["end_time"], "%H:%M:%S")                    

                    adjusted_start = start_dt + delay
                    adjusted_end = end_dt - delay

                    if adjusted_end < adjusted_start:
                        continue

                    # Available duration in hours
                    available_duration_h = (adjusted_end - adjusted_start).total_seconds() / 3600

                    if available_duration_h > 0:
                        # Max energy we can charge during this slot
                        max_possible_energy = available_duration_h * power_rate
                        energy_to_charge = min(max_possible_energy, remaining_need)
                        charging_duration_h = energy_to_charge / power_rate

                        actual_end = adjusted_start + timedelta(hours=charging_duration_h)

                        charging_events.append({
                            "start_time": adjusted_start.strftime("%H:%M:%S"),
                            "end_time": actual_end.strftime("%H:%M:%S"),
                            "location": "depot",
                            "power": power_rate,
                            "energy_charged_kWh": energy_to_charge
                        })

                        remaining_need -= energy_to_charge
                        if remaining_need <= 0:
                            break  # charging need fulfilled

        if charging_strategy == "depot_night":
            remaining_need = charging_need_kWh
                
            power_rate = self.get_random_charging_power("depot") # One power rate for both last and first sequences

            if remaining_need > 0:
                # --- End of Day ---
                last_event = travel_sequence[-1]
                start_dt = datetime.strptime(last_event["start_time"], "%H:%M:%S")
                end_dt = datetime.strptime(last_event["end_time"], "%H:%M:%S")

                adjusted_start = start_dt + delay
                duration_needed = timedelta(hours=remaining_need / power_rate)

                if adjusted_start + duration_needed <= end_dt:
                    charging_events.append({
                        "start_time": adjusted_start.strftime("%H:%M:%S"),
                        "end_time": (adjusted_start + duration_needed).strftime("%H:%M:%S"),
                        "location": "depot",
                        "power": power_rate,
                        "energy_charged_kWh": remaining_need
                    })
                    remaining_need = 0
                else:
                    available_h = (end_dt - adjusted_start).total_seconds() / 3600
                    if available_h > 0:
                        energy_end_of_day = available_h * power_rate
                        charging_events.append({
                            "start_time": adjusted_start.strftime("%H:%M:%S"),
                            "end_time": end_dt.strftime("%H:%M:%S"),
                            "location": "depot",
                            "power": power_rate,
                            "energy_charged_kWh": energy_end_of_day
                        })
                        remaining_need -= energy_end_of_day

                # --- Beginning of Day ---
                if remaining_need > 0:
                    first_event = travel_sequence[0]
                    start_dt = datetime.strptime(first_event["start_time"], "%H:%M:%S")
                    end_dt = datetime.strptime(first_event["end_time"], "%H:%M:%S")

                    adjusted_end = end_dt - delay

                    available_h = (adjusted_end - start_dt).total_seconds() / 3600
                    if available_h > 0:
                        energy_morning = min(available_h * power_rate, remaining_need)
                        actual_duration = timedelta(hours=energy_morning / power_rate)

                        charging_events.append({
                            "start_time": start_dt.strftime("%H:%M:%S"),
                            "end_time": (start_dt + actual_duration).strftime("%H:%M:%S"),
                            "location": "depot",
                            "power": power_rate,
                            "energy_charged_kWh": energy_morning
                        })

                        remaining_need -= energy_morning

        charging_events.sort(key=lambda e: e["start_time"])                

        return charging_events

    # Helpers

    def get_random_charging_power(self, location_type: str) -> float:
        """
        Randomly selects a charging power based on the location type and its distribution.
            Args:
            location_type (str): The type of location ("depot", "terminal", etc.)

        Returns:
            float: Selected charging power in kW.
        """
        if location_type not in self.charging_powers_kW:
            raise ValueError(f"Unknown location type '{location_type}'")

        options = self.charging_powers_kW[location_type]
        powers = [item[0] for item in options]
        probabilities = [item[1] for item in options]

        return random.choices(powers, weights=probabilities, k=1)[0]

    def find_minimum_battery_capacity(self, travel_sequence, charging_events):
        def to_time(t): return datetime.strptime(t, "%H:%M:%S")

        # Merge and sort all events by time
        events = []

        # Energy consumption 
        energy_consumption_kWh_per_km = self.energy_consumption_kWh_per_km

        for event in travel_sequence:
            if event["status"] == "travelling":
                energy = -event["distance_km"] * energy_consumption_kWh_per_km
                events.append((to_time(event["start_time"]), energy))

        for event in charging_events:
            energy = event["energy_charged_kWh"]
            events.append((to_time(event["start_time"]), energy))

        # Sort all events chronologically
        events.sort(key=lambda x: x[0])

        # Simulate battery usage
        battery_level = 0.0
        min_battery_level = 0.0
        max_battery_level = 0.0

        for time, energy in events:
            battery_level += energy
            min_battery_level = min(min_battery_level, battery_level)
            max_battery_level = max(max_battery_level, battery_level)

        # Battery profile is shifted so the minimum is 0
        required_capacity = max_battery_level - min_battery_level

        return required_capacity

    # def test_battery_capacity(self, travel_sequence, charging_events, energy_consumption_kWh_per_km, initial_battery_capacity):
    #     def to_time(t): return datetime.strptime(t, "%H:%M:%S")

    #     # Convert charging times and sort them
    #     charging_times = [to_time(e["start_time"]) for e in charging_events]
    #     charging_times = sorted(charging_times)

    #     # Get earliest and latest travel times
    #     travel_start = to_time(travel_sequence[1]["start_time"])
    #     travel_end = to_time(travel_sequence[-2]["end_time"])

    #     # Add dummy charge times to cover full day
    #     all_checkpoints = [travel_start] + charging_times + [travel_end]
    #     all_checkpoints = sorted(all_checkpoints)

    #     # Handle case where there's only one charging event
    #     if len(charging_events) == 1:
    #         # Total distance travelled in the journey
    #         total_distance = sum(event["distance_km"] for event in travel_sequence)
    #         # Energy required for the total distance
    #         required_capacity = total_distance * energy_consumption_kWh_per_km
    #         print(f"Only one charging event. Minimum battery capacity needed: {required_capacity:.2f} kWh")
    #         return required_capacity  # Return the calculated required capacity

    #     # Battery status and current battery level
    #     battery_capacity = initial_battery_capacity 
    #     battery_level = battery_capacity

    #     print(f"Starting with battery capacity: {battery_capacity} kWh")

    #     # Loop over intervals between charging sessions
    #     for i in range(len(all_checkpoints) - 1):
    #         interval_start = all_checkpoints[i]
    #         interval_end = all_checkpoints[i + 1]
    #         energy_needed = 0.0

    #         # Calculate energy consumption between charging events
    #         for event in travel_sequence:
    #             event_start = to_time(event["start_time"])
    #             event_end = to_time(event["end_time"])

    #             if event_end <= interval_start or event_start >= interval_end:
    #                 continue  # Skip events outside this interval

    #             if event["status"] == "travelling":
    #                 energy_needed += event["distance_km"] * energy_consumption_kWh_per_km

    #         # Check if the battery level is sufficient for this leg
    #         if battery_level < energy_needed:
    #             print(f"Battery ran out! Needed {energy_needed:.2f} kWh, but only {battery_level:.2f} kWh left.")
    #             return False  # The battery was not sufficient

    #         # If battery level is sufficient, reduce battery by the energy consumed
    #         battery_level -= energy_needed

    #         print(f"Traveling from {interval_start.strftime('%H:%M:%S')} → {interval_end.strftime('%H:%M:%S')}")
    #         print(f"Energy needed: {energy_needed:.2f} kWh | Remaining battery: {battery_level:.2f} kWh")

    #         # Refill battery at the charging stop
    #         if interval_end in charging_times:
    #             print(f"Charging at {interval_end.strftime('%H:%M:%S')}")
    #             # Find the corresponding charging event and add the energy charged
    #             for charge_event in charging_events:
    #                 if to_time(charge_event["start_time"]) == interval_end:
    #                     battery_level += charge_event["energy_charged_kWh"]
    #                     print(f"Energy charged: {charge_event['energy_charged_kWh']} kWh | Total battery: {battery_level:.2f} kWh")
    #                     break

    #     print("Battery successfully handled the entire trip.")
    #     return True  # The battery was sufficient for the entire trip


        # def compute_charging_schedule(self):
    #     """
    #     Simulates charging behavior for a fleet of vehicles.
    #     Charges only at 11kW when the vehicle status is 'at_terminal'.

    #     Parameters:
    #     - fleet_operation: List of dicts, each with key 'travel_sequences'

    #     Returns:
    #     - charging_schedules: DataFrame with vehicle_id, the corresponding charging sequence, and total energy charged
    #     """
    #     charging_power_kW = 11  # constant charging rate
    #     energy_consumption = 0.5 # kWh per km
    #     charging_schedules = []  # List to hold each vehicle's charging schedule

    #     fleet_operation = self.fleet_sim.fleet_operation 
    #     trip_travel_sequences = self.fleet_sim.trip_travel_sequences  

    #     # Step 1: Organize base sequences by trip_id
    #     trip_base_sequences = {}
    #     for _, event in trip_travel_sequences.iterrows():
    #         trip_id = event["trip_id"]
    #         trip_base_sequences.setdefault(trip_id, []).append(event)

    #     # Loop through vehicles in fleet_operation
    #     for veh_idx, vehicle_record in fleet_operation.iterrows():
    #         charging_schedule = []
    #         total_energy_charged = 0  # Variable to accumulate the total energy charged for the vehicle

    #         # Get the travel_sequences for the current vehicle
    #         vehicle_travel_sequences = vehicle_record["travel_sequences"]
    #         trip_id = vehicle_record["trip_id"]

    #         for seq in vehicle_travel_sequences:
    #             # Check if the sequence is not operating (skip if operating is False)
    #             if not seq.get("operating", True):
    #                 continue  # Skip this sequence if it's not operating

    #             # Retrieve the base sequence for the current vehicle and trip_id
    #             base_seq = trip_base_sequences.get(trip_id)                
                    
    #             # Extract start time and compute the cumulative durations for base sequence
    #             start_time = datetime.strptime(seq["start_time"], "%H:%M:%S")
    #             durations = [e["duration"] for e in base_seq]
    #             cumulative_durations = np.cumsum([0] + durations[:-1])

    #             # Loop through the base sequence to handle charging during "at_terminal" status
    #             for event, offset in zip(base_seq, cumulative_durations):
    #                 if event["status"] != "at_terminal":
    #                     continue  # Only process "at_terminal" events

    #                 # Compute event's start and end time based on cumulative duration offset
    #                 event_start = start_time + timedelta(seconds=offset)
    #                 event_end = event_start + timedelta(seconds=event["duration"])
                    
    #                 # Calculate charging power (kW) and energy consumption (kWh)
    #                 duration_h = event["duration"] / 3600
    #                 energy_kWh = duration_h * charging_power_kW

    #                 # Add the energy to the total energy charged for the vehicle
    #                 total_energy_charged += energy_kWh

    #                 # Append the charging event to the charging schedule
    #                 charging_schedule.append({
    #                     "start_time": event_start.strftime("%H:%M:%S"),
    #                     "end_time": event_end.strftime("%H:%M:%S"),
    #                     "status": event["status"],
    #                     "power_kW": charging_power_kW,
    #                     "energy_kWh": energy_kWh,
    #                     "duration_h": duration_h,
    #                     "stop_id": event.get("stop_id", None)
    #                 })

    #         # Append the charging schedule and total energy charged for the current vehicle to the list
    #         charging_schedules.append({
    #             "vehicle_id": vehicle_record["vehicle_id"],
    #             "trip_id": trip_id,  
    #             "charging_sequence": charging_schedule,  # Store the entire charging sequence for this vehicle
    #             "total_charged_energy_kWh": total_energy_charged,  # Store the total energy charged,
    #             "charging_demand_kWh": vehicle_record["total_distance_km"] * energy_consumption
    #         })

    #     # Convert the list of dictionaries into a DataFrame and return it
    #     return pd.DataFrame(charging_schedules)