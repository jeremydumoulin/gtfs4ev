# coding: utf-8

from gtfs4ev.fleetsimulator import FleetSimulator

class ChargingSimulator:
    """A class to simulate the scenario-based charging of the EV fleet and the related implications."""

    def __init__(self, fleet_sim: FleetSimulator, vehicle_properties: dict, charging_powers_kw: dict, charging_strategy: str = "at_depot", charging_efficiency: float = 0.9):
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
        self.vehicle_properties = vehicle_properties
        self.charging_powers_kw = charging_powers_kw
        self.charging_efficiency = charging_efficiency

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
    @property
    def vehicle_properties(self):
        return self._vehicle_properties

    @vehicle_properties.setter
    def vehicle_properties(self, value: dict):
        required_keys = {
            "energy_consumption_kwh_per_km",
            "additional_travel_distance_km",
            "depot_availability_hours"
        }
        missing = required_keys - value.keys()
        if missing:
            raise ValueError(f"Missing vehicle property keys: {missing}")

        depot_hours = value["depot_availability_hours"]
        if not isinstance(depot_hours, (list, tuple)) or len(depot_hours) != 3:
            raise ValueError("depot_availability_hours must be a list or tuple with three values: [start, stop, randomness].")

        self._vehicle_properties = value

    # --- charging_powers_kw ---
    @property
    def charging_powers_kw(self):
        return self._charging_powers_kw

    @charging_powers_kw.setter
    def charging_powers_kw(self, value: dict):
        for loc, options in value.items():
            if not isinstance(options, list):
                raise ValueError(f"charging_powers_kw[{loc}] must be a list of [power_kW, share] pairs.")
            for option in options:
                if not (isinstance(option, list) and len(option) == 2):
                    raise ValueError(f"Each charging option at {loc} must be a list of two elements: [power_kW, share].")
                power, share = option
                if not (isinstance(power, (int, float)) and power > 0):
                    raise ValueError(f"Invalid charging power at {loc}: {power}. Must be a positive number.")
                if not (isinstance(share, (int, float)) and 0 <= share <= 1):
                    raise ValueError(f"Invalid share at {loc}: {share}. Must be between 0 and 1.")
        self._charging_powers_kw = value

    # --- depot_hours ---
    @property
    def depot_hours(self):
        return self._depot_hours

    @depot_hours.setter
    def depot_hours(self, value: dict):
        self._depot_hours = value

    # --- charging_efficiency ---
    @property
    def charging_efficiency(self):
        return self._charging_efficiency

    @charging_efficiency.setter
    def charging_efficiency(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError("Charging efficiency must be between 0 and 1.")
        self._charging_efficiency = value

    # --- charging_strategy ---
    @property
    def charging_strategy(self):
        return self._charging_strategy

    @charging_strategy.setter
    def charging_strategy(self, value: str):
        self._charging_strategy = value