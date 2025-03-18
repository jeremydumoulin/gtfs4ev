# coding: utf-8

class VehicleFleet:
    """A class to represent a fleet of electric vehicles (EVs) with specified types and their shares."""

    def __init__(self, vehicle_types: list):
        """
        Initializes the VehicleFleet class.

        Args:
            vehicle_types (list): A list of pairs [Vehicle, share], where each share is a value between 0 and 1.            
        """
        print("=========================================")
        print(f"INFO \t Creation of a VehicleFleet object.")
        print("=========================================")

        self.vehicle_types = vehicle_types

        print(f"INFO \t Successful initialization of input parameters.")

    # Properties and Setters
    @property
    def vehicle_types(self) -> list:
        """list: The list of vehicle types and their shares in the fleet."""
        return self._vehicle_types

    @vehicle_types.setter
    def vehicle_types(self, value: list):
        total_share = sum(share for _, share in value)
        if not all(0 <= share <= 1 for _, share in value):
            raise ValueError("Each vehicle share must be a positive value between 0 and 1.")
        if not total_share == 1:
            raise ValueError("The sum of vehicle shares must be equal to 1.")
        self._vehicle_types = value

    # Fleet metrics
    def average_battery_capacity(self) -> float:
        """Calculates the average battery capacity based on vehicle shares.

        Returns:
            float: The average battery capacity in kWh.
        """
        return sum(vehicle.battery_capacity_kWh * share for vehicle, share in self.vehicle_types)

    def average_consumption(self) -> float:
        """Calculates the average consumption based on vehicle shares.

        Returns:
            float: The average consumption in kWh/km.
        """
        return sum(vehicle.consumption_kWh_per_km * share for vehicle, share in self.vehicle_types)