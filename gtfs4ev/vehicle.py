# coding: utf-8

class Vehicle:
    """A class to represent an Electric Vehicle (EV) Bus with configurable attributes."""

    def __init__(self, name: str, battery_capacity_kWh: float, consumption_kWh_per_km: float, max_charging_power_kW: float = float("inf")):
        """
        Initializes the Vehicle class.

        Args:
            name (str): The name of the vehicle.
            battery_capacity_kWh (float): Battery capacity in kWh.
            consumption_kWh_per_km (float): Energy consumption in kWh/km.
            max_charging_power_kW (float, optional): [Default: float("inf")] Maximum charging power in kW.            
        """
        self.name = name
        self.battery_capacity_kWh = battery_capacity_kWh
        self.max_charging_power_kW = max_charging_power_kW
        self.consumption_kWh_per_km = consumption_kWh_per_km

    # Properties and Setters
    @property
    def name(self) -> str:
        """str: The name of the vehicle."""
        return self._name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str) or not value:
            raise ValueError("Name must be a non-empty string.")
        self._name = value

    @property
    def battery_capacity_kWh(self) -> float:
        """float: The battery capacity of the vehicle in kWh."""
        return self._battery_capacity_kWh

    @battery_capacity_kWh.setter
    def battery_capacity_kWh(self, value: float):
        if value <= 0:
            raise ValueError("Battery capacity must be a positive value.")
        self._battery_capacity_kWh = value

    @property
    def max_charging_power_kW(self) -> float:
        """float: The maximum charging power of the vehicle in kW."""
        return self._max_charging_power_kW

    @max_charging_power_kW.setter
    def max_charging_power_kW(self, value: float):
        if value is not None and value <= 0:
            raise ValueError("Max charging power must be a positive value.")
        self._max_charging_power_kW = value

    @property
    def consumption_kWh_per_km(self) -> float:
        """float: The energy consumption of the vehicle per kilometer in kWh/km."""
        return self._consumption_kWh_per_km

    @consumption_kWh_per_km.setter
    def consumption_kWh_per_km(self, value: float):
        if value <= 0:
            raise ValueError("Consumption per km must be a positive value.")
        self._consumption_kWh_per_km = value