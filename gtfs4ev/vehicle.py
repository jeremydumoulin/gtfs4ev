# coding: utf-8

import numpy as np

from gtfs4ev import constants as cst
from gtfs4ev import environment as env

class Vehicle:  

    """
    ATTRIBUTES
    """
    
    consumption = 0.2 # Electric vehicle consumption (kWh/km)
