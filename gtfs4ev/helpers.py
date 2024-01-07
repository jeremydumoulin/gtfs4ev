# coding: utf-8

import pandas as pd

""" 
Helper functions
"""

def check_dataframe(df):
	""" Returns false if a dataframe contains neither NaN nor empty values.	  
    """

	# Check for NaN values in the entire DataFrame
	if df.isna().any().any():
		return False
	# Check for empty cells in the entire DataFrame
	elif df.apply(lambda x: x == '').any().any():
		return False
	else:
		return True