# coding: utf-8

""" 
City-specific preprocessing rules for different african cities
"""

def gtfs_preprocessing(feed, city):
	""" Returns a filtered GTFSFeed object 
	"""

	if city == "Abidjan":
		# Abidjan: Keep only gbaka minibus taxis
		feed.filter_agency('monbus', clean_all = True) 
		feed.filter_agency('monbus / Navette', clean_all = True) 
		feed.filter_agency('Express', clean_all = True) 
		feed.filter_agency('Wibus', clean_all = True) 
		feed.filter_agency('STL', clean_all = True) 
		feed.filter_agency('monbato', clean_all = True) 
		feed.filter_agency('Woro-woro de Cocody', clean_all = True) 
		feed.filter_agency('Woro-woro de Treichville', clean_all = True) 
		feed.filter_agency('Woro-woro de Yopougon', clean_all = True) 
		feed.filter_agency('Woro-woro d\'Adjamé', clean_all = True) 
		feed.filter_agency('Aqualines', clean_all = True) 
		feed.filter_agency('Woro-woro de Port-Bouët', clean_all = True) 
		feed.filter_agency('Woro-woro d\'Abobo', clean_all = True) 
		feed.filter_agency('Woro-woro de Koumassi', clean_all = True) 
		feed.filter_agency('Woro-woro de Marcory', clean_all = True) 
		feed.filter_agency('Woro-woro d\'Attecoubé', clean_all = True) 
		feed.filter_agency('Woro-woro de Bingerville', clean_all = True)

	elif city == "Alexandria":
		# Alexandria: drop buses
		feed.filter_agency('Bus', clean_all = True)

	elif city == "Cairo":
		# Cairo: drop the public formal transport
		feed.filter_agency('CTA', clean_all = True)
		feed.filter_agency('CTA_M', clean_all = True)

		# Delete the repeated trips (i.e. those who correspond to the same taxi but a different timespan)
		df = feed.trips
		
		df['common_trip_id'] = df['trip_id'].str.extract(r'(.+?)\s*\(') # Extract common part of trip_id		
		unique_df = df.drop_duplicates(subset=['route_id', 'service_id', 'common_trip_id', 'shape_id']) # Keep only unique rows
		unique_df = unique_df.drop(columns=['common_trip_id']) # Drop the extra column		
		unique_df.reset_index(drop=True, inplace=True) # Reset the index

		feed.trips = unique_df

		# Correct to frequencies so that they match with the trips
		df = feed.frequencies 
		
		df['common_trip_id'] = df['trip_id'].str.extract(r'(.+?)\s*\(') # Extract common part of trip_id		
		df['trip_id'] = df['common_trip_id'] + ' (15-00-00)' # Replace the time part with '(19-00-00)'		
		df = df.drop(columns=['common_trip_id']) # Drop the extra column

		# Sort the DataFrame by trip_id and start_time
		df = df.sort_values(by=['trip_id', 'start_time'])

		feed.frequencies = df	

	elif city == "Freetown":
		# Freetown: keep only weekdays and poda-podas
		feed.filter_services('service_0001', clean_all = True) 
		feed.filter_services('service_0003', clean_all = True)
		feed.filter_agency('Freetown_SLRTC_03', clean_all = True)
		feed.filter_agency('Freetown_Tagrin_Ferry_01', clean_all = True)
		feed.filter_agency('Freetown_Taxi_Cab_04', clean_all = True)

	elif city == "Harare":
		# Harare: drop the weekends
		feed.filter_services('service_0001', clean_all = True)

	elif city == "Kampala":
		# Kampala: drop buses
		feed.filter_agency('bus', clean_all = True)

		# Delete the repeated trips (i.e. those who correspond to the same taxi but a different timespan)
		df = feed.trips
		
		df['common_trip_id'] = df['trip_id'].str.extract(r'(.+?)\s*\(') # Extract common part of trip_id		
		unique_df = df.drop_duplicates(subset=['route_id', 'service_id', 'common_trip_id', 'shape_id']) # Keep only unique rows
		unique_df = unique_df.drop(columns=['common_trip_id']) # Drop the extra column		
		unique_df.reset_index(drop=True, inplace=True) # Reset the index

		feed.trips = unique_df

		# Correct to frequencies so that they match with the trips
		df = feed.frequencies 
		
		df['common_trip_id'] = df['trip_id'].str.extract(r'(.+?)\s*\(') # Extract common part of trip_id		
		df['trip_id'] = df['common_trip_id'] + ' (19-00-00)' # Replace the time part with '(19-00-00)'		
		df = df.drop(columns=['common_trip_id']) # Drop the extra column

		# Sort the DataFrame by trip_id and start_time
		df = df.sort_values(by=['trip_id', 'start_time'])

		feed.frequencies = df		

	else:
		print(f"INFO \t Preprocessing: The city named {city} is not associated with any GTFS preprocessing rules")

	return feed