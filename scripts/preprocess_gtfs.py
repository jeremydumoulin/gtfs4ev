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

	else:
		print(f"INFO \t Preprocessing: The city named {city} is not associated with any GTFS preprocessing rules")

	return feed