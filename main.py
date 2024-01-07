# coding: utf-8

from gtfs4ev.vehicle import Vehicle
from gtfs4ev.trafficfeed import TrafficFeed

"""
Main function
"""
def main():

	feed = TrafficFeed("GTFS_Nairobi")

	df = feed.stops
	
	

if __name__ == "__main__":
	main()