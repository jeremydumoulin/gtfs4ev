# coding: utf-8

""" 
Some useful generic functions for gtfs4ev classes.
"""

import pandas as pd
from shapely.geometry import LineString, Point, shape, Polygon, box
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from pyproj import Geod

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

def find_closest_point(line, point):
	""" Among all the points of a line (LineString object), returns the one closest to the point (Point object) coordinates  
	"""

	min_distance = float('inf')
	closest_point = None

	for coordinate in line.coords:
		current_point = Point(coordinate)
		distance = point.distance(current_point)

		if distance < min_distance:
			min_distance = distance
			closest_point = current_point

	return closest_point


def local_emission_index(vkm_list, linestring_list, ref_raster_path, output_raster_path, C = 1):
    """ Map of local emission factor, prop. to vkm of all trips within each pixel  
    """
    # Open the reference raster
    with rasterio.open(ref_raster_path) as src:
        # Read the raster as a numpy array
        raster_array = src.read(1).astype(float)

        # Get the affine transformation matrix
        transform = src.transform

        half_pixel_width = transform[0] / 2
        half_pixel_height = transform[4] / 2

        # Create an empty emission index map
        emission_index = np.zeros_like(raster_array, dtype=np.uint8)

        # Iterate over each pixel
        i = 1
        vkm_tot = .0 # Total vkm - just to check the result in the end
        for y in range(src.height):
            for x in range(src.width):
                # Get the coordinates of the pixel
                pixel_coords = rasterio.transform.xy(transform, y, x)

                print(f"Processing pixel {i} out of {src.height*src.width}", end='\r')
                i = i+1
                
				# Calculate the coordinates of the square centered on the pixel                
                square_coords = [
                    (pixel_coords[0] - half_pixel_width, pixel_coords[1] - half_pixel_height),
                    (pixel_coords[0] + half_pixel_width, pixel_coords[1] - half_pixel_height),
                    (pixel_coords[0] + half_pixel_width, pixel_coords[1] + half_pixel_height),
                    (pixel_coords[0] - half_pixel_width, pixel_coords[1] + half_pixel_height),
                    (pixel_coords[0] - half_pixel_width, pixel_coords[1] - half_pixel_height)  # Close the polygon
                ]
                pixel_square = box(square_coords[0][0], square_coords[0][1], square_coords[2][0], square_coords[2][1])

                j = 0
                # Iterate over all linestrings and calculate the VKM within the current pixel
                for linestring in linestring_list:             	

                	if linestring.intersects(pixel_square):                		
                		length = length_km(linestring)                		

                		intersection = linestring.intersection(pixel_square)
                		intersection_length = length_km(intersection)

                		vkm = vkm_list[j] # VKM of the trip associated with the linestring

                		vkm_pixel = vkm * (intersection_length/length) # VKM weigthed by the intersection length
                		vkm_tot += vkm_pixel                  		      		

	                	emission_index[y, x] += vkm_pixel # Add contribution to the emission index (Muliplication by C is done at the end)

	                j = j + 1 	

        # Write the mask to a new raster
        with rasterio.open(
            output_raster_path,
            'w',
            driver='GTiff',
            height=emission_index.shape[0],
            width=emission_index.shape[1],
            count=1,
            dtype=np.uint8,
            crs=src.crs,
            transform=transform
        ) as dst:
            dst.write(emission_index * C, 1)

    print(f"Rasterization completed successfully. Total vkm: {vkm_tot} km")

def crop_raster(raster_path, bbox, output_raster_path):
    """ Creates a new raster cropped to the bbox
    """
    data_path = raster_path

    with rasterio.open(data_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, [bbox], crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image) 

def length_km(linestring, geodesic = True):
    """ Calculates the lenght in km of a linestring
    """
    if geodesic: 
    	geod = Geod(ellps="WGS84")
    	distance = geod.geometry_length(linestring) / 1000.0
    else:
    	web_mercator_projection = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    	linestring_projection = transform(web_mercator_projection, linestring)
    	distance = linestring_projection.length / 1000.0

    return distance


def exponential_decay_kernel(size, decay_factor):
    center = (size - 1) / 2  # Center of the kernel
    x = np.arange(size) - center
    kernel_2d = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel_2d[i, j] = np.exp(-decay_factor * np.abs(distance))
    #kernel_2d /= np.sum(kernel_2d)  # Normalize the kernel

    return kernel_2d

# Define a mask for the 5-pixel radius
def mask_within_radius(size, radius):
    mask = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 <= radius**2:
                mask[i, j] = 1
    return mask
