import gdal
import requests
import matplotlib.pyplot as plt
import sys
import numpy as np
import urllib.request
import math

import gis


WGS84_EPSG = 4326


class NDVITimeseries:
    def __init__(self, longitude: float, latitude: float, min_cloud: int=0, max_cloud: int=100):
        self.longitude = longitude
        self.latitude = latitude
        self.min_cloud = min_cloud
        self.max_cloud = max_cloud


    def get_ndvi(self, scene_id: str) -> float:
        path, row = scene_id[3:6], scene_id[6:9]

        bands = ["5", "4"] #for ndvi
	#bands=["6","5"] for ndmi
	#bands=["7","5"] for nbr
	#bands=["7","6"] for nbr2

        meta_data = self._get_aws_meta(scene_id, path, row)

        data = []
        for band in bands:
            
            band=int(band)
            image_dataset = gdal.Open(path)#path to the image
            prorjection = image_dataset.GetProjection()
            epsg_out = prorjection.split('"EPSG",')[-1].strip(']').strip('"')

            lng2, lat2 = gis.convert_coords(self.longitude, self.latitude, WGS84_EPSG, epsg_out)
            x, y = gis.world_to_pixel(image_dataset, lng2, lat2)

            digital_number = image_dataset.ReadAsArray(y, x, 1, 1).astype('float32')

            reflectance = self._radiance2reflectance(digital_number, band, meta_data)

            data.append(reflectance)

        ndvi = (data[0] - data[1]) / (data[0] + data[1])

        return ndvi

    @staticmethod
    def _get_aws_meta(scene_id: str, path: int, row: int) -> str:
        meta_data=df.read(path)#path to the metadata file obtained in the image product

        return meta_data

    

    def _radiance2reflectance(self, digital_number: float, band: int, meta_data: str) -> float:
        """ Conversion Top Of Atmosphere planetary reflectance
        REF: http://landsat.usgs.gov/Landsat8_Using_Product.php
        Following function based on work by Vincent Sarago:
        https://github.com/vincentsarago/landsatgif/blob/master/landsat_gif.py
        :param digital_number: digital number value
        :param band:
        :param meta_data:
        :return:
        """

        mp = float(self._landsat_extract_mtl(meta_data, "REFLECTANCE_MULT_BAND_{}".format(band)))
        ap = float(self._landsat_extract_mtl(meta_data, "REFLECTANCE_ADD_BAND_{}".format(band)))
        se = math.radians(float(self._landsat_extract_mtl(meta_data, "SUN_ELEVATION")))

        reflect_toa = (np.where(digital_number > 0, (mp * digital_number + ap) / math.sin(se), 0))

        return reflect_toa

    @staticmethod
    def _landsat_extract_mtl(meta_data: list, param: str) -> str:
        """ Extract Parameters from MTL file """
        for line in meta_data:
            data = line.split(' = ')
            if (data[0]).strip() == param:
                return (data[1]).strip()



d=NDVITimeseries(lattitude, longitude)
d.get_ndvi(scene_id);






ndvi_values = [data[result]['ndvi'][0][0] for result in data]
date_values = [data[result]['date'] for result in data]

plt.figure(figsize=(15, 5))
plt.plot(ndvi_values[::-1])
plt.xticks(range(len(date_values)), date_values[::-1])
plt.suptitle('lng: -4.557, lat: 50.349')
plt.xlabel('Time')
plt.ylabel('NDVI')
plt.show()

