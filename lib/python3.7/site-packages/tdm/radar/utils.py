# Copyright 2018-2019 CRS4
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime, timedelta
from itertools import product
import itertools as it
import os
import numpy as np

import gdal
from gdal import osr
import imageio

gdal.UseExceptions()

strptime = datetime.strptime
splitext = os.path.splitext

FMT = "%Y-%m-%d_%H:%M:%S"
FMT_LEN = 4 + 5 * 3  # %Y is 4 chars, other fields are 2 chars
MIN_DT, MAX_DT = datetime.min, datetime.max

# The radar is supposed to generate an image every minute. In practice, while
# the peak is at 60s, deltas can go below 50 and above 70 (data from >120K
# images). 200s seems a good threshold for separating between events (i.e., a
# delta larger than that means the radar has been turned off and on again).
EVENT_THRESHOLD = 200

# Ignore events that last less than this, in seconds
MIN_EVENT_LEN = 24 * 60 * 60

RAINFALL_FILL_VALUE = -1.0

# a and b empirical parameters for Z = a * R ^ b (reflectivity vs rain
# intensity). These values are the recommended ones for cag01est2400
A_PARAM, B_PARAM = 100, 1.5


class GeoAdapter(object):

    def __init__(self, template):
        self.raster = gdal.Open(template)
        oX, pxlW, _1, oY, _2, pxlH = self.raster.GetGeoTransform()
        # we only deal with rectangular, axis-aligned images
        if _1 or _2:
            raise RuntimeError("%s: unsupported transform")
        self.wkt = self.raster.GetProjectionRef()
        self.sr = osr.SpatialReference(wkt=self.wkt)
        factor = self.sr.GetLinearUnits()  # mult factor to get meters
        self.cols, self.rows = self.raster.RasterXSize, self.raster.RasterYSize
        self.oX, self.oY = oX, oY
        self.pxlW, self.pxlH = factor * pxlW, factor * pxlH

    def save_as_gtiff(self, fname, data, metadata=None):
        raster = gdal.GetDriverByName('GTiff').Create(
            fname, self.cols, self.rows, 1, gdal.GDT_Float32
        )
        band = raster.GetRasterBand(1)
        if isinstance(data, np.ma.MaskedArray):
            band.WriteArray(data.filled())
            band.SetNoDataValue(data.fill_value)
        else:
            band.WriteArray(data)
        band.FlushCache()
        raster.SetGeoTransform((self.oX, self.pxlW, 0, self.oY, 0, self.pxlH))
        raster.SetProjection(self.wkt)
        if isinstance(metadata, dict):
            raster.SetMetadata(metadata)

    def compute_distance_field(self):
        x = self.pxlW * (np.arange(-(self.cols/2), (self.cols/2), 1) + 0.5)
        y = self.pxlH * (np.arange(-(self.rows/2), (self.rows/2), 1) + 0.5)
        xx, yy = np.meshgrid(x, y, sparse=True)
        return 10 * np.log(xx**2 + yy**2)

    def xpos(self):
        return self.oX + self.pxlW * np.arange(self.cols)

    def ypos(self):
        return self.oY + self.pxlH * np.arange(self.rows)


def get_images(root, after=MIN_DT, before=MAX_DT):
    """\
    Get the file names of raw PNG radar images. The pattern seen so far is:

      <RADAR_TAG><TIMESTAMP>.png

    Assumes that the basename, after removing the '.png' extension, ends with
    the formatted datetime, allowing for any combination of characters before
    that. Returns (datetime, path) pairs, ignoring names that do not match.
    """
    ls = []
    for entry in os.scandir(root):
        if entry.is_dir():
            continue
        dt_string = splitext(entry.name)[0][-FMT_LEN:]
        try:
            dt = strptime(dt_string, FMT)
        except ValueError:
            continue
        if dt < after or dt > before:
            continue
        ls.append((dt, entry.path))
    ls.sort()
    return ls


def group_images(dt_path_pairs, delta, after=MIN_DT):
    assert isinstance(delta, timedelta)

    def grouper(p):
        return after + delta * ((p[0] - after) // delta)

    return it.groupby(dt_path_pairs, grouper)


def get_grouped_images(root, delta, after=MIN_DT, before=MAX_DT):
    pairs = get_images(root, after=after, before=before)
    return group_images(pairs, delta, after=after)


def get_image_data(path):
    im = imageio.imread(path)
    signal = im[:, :, 0].view(np.ma.MaskedArray)
    signal.mask = im[:, :, 3] != 255
    return signal


def estimate_rainfall(masked_signal, a=A_PARAM, b=B_PARAM):
    Z = 10**(0.1*(0.39216 * masked_signal - 8.6))
    rf = (Z/a)**(1/b)
    rf.set_fill_value(RAINFALL_FILL_VALUE)
    return rf


def band_to_ma(band):
    """
    Read a GDAL band as a numpy.ma.MaskedArray.

    https://trac.osgeo.org/gdal/wiki/rfc15_nodatabitmask
    """
    flags = band.GetMaskFlags()
    if (flags & gdal.GMF_ALPHA):
        raise RuntimeError("mask is actually an alpha channel")
    m_band = band.GetMaskBand()
    kwargs = {"mask": m_band.ReadAsArray() == 0}
    if (flags & gdal.GMF_NODATA):
        kwargs["fill_value"] = band.GetNoDataValue()
    return np.ma.masked_array(band.ReadAsArray(), **kwargs)


def events(dt_path_pairs, min_len=MIN_EVENT_LEN, threshold=EVENT_THRESHOLD):
    if not isinstance(min_len, timedelta):
        min_len = timedelta(seconds=min_len)
    p, N = dt_path_pairs, len(dt_path_pairs)
    if N == 0:
        return
    deltas = np.array([(p[i+1][0] - p[i][0]).total_seconds()
                       for i in range(N - 1)])
    big_delta_idx = np.argwhere(deltas > threshold)[:, 0]
    dts_idx = np.insert(1 + big_delta_idx, 0, 0)
    dts_idx = np.append(dts_idx, N)
    for i in range(dts_idx.size - 1):
        b, e = dts_idx[i], dts_idx[i+1]
        if p[e - 1][0] - p[b][0] < min_len:
            continue
        yield p[b: e]


def get_lat_lon(source_sr, xpos, ypos):
    """\
    Convert (x, y) points from source_sr to EPSG 4326.

    Return lat and lon arrays corresponding to the input x and y positions
    vectors, so that lat[i, j] and lon[i, j] are, respectively, the latitude
    and longitude values for the (xpos[j], ypos[i]) point in the original ref.
    """
    target_sr = osr.SpatialReference()
    target_sr.ImportFromEPSG(4326)
    transform = osr.CoordinateTransformation(source_sr, target_sr)
    lon, lat, _ = zip(*transform.TransformPoints(list(product(xpos, ypos))))
    lon = np.array(lon).reshape(len(xpos), len(ypos)).T
    lat = np.array(lat).reshape(len(xpos), len(ypos)).T
    return lat, lon
