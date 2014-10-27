import numpy as np
# from memory_profiler import profile
from pandas import DataFrame, Series

class BasicSatelliteInterpolator(object):
    """
    Handles interpolation of geolocation data from a grid of tie points.  

    Currently, it is assumed, that tie-points reach out til the edges if the
    tiepoint grid. Methods for extrapolation need to be added if needed.

    Uses numpy and pandas.

    The constructor gets the tiepointed lat/lon data as *lat_data* and 
    *lon_data*.
    """

    def __init__(self, cols, rows, lat_data, lon_data):
        self.row_indices = rows
        self.col_indices = cols

        self.lon_tiepoint = lon_data
        self.lat_tiepoint = lat_data

        self.longitudes = None
        self.latitudes = None


    def _interp(self, data):
        """The interpolation method implemented here is a kind of a billinear
        interpolation. The input *data* field is first interpolated along the 
        rows and subsequently along its columns.

        The final size of the interpolated *data* field is determined by the 
        last indices in self.row_indices and self.col_indices.
        """
        row_interpol_data = self._interp_axis(data, 0)
        interpol_data = self._interp_axis(row_interpol_data, 1)

        return interpol_data


    # @profile
    def _interp_axis(self, data, axis):
        """The *data* field contains the data to be interpolated. It is
        expected that values reach out to the *data* boundaries.
        With *axis*=0 this method interpolates along rows and *axis*=1 it
        interpolates along colums.

        For column mode the *data* input is transposed before interpolation
        and subsequently transposed back.
        """
        if axis == 0:
            return self._pandas_interp(data, self.row_indices)
        
        if axis == 1:
            data_transposed = data.as_matrix().T
            data_interpol_transposed = self._pandas_interp(data_transposed, 
                                                            self.col_indices)
            data_interpol = data_interpol_transposed.as_matrix().T

            return data_interpol


    def _pandas_interp(self, data, indices):
        """The actual transformation based on the following stackoverflow 
        entry: http://stackoverflow.com/a/10465162
        """
        new_index = np.arange(indices[-1] + 1)

        data_frame = DataFrame(data, index=indices)
        data_frame_reindexed = data_frame.reindex(new_index)
        data_interpol = data_frame_reindexed.apply(Series.interpolate)

        del new_index
        del data_frame
        del data_frame_reindexed

        return data_interpol


    def interpolate(self):
        """Do the interpolation and return resulting longitudes and latitudes.
        """
        self.latitude = self._interp(self.lat_tiepoint)
        self.longitude = self._interp(self.lon_tiepoint)

        return self.latitude, self.longitude