"""
Unit tests for python-geotiepoints: MODIS examples
"""

import unittest
import numpy as np

from geotiepoints import (modis5kmto1km, modis1kmto250m, 
                          faster_modis1kmto250m)

from geotiepoints import get_scene_splits


class TestUtils(unittest.TestCase):
    """Class for unit testing the ancillary interpolation functions
    """

    def setUp(self):
        pass


    def test_get_numof_subscene_lines(self):
        """Test getting the number of sub-scene lines dependent on the number
        of CPUs and for various number of lines in a scan"""

        ncpus = 3 
        scene_splits = get_scene_splits(1060,
                                        10, ncpus)
        
    

        
class TestMODIS(unittest.TestCase):
    """Class for system testing the MODIS interpolation.
    """

    def setUp(self):
        pass


    def test_5_to_1(self):
        """test the 5km to 1km interpolation facility
        """
        #gfilename = "testdata/MOD03_A12097_174256_2012097175435.hdf"
        gfilename = "/san1/test/data/modis/MOD03_A12097_174256_2012097175435.hdf"
        #filename = "testdata/MOD021km_A12097_174256_2012097175435.hdf"
        filename = "/san1/test/data/modis/MOD021km_A12097_174256_2012097175435.hdf"
        from pyhdf.SD import SD
        from pyhdf.error import HDF4Error

        try:
            gdata = SD(gfilename)
            data = SD(filename)
        except HDF4Error:
            print "Failed reading both eos-hdf files %s and %s" % (gfilename, filename)
            return
        
        glats = gdata.select("Latitude")[:]
        glons = gdata.select("Longitude")[:]
    
        lats = data.select("Latitude")[:]
        lons = data.select("Longitude")[:]
        
        tlons, tlats = modis5kmto1km(lons, lats)

        self.assert_(np.allclose(tlons, glons, atol=0.05))
        self.assert_(np.allclose(tlats, glats, atol=0.05))


    def test_1000m_to_250m(self):
        """test the 1 km to 250 meter interpolation facility
        """
        gfilename_hdf = "testdata/MOD03_A12278_113638_2012278145123.hdf"
        gfilename = "testdata/250m_lonlat_section_input.npz"
        result_filename = "testdata/250m_lonlat_section_result.npz"

        from pyhdf.SD import SD
        from pyhdf.error import HDF4Error
        
        gdata = None
        try:
            gdata = SD(gfilename_hdf)
        except HDF4Error:
            print "Failed reading eos-hdf file %s" % gfilename_hdf
            try:
                indata = np.load(gfilename)
            except IOError:
                return

        if gdata:
            lats = gdata.select("Latitude")[20:50, :]
            lons = gdata.select("Longitude")[20:50, :]
        else:
            lats = indata['lat'] / 1000.
            lons = indata['lon'] / 1000.

        verif = np.load(result_filename)
        vlons = verif['lon'] / 1000.
        vlats = verif['lat'] / 1000.
        tlons, tlats = modis1kmto250m(lons, lats, parallel=False)
        self.assert_(np.allclose(tlons, vlons, atol=0.05))
        self.assert_(np.allclose(tlats, vlats, atol=0.05))

        tlons, tlats = modis1kmto250m(lons, lats)

        self.assert_(np.allclose(tlons, vlons, atol=0.05))
        self.assert_(np.allclose(tlats, vlats, atol=0.05))


if __name__ == "__main__":
    unittest.main()
