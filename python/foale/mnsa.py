import os
import fitsio


class MNSA(object):
    """MNSA object to read cube and maps

    Parameters
    ----------

    version : str
        version of results (default 'dr17-v0.1')

    plateifu : str
        Plate-IFU of result

    dr17 : bool
        Download DR17 version of the cube and maps

    Notes
    -----

    Reads data from $MNSA_DATA/{version} unless dr17=True.

    Example to read in cube and maps:

      m = mnsa.MNSA(plateifu='10001-12701')
      m.read_cube()
      m.read_maps()
"""
    def __init__(self, version='dr17-v0.1', plateifu=None, dr17=False):

        self.daptype = 'HYB10-MILESHC-MASTARHC2'

        plate, ifu = [int(x) for x in plateifu.split('-')]

        self.plate = plate
        self.ifu = ifu
        self.plateifu = plateifu
        self.version = version
        self.dr17 = dr17

        manga_dir = os.path.join(os.getenv('MNSA_DATA'),
                                 self.version, 'manga', 'redux',
                                 self.version,
                                 str(self.plate), 'stack')

        if(self.dr17):
            manga_dir = os.path.join('/uufs', 'chpc.utah.edu',
                                     'common', 'home', 'sdss50',
                                     'dr17', 'manga', 'spectro',
                                     'redux', 'v3_1_1', str(self.plate),
                                     'stack')

        manga_file = os.path.join(manga_dir,
                                  'manga-{plateifu}-LOGCUBE.fits.gz')
        self.manga_file = manga_file.format(plateifu=self.plateifu)

        self.manga_irg_png = self.manga_file.replace('.fits.gz',
                                                     '.irg.png')
        return

    def read_cube(self):
        """Read cube and store as attribute 'cube'"""
        self.cube = fitsio.FITS(self.manga_file)
        return

    def read_maps(self):
        """Read maps and store as attribute 'maps'"""

        dap_dir = os.path.join(os.getenv('MNSA_DATA'),
                               self.version, 'manga', 'analysis',
                               self.version, self.version, self.daptype,
                               str(self.plate), str(self.ifu))
        if(self.dr17):
            dap_dir = os.path.join('/uufs', 'chpc.utah.edu',
                                   'common', 'home', 'sdss50',
                                   'dr17', 'manga', 'spectro',
                                   'analysis', 'v3_1_1', '3.1.0',
                                   self.daptype,
                                   str(self.plate), str(self.ifu))
        dap_file = os.path.join(dap_dir, 'manga-{p}-{t}-{d}.fits.gz')
        dap_maps = dap_file.format(p=self.plateifu, d=self.daptype,
                                   t='MAPS')
        self.maps = fitsio.FITS(dap_maps)
        return
