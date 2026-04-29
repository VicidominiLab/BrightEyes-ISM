from fileinput import filename

import h5py
import re
import warnings

class metadata:
    """
    Object containing the metadata read from a mcs data file.

    Attributes
    ----------
    version : str
        Version of the MCS software that generated the h5 file.
    comment : str
        User comment 
    rangex : float
        Field of view along the x-axis (um).
    rangey : float
        Field of view along the y-axis (um).
    rangez : float
        Field of view along the z-axis (um).
    nbin : int
        Number of time bins per pixel.
    nbin : int
        Number of time bins per pixel.
    dt : float
        Duration of each time bin (us)
    nx : int
        Number of pixel along the x-axis.
    ny : int
        Number of pixel along the y-axis.
    nz : int
        Number of pixel along the z-axis.
    nrep : int
        Number of repetitions.
    calib_x : float
        calibration value of the x pixel size (um/V).
    calib_y : float
        calibration value of the y pixel size (um/V).
    calib_z : float
        calibration value of the y pixel size (um/V).
    bitfile : str
        Bitfile name or path stored in the acquisition metadata.
    dfd_freq : int | None
        DFD repetition frequency parsed from the bitfile name, in MHz when available.
    dfd_nbins : int | None
        DFD histogram bin count parsed from the bitfile name, when available.
    dfd_activate : bool
        Whether DFD mode was active according to the FPGA configuration.
        
    Methods
    -------
    pxdwelltime : float
        Returns the pixel dwell time.
    frametime : float
        Returns the duration of each fram (xy) acquisition.
    dx : float
        Returns the x pixel size (um).
    dy : float
        Returns the y pixel size (um).
    dz : float
        Returns the z pixel size (um).
    pxsizes : list
        Returns the list of pixel sizes in the z, y, x dimensions (um).
    nmicroim : int
        <returns total number of microimages read during the measurement.
    ndatapoints : int
        Returns the total number of words transferred from low level
        to high level (2 words per microimage).
    duration : float
        Returns thetotal measurement duration (s).
    Print():
        Prints all the metadata on screen (name and value).
    """

    # __slots__ = ['version', 'comment', 'rangex', 'rangey', 'rangez', 'nbin', 'dt', 'nx', 'ny', 'nz', 'nrep', 'calib_x', 'calib_y', 'calib_z']

    def __init__(self, f):

        # MCS version
        self.version = f.attrs['data_format_version']
        try:
            self.comment = f.attrs['comment']
        except:
            self.comment = ''

        # range in um
        self.rangex = f['configurationGUI'].attrs['range_x']
        self.rangey = f['configurationGUI'].attrs['range_y']
        self.rangez = f['configurationGUI'].attrs['range_z']

        # number of time bins per pixel
        self.nbin = f['configurationGUI'].attrs['timebin_per_pixel']

        # time resolution in us
        self.dt = f['configurationGUI'].attrs['time_resolution']

        # number of pixels in x, y, z direction
        self.nx = f['configurationGUI'].attrs['nx']
        self.ny = f['configurationGUI'].attrs['ny']
        self.nz = f['configurationGUI'].attrs['nframe']

        # number of repetitions
        self.nrep = f['configurationGUI'].attrs['nrep']

        # calibration values
        self.calib_x = f['configurationGUI'].attrs['calib_x']
        self.calib_y = f['configurationGUI'].attrs['calib_y']
        self.calib_z = f['configurationGUI'].attrs['calib_z']
        
        try:
            self.bitfile = f["configurationGUI"].attrs["bitFile"]
        except Exception:
            self.bitfile = ""

        self._dfd_metadata_loaded = False
        self._dfd_freq = None
        self._dfd_nbins = None

        try:
            self.dfd_activate = f["configurationFPGA"].attrs["DFD_Activate"]
        except Exception:
            self.dfd_activate = False

    @property
    def pxdwelltime(self):
        # pixel dwell time in us
        return self.dt * self.nbin

    @property
    def frametime(self):
        # frame time in s
        return self.pxdwelltime * self.nx * self.ny / 1e6

    @property
    def framerate(self):
        # frame rate in Hz
        return 1 / self.frametime

    @property
    def dx(self):
        # pixel size in x direction
        return self.rangex / self.nx

    @property
    def dy(self):
        # pixel size in y direction
        return self.rangey / self.ny

    @property
    def dz(self):
        # pixel size in z direction
        return self.rangez / self.nz

    @property
    def pxszizes(self):
        # List of pixel sizes in z, y, x directions
        return [self.dz, self.dy, self.dx]

    @property
    def nmicroim(self):
        # total number of microimages read during the measurement
        return self.nx * self.ny * self.nz * self.nrep * self.nbin

    @property
    def ndatapoints(self):
        # total number of words transferred from low level to high level
        # 2 words per microimage
        return 2 * self.nmicroim

    @property
    def duration(self):
        # total measurement duration in s
        return self.nmicroim * self.dt * 1e-6

    def _load_dfd_metadata_from_bitfile_name(self):
        if self._dfd_metadata_loaded:
            return

        self._dfd_metadata_loaded = True
        self._dfd_freq, self._dfd_nbins = self.parse_dfd_metadata_from_bitfile_name(self.bitfile)

    @property
    def dfd_freq(self):
        self._load_dfd_metadata_from_bitfile_name()
        return self._dfd_freq

    @property
    def dfd_nbins(self):
        self._load_dfd_metadata_from_bitfile_name()
        return self._dfd_nbins
    
    @staticmethod
    def parse_dfd_metadata_from_bitfile_name(bitfile="", default_cycle_mhz=40):
        """
        Infer DFD metadata from a bitfile name token like ``40M91``.

        Accept any ``xxxxxMyyyyyyy`` token found in the basename, provided:
        - 3 < xxxxx < 100
        - 3 < yyyyyyy < 1000
        """

        filename = str(bitfile).replace("\\", "/").split("/")[-1]
        match = re.search(r"(?P<cycle>\d+)M(?P<bins>\d+)", filename, re.IGNORECASE)
        if not match:
            warnings.warn(
                (
                    "\n"
                    "================ WARNING ==============\n"
                    "brighteyes_ism.dataio.mcs.load() failed to extract DFD metadata "
                    f"from the bitfile name ({filename!r}).\n\n"
                    "Falling back to defaults:\n"
                    f"  - Laser cycle frequency: {default_cycle_mhz} MHz\n"
                    "  - DFD bin count: NOT set\n\n"
                    "If your data was acquired in DFD mode, THESE DEFAULTS ARE VERY "
                    "LIKELY WRONG and will corrupt your analysis.\n\n"
                    "You must explicitly set the correct DFD parameters in your "
                    "analysis code.\n"
                    "==========================================="
                ),
                stacklevel=2,
            )
            return default_cycle_mhz, None

        parsed_cycle_mhz = int(match.group("cycle"))
        parsed_bins = int(match.group("bins"))
        if not (3 < parsed_cycle_mhz < 100 and 3 < parsed_bins < 1000):
            warnings.warn(
                (
                    "\n"
                    "================ WARNING ==============\n"
                    "brighteyes_ism.dataio.mcs.load() failed to extract DFD metadata "
                    f"from the bitfile name ({filename!r}).\n\n"
                    "Falling back to defaults:\n"
                    f"  - Laser cycle frequency: {default_cycle_mhz} MHz\n"
                    "  - DFD bin count: NOT set\n\n"
                    "If your data was acquired in DFD mode, THESE DEFAULTS ARE VERY "
                    "LIKELY WRONG and will corrupt your analysis.\n\n"
                    "You must explicitly set the correct DFD parameters in your "
                    "analysis code.\n"
                    "==========================================="
                ),
                stacklevel=2,
            )
            return default_cycle_mhz, None

        return parsed_cycle_mhz, parsed_bins

    def Print(self):
        dic = self.__dict__
        names = list(dic)
        values = list(dic.values())
        for n, name in enumerate(names):
            print(name, end='')
            print(' ' * int(14 - len(name)), end='')
            print(str(values[n]))


def metadata_load(fname: str):
    '''
    It loads the metadata of a mcs h5 file.

    Parameters
    ----------
    fname : str
        h5 file path.

    Returns
    -------
    meta : metadata object
        MCS metadata of the dataset.

    '''

    with h5py.File(fname, "r") as f:
        meta = metadata(f)
        return meta


def metadata_print(fname: str):
    '''
    It prints the metadata of a mcs h5 file.

    Parameters
    ----------
    fname : str
        h5 file path.

    Returns
    -------
    None.

    '''

    with h5py.File(fname, "r") as f:
        meta = metadata(f)
        meta.Print()


def load(fname: str, key: str = 'data', data_format: str = 'numpy'):
    '''
    Return numpy array with image data from MCS .h5 file

    Parameters
    ----------
    fname : str
        h5 file path.
    key : str, optional
        Key name of the stored data. The default is 'data'.
    data_format : str, optional
        Return the data either as numpy array ('numpy') or as h5 file ('h5'). The default is 'numpy'.

    Returns
    -------
    data: ndarray
        ISM dataset.
    meta : metadata object
        MCS metadata of the dataset.

    '''

    with h5py.File(fname, "r") as f:
        data = f[key]
        meta = metadata(f)
        if data_format == 'numpy':
            return data[:], meta
        elif data_format == 'h5':
            return data, meta
        
