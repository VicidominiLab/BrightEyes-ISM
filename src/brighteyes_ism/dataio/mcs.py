import h5py

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
    
    __slots__ = ['version', 'comment', 'rangex', 'rangey', 'rangez', 'nbin', 'dt', 'nx', 'ny', 'nz', 'nrep', 'calib_x', 'calib_y', 'calib_z']
    
    def __init__(self, f):
            
        # MCS version
        self.version = f.attrs['data_format_version']
        self.comment = f.attrs['comment']
        
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
            
    @property
    def pxdwelltime(self):
        # pixel dwell time in us
        return(self.dt * self.nbin)
    
    @property
    def frametime(self):
        # frame time in s
        return(self.pxdwelltime * self.nx * self.ny / 1e6)

    @property
    def framerate(self):
        # frame rate in Hz
        return(1 / self.frametime)

    @property
    def dx(self):
        # pixel size in x direction
        return(self.rangex / self.nx)
    
    @property
    def dy(self):
        # pixel size in y direction
        return(self.rangey / self.ny)
    
    @property
    def dz(self):
        # pixel size in z direction
        return(self.rangez / self.nz)
    
    @property
    def nmicroim(self):
        # total number of microimages read during the measurement
        return(self.nx * self.ny * self.nz * self.nrep * self.nbin)
    
    @property
    def ndatapoints(self):
        # total number of words transferred from low level to high level
        # 2 words per microimage
        return(2 * self.nmicroim)
    
    @property
    def duration(self):
        # total measurement duration in s
        return(self.nmicroim * self.dt * 1e-6)
    
    def Print(self):
        for prop in self.__slots__:
            print(prop, end = '')
            print(' ' * int(14 - len(prop)), end = '')
            print(str(getattr(self, prop)))
            

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

def load(fname: str, key: str ='data'):
    '''
    Return numpy array with image data from MCS .h5 file

    Parameters
    ----------
    fname : str
        h5 file path.
    key : str, optional
        Key name of the stored data. The default is 'data'.

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
        return data[:], meta