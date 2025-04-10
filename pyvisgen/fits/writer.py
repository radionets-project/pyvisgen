import warnings

import astropy.constants as const
import astropy.units as un
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.time import Time
from astropy.utils import iers

import pyvisgen.layouts.layouts as layouts


def create_vis_hdu(data, obs, source_name="sim-source-0") -> fits.GroupsHDU:
    """Creates the visibility HDU for the given visibility
    data and observation.

    Parameters
    ----------
    data : :class:`~pyvisgen.simulation.Visibilities`
        :class:`~pyvisgen.simulation.Visibilities` object
        containing visibility data.
    obs : :class:`~pyvisgen.simulation.Observation`
        :class:`~pyvisgen.simulation.Observation` class object
        containing information on the observation, such as
        baselines.
    source_name : str, optional
        Source name saved to the ``OBJECT`` key inside the
        FITS file. Default: ``'sim-source-0'``

    Returns
    -------
    hdu_vis : :class:`~astropy.io.fits.GroupsHDU`
        :class:`~astropy.io.fits.GroupsHDU` containing visibility
        data for the FITS file.
    """
    u = data.u

    v = data.v

    w = data.w

    DATE = np.trunc(data.date)

    _DATE = data.date % 1

    BASELINE = data.base_num

    FREQSEL = np.repeat(np.array([1.0], dtype=">f4"), len(u))

    # visibility data
    values = data.get_values()

    vis = np.stack([values.real, values.imag, np.ones(values.shape)], axis=3)[
        :, None, None, None, ...
    ]

    DATA = vis
    # in dim 4 = IFs , dim = 1, dim 4 = number of jones, 3 = real, imag, weight

    # wcs
    ra = obs.ra.cpu().numpy().item()
    dec = obs.dec.cpu().numpy().item()
    freq = obs.ref_frequency.cpu().numpy().item()
    freq_d = obs.bandwidths[0].cpu().numpy().item()

    ws = wcs.WCS(naxis=7)

    crval_stokes = -1
    stokes_comment = "-1=RR, -2=LL, -3=RL, -4=LR"
    if obs.polarization == "linear":
        crval_stokes = -5
        stokes_comment = "-5=XX, -6=YY, -7=XY, -8=YX"
        stokes_comment += " or -5=VV, -6=HH, -7=VH, -8=HV"

    ws.wcs.crpix = [0, 1, 1, 1, 1, 1, 1]
    ws.wcs.crota = [0, 0, 0, 0, 0, 0, 0]
    ws.wcs.cdelt = [1, 1, -1, freq_d, 1, 1, 1]
    ws.wcs.crval = [0, 1, crval_stokes, freq, 1, ra, dec]
    ws.wcs.ctype = ["", "COMPLEX", "STOKES", "FREQ", "IF", "RA", "DEC"]

    ws.wcs.specsys = "TOPOCENTER"
    ws.wcs.radesys = "FK5"
    ws.wcs.equinox = 2000.0
    ws.wcs.dateobs = obs.start.isot
    ws.wcs.mjdobs = obs.start.mjd

    h = ws.to_header()

    u_scale = u / const.c
    v_scale = v / const.c
    w_scale = w / const.c

    groupdata_vis = fits.GroupData(
        DATA,
        bitpix=-32,
        parnames=[
            "UU",
            "VV",
            "WW",
            "DATE",
            "DATE",
            "BASELINE",
            "FREQSEL",
        ],
        pardata=[u_scale, v_scale, w_scale, DATE, _DATE, BASELINE, FREQSEL],
    )

    hdu_vis = fits.GroupsHDU(groupdata_vis, header=h)

    # add scales and zeors
    scale = 1.0
    zero = 0.0
    parbscales = [scale, scale, scale, scale, scale, scale, scale]
    parbzeros = [zero, zero, zero, zero, zero, zero, zero]

    for i in range(len(parbscales)):
        hdu_vis.header["PSCAL" + str(i + 1)] = parbscales[i]
        hdu_vis.header["PZERO" + str(i + 1)] = parbzeros[i]

    # add comments
    hdu_vis.header.comments["CTYPE2"] = "1=real, 2=imag, 3=weight"
    hdu_vis.header.comments["CTYPE3"] = stokes_comment
    hdu_vis.header.comments["PTYPE1"] = "u baseline coordinate in light seconds"
    hdu_vis.header.comments["PTYPE2"] = "v baseline coordinate in light seconds"
    hdu_vis.header.comments["PTYPE3"] = "w baseline coordinate in light seconds"
    hdu_vis.header.comments["PTYPE4"] = "Baseline number"
    hdu_vis.header.comments["PTYPE5"] = "Julian date"
    hdu_vis.header.comments["PTYPE6"] = "Relative Julian date ?"
    hdu_vis.header.comments["PTYPE7"] = "Integration time"

    date_obs = obs.start.strftime("%Y-%m-%d")

    date_map = Time.now().to_value(format="iso", subfmt="date")

    # add additional keys
    hdu_vis.header["EXTNAME"] = ("AIPS UV", "AIPS UV")
    hdu_vis.header["EXTVER"] = (1, "Version number of table")
    hdu_vis.header["OBJECT"] = (source_name, "Source name")
    hdu_vis.header["TELESCOP"] = (obs.layout, "Telescope name")
    hdu_vis.header["INSTRUME"] = (obs.layout, "Instrument name (receiver or ?)")
    hdu_vis.header["DATE-OBS"] = (date_obs, "Observation date")
    hdu_vis.header["DATE-MAP"] = (date_map, "File processing date")
    hdu_vis.header["EPOCH"] = (2000.0, "")
    hdu_vis.header["BSCALE"] = (1.0, "Always 1")
    hdu_vis.header["BZERO"] = (0.0, "Always 0")
    hdu_vis.header["BUNIT"] = ("UNCALIB", "Units of flux")
    hdu_vis.header["ALTRPIX"] = (1.0, "Reference pixel for velocity")
    hdu_vis.header["OBSRA"] = (ra, "Antenna pointing Ra")
    hdu_vis.header["OBSDEC"] = (dec, "Antenna pointing Dec")

    return hdu_vis


def create_time_hdu(data) -> fits.BinTableHDU:
    """Creates the time HDU for the FITS file.

    Parameters
    ----------
    data : :class:`~pyvisgen.simulation.Visibilities`
        :class:`~pyvisgen.simulation.Visibilities` object
        containing visibility and time data.

    Returns
    -------
    hdu_vis : :class:`~astropy.io.fits.BinTableHDU`
        :class:`~astropy.io.fits.BinTableHDU` containing time
        data for the FITS file.
    """
    TIME = np.array(
        [data.date.mean() - int(data.date.min())],
        dtype=">f4",
    )
    col1 = fits.Column(name="TIME", format="1E", unit="days", array=TIME)

    TIME_INTERVAL = np.array(
        [data.date.max() - data.date.min()],
        dtype=">f4",
    )
    col2 = fits.Column(
        name="TIME INTERVAL", format="1E", unit="days", array=TIME_INTERVAL
    )

    SOURCE_ID = np.ones((1), dtype=">i4")  # always the same source
    col3 = fits.Column(name="SOURCE ID", format="1J", unit=" ", array=SOURCE_ID)

    SUBARRAY = np.ones((1), dtype=">i4")  # always same array
    col4 = fits.Column(name="SUBARRAY", format="1J", unit=" ", array=SUBARRAY)

    FREQ_ID = np.ones((1), dtype=">i4")  # always same frequencies
    col5 = fits.Column(name="FREQ ID", format="1J", unit=" ", array=FREQ_ID)

    START_VIS = np.array(
        [data.num.min()],
        dtype=">i4",
    )
    col6 = fits.Column(name="START VIS", format="1J", unit=" ", array=START_VIS)

    END_VIS = np.array(
        [data.num.max()],
        dtype=">i4",
    )
    col7 = fits.Column(name="END VIS", format="1J", unit=" ", array=END_VIS)

    coldefs_time = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7])
    hdu_time = fits.BinTableHDU.from_columns(coldefs_time)

    # add additional keywords
    hdu_time.header["EXTNAME"] = ("AIPS NX", "AIPS NX")
    hdu_time.header["EXTVER"] = (1, "Version number of table")

    # add comments
    hdu_time.header.comments["TTYPE1"] = "Center time of interval"
    hdu_time.header.comments["TTYPE2"] = "Duration of interval"
    hdu_time.header.comments["TTYPE3"] = "Source number"
    hdu_time.header.comments["TTYPE4"] = "Subarray"
    hdu_time.header.comments["TTYPE5"] = "Frequency setup ID number"
    hdu_time.header.comments["TTYPE6"] = "First visibility number"
    hdu_time.header.comments["TTYPE7"] = "End visibility number"

    return hdu_time


def create_frequency_hdu(obs) -> fits.BinTableHDU:
    """Creates the frequency HDU of the FITS file.

    Parameters
    ----------
    obs : :class:`~pyvisgen.simulation.Observation`
        :class:`~pyvisgen.simulation.Observation` class object
        containing information on the observation, including
        frequency data.

    Returns
    -------
    hdu_freq : :class:`~astropy.io.fits.BinTableHDU`
        :class:`~astropy.io.fits.BinTableHDU` containing
        frequency data for the FITS file.
    """
    FRQSEL = np.array([1], dtype=">i4")
    col1 = fits.Column(name="FRQSEL", format="1J", array=FRQSEL)

    IF_FREQ = np.array(
        [0.0],
        dtype=">f8",
    )  # start with 0, add ch_with per IF
    col2 = fits.Column(name="IF FREQ", format="1D", unit="Hz", array=IF_FREQ)

    CH_WIDTH = np.array([obs.bandwidths[0].cpu().numpy()], dtype=">f4")
    col3 = fits.Column(name="CH WIDTH", format="1E", unit="Hz", array=CH_WIDTH)

    TOTAL_BANDWIDTH = np.array(
        [(obs.bandwidths[0] * len(obs.bandwidths)).cpu().numpy()], dtype=">f4"
    )
    col4 = fits.Column(
        name="TOTAL BANDWIDTH",
        format=str(TOTAL_BANDWIDTH.shape[-1]) + "E",
        unit="Hz",
        array=TOTAL_BANDWIDTH,
    )

    SIDEBAND = np.array([1])
    col5 = fits.Column(name="SIDEBAND", format="1J", unit=" ", array=SIDEBAND)

    coldefs_freq = fits.ColDefs([col1, col2, col3, col4, col5])
    hdu_freq = fits.BinTableHDU.from_columns(coldefs_freq)

    # add additional keywords
    hdu_freq.header["EXTNAME"] = ("AIPS FQ", "AIPS FQ")
    hdu_freq.header["EXTVER"] = (1, "Version number of table")

    # add comments
    hdu_freq.header.comments["TTYPE1"] = "Frequency setup ID number"
    hdu_freq.header.comments["TTYPE2"] = "Frequency offset"
    hdu_freq.header.comments["TTYPE3"] = "Spectral channel separation"
    hdu_freq.header.comments["TTYPE4"] = "Total width of spectral window"
    hdu_freq.header.comments["TTYPE5"] = "Sideband"

    return hdu_freq


def create_antenna_hdu(obs) -> fits.BinTableHDU:
    """Creates the antenna HDU for the FITS file.

    Parameters
    ----------
    obs : :class:`~pyvisgen.simulation.Observation`
        :class:`~pyvisgen.simulation.Observation` class object
        containing information on the observation, including
        antenna data.

    Returns
    -------
    hdu_ant : :class:`~astropy.io.fits.BinTableHDU`
        :class:`~astropy.io.fits.BinTableHDU` containing
        antenna data for the FITS file.
    """
    array = layouts.get_array_layout(obs.layout, writer=True)

    ANNAME = np.chararray(len(array), itemsize=8, unicode=True)
    ANNAME[:] = array["station_name"].values
    col1 = fits.Column(name="ANNAME", format="8A", array=ANNAME)

    STABXYZ = np.array([array["X"], array["Y"], array["Z"]], dtype=">f8").T
    col2 = fits.Column(name="STABXYZ", format="3D", unit="METERS", array=STABXYZ)

    ORBPARM = np.array([], dtype=">f8")
    col3 = fits.Column(name="ORBPARM", format="0D", unit=" ", array=ORBPARM)

    NOSTA = np.arange(len(array), dtype=">i4") + 1
    col4 = fits.Column(name="NOSTA", format="1J", unit=" ", array=NOSTA)

    MNTSTA = np.zeros(len(array), dtype=">i4")
    col5 = fits.Column(name="MNTSTA", format="1J", unit=" ", array=MNTSTA)

    STAXOF = np.zeros(len(array), dtype=">f4")
    col6 = fits.Column(name="STAXOF", format="1E", unit="METERS", array=STAXOF)

    POLTYA = np.chararray(len(array), itemsize=1, unicode=True)
    POLTYA[:] = "X"
    col7 = fits.Column(name="POLTYA", format="1A", unit=" ", array=POLTYA)

    POLAA = np.ones(len(array), dtype=">f4") * -90
    col8 = fits.Column(name="POLAA", format="1E", unit="DEGREES", array=POLAA)

    POLCALA = np.zeros((len(array)), dtype=">f4")
    col9 = fits.Column(name="POLCALA", format="1E", unit=" ", array=POLCALA)

    POLTYB = np.chararray(len(array), itemsize=1, unicode=True)
    POLTYB[:] = "Y"
    col10 = fits.Column(name="POLTYB", format="1A", unit=" ", array=POLTYB)

    POLAB = np.ones(len(array), dtype=">f4") * -90
    col11 = fits.Column(name="POLAB", format="1E", unit="DEGREES", array=POLAB)

    POLCALB = np.zeros((len(array)), dtype=">f4")
    col12 = fits.Column(name="POLCALB", format="1E", unit=" ", array=POLCALB)

    DIAMETER = np.array(array["dish_dia"].values, dtype=">f4")
    col13 = fits.Column(name="DIAMETER", format="1E", unit="METERS", array=DIAMETER)

    coldefs_ant = fits.ColDefs(
        [
            col1,
            col2,
            col3,
            col4,
            col5,
            col6,
            col7,
            col8,
            col9,
            col10,
            col11,
            col12,
            col13,
        ]
    )
    hdu_ant = fits.BinTableHDU.from_columns(coldefs_ant)

    freq = (obs.ref_frequency.cpu().numpy() * un.Hz).value
    ref_date = Time(
        obs.start.isot.split("T")[0] + "T0:00:00.000", format="isot", scale="utc"
    )

    iers_b = iers.IERS_B.open()

    # add additional keywords
    hdu_ant.header["EXTNAME"] = ("AIPS AN", "AIPS table file")
    hdu_ant.header["EXTVER"] = (1, "Version number of table")
    hdu_ant.header["ARRAYX"] = (0, "x coordinate of array center (meters)")
    hdu_ant.header["ARRAYY"] = (0, "y coordinate of array center (meters)")
    hdu_ant.header["ARRAYZ"] = (0, "z coordinate of array center (meters)")
    hdu_ant.header["GSTIA0"] = (
        ref_date.sidereal_time("apparent", "greenwich").deg,
        "GST at 0h on reference date (degrees)",
    )
    hdu_ant.header["DEGPDY"] = (
        360.98564497329994,
        "Earth's rotation rate (degrees/day)",
    )
    hdu_ant.header["FREQ"] = (freq, "Reference frequency (Hz)")
    hdu_ant.header["RDATE"] = (
        ref_date.to_value(format="iso", subfmt="date_hms"),
        "Reference date",
    )
    hdu_ant.header["POLARX"] = (
        iers_b.pm_xy(ref_date)[0].value,
        "x coordinate of North Pole (arc seconds)",
    )
    hdu_ant.header["POLARY"] = (
        iers_b.pm_xy(ref_date)[1].value,
        "y coordinate of North Pole (arc seconds)",
    )
    hdu_ant.header["UT1UTC"] = (iers_b.ut1_utc(ref_date).value, "UT1 - UTC (sec)")
    hdu_ant.header["DATUTC"] = (0, "time system - UTC (sec)")  # missing
    hdu_ant.header["TIMSYS"] = ("UTC", "Time system")
    hdu_ant.header["ARRNAM"] = (obs.layout, "Array name")
    hdu_ant.header["XYZHAND"] = ("RIGHT", "Handedness of station coordinates")
    hdu_ant.header["FRAME"] = ("ITRF", "Coordinate frame")
    hdu_ant.header["NUMORB"] = (0, "Number orbital parameters in table (n orb)")
    hdu_ant.header["NOPCAL"] = (
        2,
        "Number of polarization calibration values / IF(n pcal)",
    )
    hdu_ant.header["NO_IF"] = (1, "Number IFs (n IF)")
    hdu_ant.header["FREQID"] = (1, "Frequency setup number")
    hdu_ant.header["IATUTC"] = (
        ref_date.tai.ymdhms[-1],
        "Difference between TAI and UTC",
    )
    hdu_ant.header["POLTYPE"] = (" ", "Type of polarization calibration")

    # add comments
    hdu_ant.header.comments["TTYPE1"] = "Antenna name"
    hdu_ant.header.comments["TTYPE2"] = "Antenna station coordinates (x, y, z)"
    hdu_ant.header.comments["TTYPE3"] = "Orbital parameters"
    hdu_ant.header.comments["TTYPE4"] = "Antenna number"
    hdu_ant.header.comments["TTYPE5"] = "Mount type"
    hdu_ant.header.comments["TTYPE6"] = "Axis offset"
    hdu_ant.header.comments["TTYPE7"] = "R, L, feed A"
    hdu_ant.header.comments["TTYPE8"] = "Position angle feed A"
    hdu_ant.header.comments["TTYPE9"] = "Calibration parameters feed A"
    hdu_ant.header.comments["TTYPE10"] = "R, L, polarization 2"
    hdu_ant.header.comments["TTYPE11"] = "Position angle feed B"
    hdu_ant.header.comments["TTYPE12"] = "Calibration parameters feed B"
    hdu_ant.header.comments["TTYPE13"] = "Antenna diameter"

    return hdu_ant


def create_hdu_list(data, obs) -> fits.HDUList:
    """Creates a :class:`~astropy.io.fits.HDUList` as the
    top-level object for the FITS file.

    Parameters
    ----------
    data : :class:`~pyvisgen.simulation.Visibilities`
        :class:`~pyvisgen.simulation.Visibilities` object containing
        data on visibilities and observation time.
    obs : :class:`~pyvisgen.simulation.Observation`
        :class:`~pyvisgen.simulation.Observation` object containing
        data on the source position, baselines, antenna configuration,
        and frequencies.

    Returns
    -------
    hdu_list : :class:`~astropy.io.fits.HDUList`
        :class:`~astropy.io.fits.HDUList` that comprises of
        HDU objects.
    """
    warnings.filterwarnings("ignore", module="astropy.io.fits")
    vis_hdu = create_vis_hdu(data, obs)
    time_hdu = create_time_hdu(data)
    freq_hdu = create_frequency_hdu(obs)
    ant_hdu = create_antenna_hdu(obs)
    hdu_list = fits.HDUList([vis_hdu, freq_hdu, ant_hdu, time_hdu])
    return hdu_list
