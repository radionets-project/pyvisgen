from astropy.io import fits
import numpy as np
from astropy import wcs
import astropy.units as un
from astropy.time import Time
import pandas as pd
import astropy.constants as const


def create_vis_hdu(data, conf, layout="EHT", source_name="sim-source-0"):
    u = data.u

    v = data.v

    w = data.w

    DATE = data.date - int(
        data.date.min()
    )  # placeholder, julian date of vis, central time in the integration period
    print(DATE)

    # I think this is not really needed, but dunno, documentation is again insane
    _DATE = (
        data._date
    )  # relative julian date for the observation day??, central time in the integration period

    BASELINE = data.base_num

    INTTIM = np.repeat(np.array(conf["corr_int_time"], dtype=">f4"), len(u))

    # visibility data
    values = data.get_values()
    vis = np.swapaxes(
        np.stack([values.real, values.imag, np.ones(values.shape)], axis=1), 1, 2
    ).reshape(-1, 1, 1, 1, 1, 4, 3)
    DATA = vis  # placeholder, get from sim
    # in dim 4 = IFs , dim = 1, dim 4 = number of jones, 3 = real, imag, weight

    # wcs
    ra = conf["src_coord"].ra.value
    dec = conf["src_coord"].dec.value
    freq, freq_d = freq, freq_d = (
        (np.array(conf["channel"].split(":")).astype("int") * un.MHz).to(un.Hz).value
    )
    ws = wcs.WCS(naxis=7)
    ws.wcs.crpix = [1, 1, 1, 1, 1, 1, 1]
    ws.wcs.cdelt = np.array([1, 1, -1, freq_d, 1, 1, 1])
    ws.wcs.crval = [1, 1, -5, freq, 1, ra, dec]
    ws.wcs.ctype = ["", "COMPLEX", "STOKES", "FREQ", "IF", "RA", "DEC"]
    h = ws.to_header()

    scale = 1  # / freq
    u_scale = u / const.c
    v_scale = v / const.c
    w_scale = w / const.c
    print(u_scale)
    groupdata_vis = fits.GroupData(
        DATA,
        bitpix=-32,
        parnames=[
            "UU---SIN",
            "VV---SIN",
            "WW---SIN",
            "BASELINE",
            "DATE",
            "_DATE",
            "INTTIM",
        ],
        pardata=[u_scale, v_scale, w_scale, BASELINE, DATE, _DATE, INTTIM],
    )

    hdu_vis = fits.GroupsHDU(groupdata_vis, header=h)

    # add scales and zeors
    parbscales = [scale, scale, scale, 1, 1, 1, 1]
    parbzeros = [0, 0, 0, 0, int(data.date.min()), 0, 0]

    for i in range(len(parbscales)):
        hdu_vis.header["PSCAL" + str(i + 1)] = parbscales[i]
        hdu_vis.header["PZERO" + str(i + 1)] = parbzeros[i]

    # add comments
    hdu_vis.header.comments["PTYPE1"] = "u baseline coordinate in light seconds"
    hdu_vis.header.comments["PTYPE2"] = "v baseline coordinate in light seconds"
    hdu_vis.header.comments["PTYPE3"] = "w baseline coordinate in light seconds"
    hdu_vis.header.comments["PTYPE4"] = "Baseline number"
    hdu_vis.header.comments["PTYPE5"] = "Julian date"
    hdu_vis.header.comments["PTYPE6"] = "Relative Julian date ?"
    hdu_vis.header.comments["PTYPE7"] = "Integration time"

    date_obs = Time(conf["scan_start"], format="yday").to_value(
        format="iso", subfmt="date"
    )
    date_map = Time.now().to_value(format="iso", subfmt="date")

    # add additional keys
    hdu_vis.header["EXTNAME"] = ("AIPS UV", "AIPS UV")
    hdu_vis.header["EXTVER"] = (1, "Version number of table")
    hdu_vis.header["OBJECT"] = (source_name, "Source name")
    hdu_vis.header["TELESCOP"] = (layout, "Telescope name")
    hdu_vis.header["INSTRUME"] = (layout, "Instrument name (receiver or ?)")
    hdu_vis.header["DATE-OBS"] = (date_obs, "Observation date")
    hdu_vis.header["DATE-MAP"] = (date_map, "File processing date")
    hdu_vis.header["BSCALE"] = (1, "Always 1")
    hdu_vis.header["BZERO"] = (0, "Always 0")
    hdu_vis.header["BUNIT"] = ("UNCALIB", "Units of flux")
    hdu_vis.header["EQUINOX"] = (2000, "Equinox of source coordinates and uvw")
    hdu_vis.header["ALTRPIX"] = (1, "Reference pixel for velocity")  # not quite sure
    hdu_vis.header["OBSRA"] = (ra, "Antenna pointing Ra")
    hdu_vis.header["OBSDEC"] = (dec, "Antenna pointing Dec")

    return hdu_vis


def create_time_hdu(data):
    TIME = np.array(
        [
            data[data.scan == i].date.mean() - int(data.date.min())
            for i in np.unique(data.scan)
        ],
        dtype=">f4",
    )
    col1 = fits.Column(name="TIME", format="1E", unit="days", array=TIME)

    TIME_INTERVAL = np.array(
        [
            (data[data.scan == i].date.max() - data[data.scan == i].date.min())
            for i in np.unique(data.scan)
        ],
        dtype=">f4",
    )
    col2 = fits.Column(
        name="TIME INTERVAL", format="1E", unit="days", array=TIME_INTERVAL
    )

    SOURCE_ID = np.ones(
        len(np.unique(data.scan)), dtype=">i4"
    )  # always the same source
    col3 = fits.Column(name="SOURCE ID", format="1J", unit=" ", array=SOURCE_ID)

    SUBARRAY = np.ones(len(np.unique(data.scan)), dtype=">i4")  # always same array
    col4 = fits.Column(name="SUBARRAY", format="1J", unit=" ", array=SUBARRAY)

    FREQ_ID = np.ones(len(np.unique(data.scan)), dtype=">i4")  # always same frequencies
    col5 = fits.Column(name="FREQ ID", format="1J", unit=" ", array=FREQ_ID)

    START_VIS = np.array(
        [data[data.scan == i].num.min() for i in np.unique(data.scan)],
        dtype=">i4",
    )
    col6 = fits.Column(name="START VIS", format="1J", unit=" ", array=START_VIS)

    END_VIS = np.array(
        [data[data.scan == i].num.max() for i in np.unique(data.scan)],
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


def create_frequency_hdu(conf):
    freq, freq_d = freq, freq_d = (
        (np.array(conf["channel"].split(":")).astype("int") * un.MHz).to(un.Hz).value
    )
    num_ifs = 1  # at the moment only 1 possible

    FRQSEL = np.array([1], dtype=">i4")
    col1 = fits.Column(name="FRQSEL", format="1J", unit=" ", array=FRQSEL)

    IF_FREQ = np.array([[0.00e00]], dtype=">f8")  # start with 0, add ch_with per IF
    col2 = fits.Column(
        name="IF FREQ", format=str(IF_FREQ.shape[-1]) + "D", unit="Hz", array=IF_FREQ
    )

    CH_WIDTH = np.repeat(np.array([[freq_d]], dtype=">f4"), 4, axis=1)
    col3 = fits.Column(
        name="CH WIDTH", format=str(CH_WIDTH.shape[-1]) + "E", unit="Hz", array=CH_WIDTH
    )

    TOTAL_BANDWIDTH = np.repeat(np.array([[freq_d]], dtype=">f4"), 4, axis=1)
    col4 = fits.Column(
        name="TOTAL BANDWIDTH",
        format=str(TOTAL_BANDWIDTH.shape[-1]) + "E",
        unit="Hz",
        array=TOTAL_BANDWIDTH,
    )

    SIDEBAND = np.zeros((1, IF_FREQ.shape[-1]))
    SIDEBAND[IF_FREQ >= 0] = 1
    SIDEBAND[IF_FREQ < 0] = -1
    col5 = fits.Column(
        name="SIDEBAND", format=str(SIDEBAND.shape[-1]) + "J", unit=" ", array=SIDEBAND
    )

    RXCODE = np.chararray(1, itemsize=32, unicode=True)
    RXCODE[:] = ""
    col6 = fits.Column(name="RXCODE", format="32A", unit=" ", array=RXCODE)

    coldefs_freq = fits.ColDefs([col1, col2, col3, col4, col5, col6])
    hdu_freq = fits.BinTableHDU.from_columns(coldefs_freq)

    # add additional keywords
    hdu_freq.header["EXTNAME"] = ("AIPS FQ", "AIPS FQ")
    hdu_freq.header["EXTVER"] = (1, "Version number of table")
    hdu_freq.header["NO_IF"] = (IF_FREQ.shape[-1], "Number IFs (n IF)")

    # add comments
    hdu_freq.header.comments["TTYPE1"] = "Frequency setup ID number"
    hdu_freq.header.comments["TTYPE2"] = "Frequency offset"
    hdu_freq.header.comments["TTYPE3"] = "Spectral channel separation"
    hdu_freq.header.comments["TTYPE4"] = "Total width of spectral window"
    hdu_freq.header.comments["TTYPE5"] = "Sideband"
    hdu_freq.header.comments["TTYPE6"] = "No one knows..."

    return hdu_freq


def create_antenna_hdu(layout_txt, conf, layout="EHT"):
    array = pd.read_csv(layout_txt, sep=" ")

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

    DIAMETER = np.array(array["dish_dia"].values, dtype=">f4")
    col7 = fits.Column(name="DIAMETER", format="1E", unit="METERS", array=DIAMETER)

    BEAMFWHM = np.zeros((len(array), 4), dtype=">f4")
    col8 = fits.Column(name="BEAMFWHM", format="4E", unit="DEGR/M", array=BEAMFWHM)

    POLTYA = np.chararray(len(array), itemsize=1, unicode=True)
    POLTYA[:] = "R"
    col9 = fits.Column(name="POLTYA", format="1A", unit=" ", array=POLTYA)

    POLAA = np.zeros(len(array), dtype=">f4")
    col10 = fits.Column(name="POLAA", format="1E", unit="DEGREES", array=POLAA)

    POLCALA = np.zeros((len(array), 8), dtype=">f4")
    col11 = fits.Column(name="POLCALA", format="8E", unit=" ", array=POLCALA)

    POLTYB = np.chararray(len(array), itemsize=1, unicode=True)
    POLTYB[:] = "L"
    col12 = fits.Column(name="POLTYB", format="1A", unit=" ", array=POLTYB)

    POLAB = np.zeros(len(array), dtype=">f4")
    col13 = fits.Column(name="POLAB", format="1E", unit="DEGREES", array=POLAB)

    POLCALB = np.zeros((len(array), 8), dtype=">f4")
    col14 = fits.Column(name="POLCALB", format="8E", unit=" ", array=POLCALB)

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
            col14,
        ]
    )
    hdu_ant = fits.BinTableHDU.from_columns(coldefs_ant)

    freq, freq_d = freq, freq_d = (
        (np.array(conf["channel"].split(":")).astype("int") * un.MHz).to(un.Hz).value
    )
    ref_date = Time(conf["scan_start"], format="yday")

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
        0.10819000005722046,
        "x coordinate of North Pole (arc seconds)",
    )  # MOJAVE
    hdu_ant.header["POLARY"] = (
        0.28815001249313354,
        "y coordinate of North Pole (arc seconds)",
    )  # MOJAVE
    hdu_ant.header["UT1UTC"] = (0, "UT1 - UTC (sec)")  # missing
    hdu_ant.header["DATUTC"] = (0, "time system - UTC (sec)")  # missing
    hdu_ant.header["TIMSYS"] = ("UTC", "Time system")
    hdu_ant.header["ARRNAM"] = (layout, "Array name")
    hdu_ant.header["XYZHAND"] = ("RIGHT", "Handedness of station coordinates")
    hdu_ant.header["FRAME"] = ("????", "Coordinate frame")
    hdu_ant.header["NUMORB"] = (0, "Number orbital parameters in table (n orb)")
    hdu_ant.header["NOPCAL"] = (
        2,
        "Number of polarization calibration values / IF(n pcal)",
    )
    hdu_ant.header["NO_IF"] = (4, "Number IFs (n IF)")
    hdu_ant.header["FREQID"] = (-1, "Frequency setup number")
    hdu_ant.header["IATUTC"] = (
        0,
        "No one knows.....",
    )  # how to calculate?? international atomic time
    hdu_ant.header["POLTYPE"] = (" ", "Type of polarization calibration")

    # add comments
    hdu_ant.header.comments["TTYPE1"] = "Antenna name"
    hdu_ant.header.comments["TTYPE2"] = "Antenna station coordinates (x, y, z)"
    hdu_ant.header.comments["TTYPE3"] = "Orbital parameters"
    hdu_ant.header.comments["TTYPE4"] = "Antenna number"
    hdu_ant.header.comments["TTYPE5"] = "Mount type"
    hdu_ant.header.comments["TTYPE6"] = "Axis offset"
    hdu_ant.header.comments["TTYPE7"] = "Antenna diameter"
    hdu_ant.header.comments["TTYPE8"] = "Antenna beam FWHM"
    hdu_ant.header.comments["TTYPE9"] = "R, L, feed A"
    hdu_ant.header.comments["TTYPE10"] = "Position angle feed A"
    hdu_ant.header.comments["TTYPE11"] = "Calibration parameters feed A"
    hdu_ant.header.comments["TTYPE12"] = "R, L, polarization 2"
    hdu_ant.header.comments["TTYPE13"] = "Position angle feed B"
    hdu_ant.header.comments["TTYPE14"] = "Calibration parameters feed B"

    return hdu_ant


def create_hdu_list(data, conf, path="../vipy/layouts/eht.txt"):
    vis_hdu = create_vis_hdu(data, conf)
    time_hdu = create_time_hdu(data)
    freq_hdu = create_frequency_hdu(conf)
    ant_hdu = create_antenna_hdu(path, conf)
    hdu_list = fits.HDUList([vis_hdu, time_hdu, freq_hdu, ant_hdu])
    return hdu_list
