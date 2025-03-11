from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from astropy import units as un
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm

import pyvisgen.fits.writer as writer
import pyvisgen.layouts.layouts as layouts
from pyvisgen.gridding import (
    calc_truth_fft,
    convert_amp_phase,
    convert_real_imag,
    grid_vis_loop_data,
    save_fft_pair,
)
from pyvisgen.simulation.observation import Observation
from pyvisgen.simulation.visibility import vis_loop
from pyvisgen.utils.config import read_data_set_conf
from pyvisgen.utils.data import load_bundles, open_bundles

DATEFMT = "%d-%m-%Y %H:%M:%S"


class SimulateDataSet:
    def __init__(self):
        pass

    @classmethod
    def from_config(
        cls,
        config: str | Path | dict,
        /,
        image_key: str = "y",
        *,
        grid: bool = True,
        slurm: bool = False,
        slurm_job_id: int | None = None,
        slurm_n: int | None = None,
        date_fmt: str = DATEFMT,
        num_images: int | None = None,
        multiprocess: int | str = 1,
    ):
        """Simulates data from parameters in a config file.

        Parameters
        ----------
        config : str or Path or dict
            Path to the config file or dict containing the configuration
            parameters.
        image_key : str, optional
            Key under which the true sky distributions are saved
            in the HDF5 file. Default: ``'y'``
        grid : bool, optional
            If ``True``, apply gridding to visibility data and
            save to HDF5 files. Default: ``True``
        slurm : bool, optional
            ``True``, if slurm is used, Default: ``False``
        slurm_job_id : int or None, optional
            ``job_id`` given by slurm. Default: ``None``
        slurm_n : int or None, optional
            Running index. Default: ``None``
        date_fmt : str, optional
            Format string for datetime objects.
            Default: ``'%d-%m-%Y %H:%M:%S'``
        num_images : int or None, optional
            Number of combined total images in the bundles.
            If not ``None``, will skip counting the images before
            drawing the random parameters. Default: ``None``
        multiprocess : int or str, optional
            Number of jobs to use in multiprocessing during the
            sampling and testing phase. If -1 or ``'all'``,
            use all available cores. Default: 1
        """
        cls = cls()
        cls.key = image_key
        cls.grid = grid
        cls.slurm = slurm
        cls.job_id = slurm_job_id
        cls.n = slurm_n
        cls.date_fmt = date_fmt
        cls.num_images = num_images
        cls.multiprocess = multiprocess

        if multiprocess in ["all"]:
            cls.multiprocess = -1

        if isinstance(config, (str, Path)):
            cls.conf = read_data_set_conf(config)
        elif isinstance(config, dict):
            cls.conf = config
        else:
            raise ValueError("Expected config to be one of str, Path or dict!")

        print("Simulation Config:\n", cls.conf)

        if grid:
            cls.out_path = Path(cls.conf["out_path_gridded"])
        else:
            cls.out_path = Path(cls.conf["out_path_fits"])

        if not cls.out_path.is_dir():
            cls.out_path.mkdir(parents=True)

        cls.data_paths = load_bundles(cls.conf["in_path"])

        if not cls.num_images:
            data_bundles = tqdm(
                range(len(cls.data_paths)),
                position=0,
                leave=False,
                desc="Counting images",
                colour="#754fc9",
            )
            # get number of random parameter draws from number of images in data
            cls.num_images = np.sum(
                [len(cls.get_images(bundle)) for bundle in data_bundles]
            )

        if isinstance(cls.num_images, (int, float)):
            if int(cls.num_images) == 0:
                raise ValueError(
                    "No images found in bundles! Please check your input path!"
                )

        if slurm:  # pragma: no cover
            cls._run_slurm()
        else:
            # draw parameters beforehand, i.e. outside the simulation loop
            cls.create_sampling_rc(cls.num_images)
            cls._run()

        return cls

    def _run(self) -> None:
        """Runs the simulation and saves visibility data either as
        bundled HDF5 files or as individual FITS files.
        """
        data = tqdm(
            range(len(self.data_paths)),
            position=0,
            desc="Processing bundles",
            colour="#52ba66",
        )

        samp_opts_idx = 0
        for i in data:
            SIs = self.get_images(i)
            truth_fft = calc_truth_fft(SIs)

            SIs = tqdm(
                SIs,
                position=1,
                desc=f"Bundle {i + 1}",
                colour="#595cbd",
                leave=False,
            )

            sim_data = []
            for SI in SIs:
                obs = self.create_observation(samp_opts_idx)
                vis = vis_loop(
                    obs, SI, noisy=self.conf["noisy"], mode=self.conf["mode"]
                )

                if self.grid:
                    gridded = grid_vis_loop_data(
                        vis.u, vis.v, vis.get_values(), self.freq_bands, self.conf
                    )

                    sim_data.append(gridded)

                samp_opts_idx += 1

            sim_data = np.array(sim_data)

            if self.grid:
                if self.conf["amp_phase"]:
                    sim_data = convert_amp_phase(sim_data, sky_sim=False)
                    truth_fft = convert_amp_phase(truth_fft, sky_sim=True)
                else:
                    sim_data = convert_real_imag(sim_data, sky_sim=False)
                    truth_fft = convert_real_imag(truth_fft, sky_sim=True)

                if sim_data.shape[1] != 2:
                    raise ValueError("Expected sim_data axis 1 to be 2!")

                out = self.out_path / Path(
                    f"samp_{self.conf['file_prefix']}_" + str(i) + ".h5"
                )

                save_fft_pair(path=out, x=sim_data, y=truth_fft)

                path_msg = Path(self.conf["out_path_gridded"]) / Path(
                    f"samp_{self.conf['file_prefix']}_<id>.h5"
                )
            else:
                for i, vis_data in enumerate(sim_data):
                    out = self.out_path / Path(
                        f"vis_{self.conf['file_prefix']}_" + str(i) + ".fits"
                    )
                    hdu_list = writer.create_hdu_list(vis_data, obs)
                    hdu_list.writeto(out, overwrite=True)

                path_msg = self.conf["out_path_fits"] / Path(
                    f"samp_{self.conf['file_prefix']}_<id>.fits"
                )

        print(
            f"Successfully simulated and saved {samp_opts_idx} images "
            f"to '{path_msg}'!"
        )

    def _run_slurm(self) -> None:  # pragma: no cover
        """Runs the simulation in slurm and saves visibility data
        as individual FITS files.
        """
        job_id = int(self.slurm_job_id + self.slurm_n * 500)
        out = self.conf["out_path_fits"] / Path("vis_" + str(job_id) + ".fits")

        bundle = torch.div(job_id, self.num_images, rounding_mode="floor")
        image = job_id - bundle * self.num_images

        SI = torch.tensor(open_bundles(self.data_paths[bundle])[image])

        if len(SI.shape) == 2:
            SI = SI.unsqueeze(0)

        self.create_sampling_rc(1)
        obs = self.create_observation(0)
        vis_data = vis_loop(obs, SI, noisy=self.conf["noisy"], mode=self.conf["mode"])

        hdu_list = writer.create_hdu_list(vis_data, obs)
        hdu_list.writeto(out, overwrite=True)

    def get_images(self, i: int) -> torch.tensor:
        """Opens bundle with index i and returns :func:`~torch.tensor`
        of images.

        Parameters
        ----------
        i : int
            Bundle index.

        Returns
        -------
        SIs : :func:`~torch.tensor`
            :func:`~torch.tensor` of images from bundle ``i``.
        """
        SIs = torch.tensor(open_bundles(self.data_paths[i], key=self.key))

        if len(SIs.shape) == 3:
            SIs = SIs.unsqueeze(1)

        return SIs

    def create_observation(self, i: int) -> Observation:
        """Creates :class:`~pyvisgen.simulation.Observation`
        dataclass object for image ``i``.

        Parameters
        ----------
        i : int
            Index of image for which the observation is created.

        Returns
        -------
        obs : Observation
            :class:`~pyvisgen.simulation.Observation` dataclass
            object for image ``i``.
        """
        rc = self.samp_opts

        # put the respective values inside the
        # pol_kwargs and field_kwargs dicts.
        pol_kwargs = dict(
            delta=rc["delta"][i],
            amp_ratio=rc["amp_ratio"][i],
            random_state=self.conf["seed"],
        )
        field_kwargs = dict(
            order=rc["order"][i],
            scale=rc["scale"][i],
            threshold=rc["threshold"],
            random_state=self.conf["seed"],
        )

        dense = False
        if self.conf["mode"] == "dense":
            dense = True

        obs = Observation(
            **self.samp_opts_const,
            src_ra=rc["src_ra"][i],
            src_dec=rc["src_dec"][i],
            start_time=rc["start_time"][i],
            scan_duration=int(rc["scan_duration"][i]),
            num_scans=int(rc["num_scans"][i]),
            pol_kwargs=pol_kwargs,
            field_kwargs=field_kwargs,
            dense=dense,
        )

        return obs

    def create_sampling_rc(self, size: int) -> None:
        """Creates sampling runtime configuration containing
        all relevant parameters for the simulation.

        Parameters
        ----------
        size : int
            Number of parameters to draw, equal to number of images.
        """
        if self.conf["seed"]:
            self.rng = np.random.default_rng(self.conf["seed"])
        else:
            self.rng = np.random.default_rng()

        if self.conf["mode"] == "dense":
            self.freq_bands = np.array(self.conf["ref_frequency"])
        else:
            self.freq_bands = np.array(self.conf["ref_frequency"]) + np.array(
                self.conf["frequency_offsets"]
            )

        # Split sampling options into two dicts:
        # samps_ops_const is always the same, values in
        # samps_ops, however, will be drawn randomly.
        self.samp_opts_const = dict(
            array_layout=self.conf["layout"][0],
            image_size=self.conf["img_size"][0],
            fov=self.conf["fov_size"],
            integration_time=self.conf["corr_int_time"],
            scan_separation=self.conf["scan_separation"],
            ref_frequency=self.conf["ref_frequency"],
            frequency_offsets=self.conf["frequency_offsets"],
            bandwidths=self.conf["bandwidths"],
            corrupted=self.conf["corrupted"],
            device=self.conf["device"],
            sensitivity_cut=self.conf["sensitivty_cut"],
            polarisation=self.conf["polarisation"],
        )  # NOTE: scan_separation and integration_time may change in the future

        # get second half of the sampling options;
        # this is the randomly drawn, i.e. non-constant, part
        self.samp_opts = self.draw_sampling_opts(size)

        test_idx = tqdm(
            range(self.samp_opts["src_ra"].size),
            position=0,
            desc="Pre-drawing and testing sample parameters",
            colour="#00c1a2",
            leave=False,
        )

        self.array = layouts.get_array_layout(self.samp_opts_const["array_layout"])

        Parallel(n_jobs=self.multiprocess, backend="threading")(
            delayed(self.test_rand_opts)(i) for i in test_idx
        )

    def draw_sampling_opts(self, size: int) -> dict:
        """Draws randomized sampling parameters for the simulation.

        Parameters
        ----------
        size : int
            Number of parameters to draw, equal to number of images.

        Returns
        -------
        samp_opts : dict
            Sampling options/parameters stored inside a dictionary.
        """
        ra = self.rng.uniform(
            self.conf["fov_center_ra"][0][0], self.conf["fov_center_ra"][0][1], size
        )
        dec = self.rng.uniform(
            self.conf["fov_center_dec"][0][0], self.conf["fov_center_dec"][0][1], size
        )

        start_time_l = datetime.strptime(self.conf["scan_start"][0], self.date_fmt)
        start_time_h = datetime.strptime(self.conf["scan_start"][1], self.date_fmt)
        start_times = np.arange(start_time_l, start_time_h, timedelta(hours=1)).astype(
            datetime
        )

        scan_start = self.rng.choice(start_times, size)
        scan_duration = self.rng.integers(
            self.conf["scan_duration"][0],
            self.conf["scan_duration"][1],
            size,
        )
        num_scans = self.rng.integers(
            self.conf["num_scans"][0], self.conf["num_scans"][1], size
        )

        if scan_duration.size == 1:
            scan_duration = scan_duration.astype(int)

        if num_scans.size == 1:
            num_scans = num_scans.astype(int)

        # if polarisation is None, we don't need to enter the
        # conditional below, so we set delta, amp_ratio, field_order,
        # and field_scale to None.
        delta, amp_ratio, field_order, field_scale = np.full((4, size), None)

        if self.conf["polarisation"]:
            if self.conf["pol_delta"]:
                delta = np.repeat(self.conf["pol_delta"], size)
            else:
                delta = self.rng.uniform(0, 180, size)

            if self.conf["pol_amp_ratio"]:
                amp_ratio = np.repeat(self.conf["pol_amp_ratio"], size)
            else:
                amp_ratio = self.rng.uniform(0, 1, size)

            if self.conf["field_order"]:
                field_order = np.repeat(self.conf["field_order"], size).reshape(-1, 2)
            else:
                field_order = np.repeat(self.rng.uniform(0, 1, size), 2).reshape(-1, 2)

            if self.conf["field_scale"]:
                field_scale = np.stack(
                    np.repeat(self.conf["field_scale"], size).reshape(2, -1), axis=1
                )
            else:
                a = self.rng.uniform(0, 1 - 1e-6, size)
                b = np.repeat(1, size, dtype=float)
                field_scale = np.stack((a, b), axis=1)

        samp_opts = dict(
            src_ra=ra,
            src_dec=dec,
            start_time=scan_start,
            scan_duration=scan_duration,
            num_scans=num_scans,
            delta=delta,
            amp_ratio=amp_ratio,
            order=field_order,
            scale=field_scale,
            threshold=self.conf["field_threshold"],
        )
        # NOTE: We don't need to draw random values for threshold
        # as threshold=None should be suitable for almost all cases.
        # However, since threshold has to be in the field_kwargs dict
        # later, we need to include it here instead of inside the
        # samp_opts_const dictionary.

        return samp_opts

    def test_rand_opts(self, i: int) -> None:
        """Tests randomized sampling parameters by checking
        if the source is visible for 50% of the telescopes in
        the array for 50% of the observation time. If that
        condition is not fullfilled, the parameters are redrawn
        and tested again.

        Parameters
        ----------
        i : int
            Index of the current set of sampling parameters.
        """
        time_steps = self.calc_time_steps(i)
        src_crd = SkyCoord(
            self.samp_opts["src_ra"][i], self.samp_opts["src_dec"][i], unit=un.deg
        )

        total_stations = len(self.array.st_num)

        locations = EarthLocation.from_geocentric(
            x=self.array.x,
            y=self.array.y,
            z=self.array.z,
            unit=un.m,
        )

        altaz_frames = AltAz(obstime=time_steps[:, None], location=locations[None])
        src_alt = src_crd.transform_to(altaz_frames).alt.degree

        # check which stations can see the source for each time step
        visible = np.logical_and(
            self.array.el_low.numpy() <= src_alt, src_alt <= self.array.el_high.numpy()
        )

        # We want at least half of the telescopes to see the source
        visible_count_per_t = visible.sum(axis=1)
        visible_half = visible_count_per_t > total_stations // 2

        # If the source is not seen by half the telescopes half of the observation time,
        # we redraw the source ra and dec and scan start times, duration,
        # and number of scans. Then we test again by calling this function recursively.
        if visible_half.sum() < time_steps.size // 2:
            redrawn_samp_opts = self.draw_sampling_opts(1)
            self.samp_opts["src_ra"][i] = redrawn_samp_opts["src_ra"][0]
            self.samp_opts["src_dec"][i] = redrawn_samp_opts["src_dec"][0]
            self.samp_opts["start_time"][i] = redrawn_samp_opts["start_time"][0]
            self.samp_opts["scan_duration"][i] = redrawn_samp_opts["scan_duration"][0]
            self.samp_opts["num_scans"][i] = redrawn_samp_opts["num_scans"][0]

            self.test_rand_opts(i)

    def calc_time_steps(self, i: int) -> Time:
        """Calculates time steps for given sampling
        parameter set. Used in testing.

        Parameters
        ----------
        i : int
            Index of the current set of sampling parameters.

        See Also
        --------
        pyvisgen.simulation.data_set.SimulateDataSet.test_rand_opts :
            Tests randomized sampling parameters.
        """
        start_time = Time(self.samp_opts["start_time"][i].isoformat(), format="isot")
        scan_separation = self.samp_opts_const["scan_separation"]
        num_scans = self.samp_opts["num_scans"][i]
        scan_duration = self.samp_opts["scan_duration"][i]
        int_time = self.samp_opts_const["integration_time"]

        time_lst = [
            start_time
            + scan_separation * i * un.second
            + i * scan_duration * un.second
            + j * int_time * un.second
            for i in range(num_scans)
            for j in range(int(scan_duration / int_time) + 1)
        ]
        # +1 because t_1 is the stop time of t_0.
        # In order to save computing power we take
        # one time more to complete interval
        time = Time(time_lst)

        return time

    @classmethod
    def _get_obs_test(
        cls,
        config: str | Path,
        /,
        image_key: str = "y",
        *,
        date_fmt: str = DATEFMT,
    ) -> tuple:  # pragma: no cover
        """Creates an :class:`~pyvisgen.simulation.Observation` class
        object for tests.

        Parameters
        ----------
        config : str or Path
            Path to the config file.
        image_key : str, optional
            Key under which the true sky distributions are saved
            in the HDF5 file. Default: ``'y'``
        date_fmt : str, optional
            Format string for datetime objects.
            Default: ``'%d-%m-%Y %H:%M:%S'``

        Returns
        -------
        self : SimulateDataSet
            Class object.
        obs : :class:`~pyvisgen.simulation.Observation`
            :class:`~pyvisgen.simulation.Observation` class object.
        """
        cls = cls()
        cls.key = image_key
        cls.date_fmt = date_fmt
        cls.multiprocess = 1

        if isinstance(config, (str, Path)):
            cls.conf = read_data_set_conf(config)
        elif isinstance(config, dict):
            cls.conf = config
        else:
            raise ValueError("Expected config to be one of str, Path or dict!")

        cls.data_paths = load_bundles(cls.conf["in_path"])[0]
        cls.create_sampling_rc(1)
        obs = cls.create_observation(0)

        return cls, obs
