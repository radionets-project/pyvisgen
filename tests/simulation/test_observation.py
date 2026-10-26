import logging
from dataclasses import fields

import astropy.units as un
import numpy as np
import pytest
import torch
from astropy.coordinates import Angle
from astropy.time import Time

from pyvisgen.simulation.observation import (
    Baselines,
    Observation,
    Scan,
    ValidBaselineSubset,
    create_scans,
)


class TestBaselines:
    def test_get_item(self, baselines: Baselines) -> None:
        idx = 0

        baselines_item = baselines[idx]

        assert isinstance(baselines_item, Baselines)

        for f in fields(Baselines):
            assert getattr(baselines_item, f.name) == getattr(baselines, f.name)[idx]

    def test_get_slice(self, baselines: Baselines) -> None:
        baselines_items = baselines[2:-2]

        assert isinstance(baselines_items, Baselines)
        assert baselines_items.u.shape == torch.Size([6])

    def test_add_baseline(
        self, baselines: Baselines, baselines_data: dict[str, torch.Tensor]
    ) -> None:
        orig_len = baselines.u.shape[0]
        new_baselines = Baselines(**baselines_data)

        baselines.add_baseline(new_baselines)

        assert (
            baselines.u.shape[0] == orig_len + new_baselines.u.shape[0]
        )  # torch.Size([20])

    def test_get_valid_subset(self, baselines: Baselines, device: str) -> None:
        valid = baselines.valid

        subset = baselines.get_valid_subset(len(baselines.time), device=device)

        assert isinstance(subset, ValidBaselineSubset)
        assert len(subset.date) == sum(valid)

    @staticmethod
    def _stacked_scans(scan_lengths: list[int], num_baselines: int) -> Baselines:
        """Baselines of consecutive scans, stacked along the time axis."""
        n_steps = sum(scan_lengths)
        step = torch.arange(n_steps, dtype=torch.float64)

        def per_baseline(values: torch.Tensor) -> torch.Tensor:
            return values[:, None].expand(n_steps, num_baselines).reshape(-1)

        st = torch.arange(num_baselines, dtype=torch.float64).repeat(n_steps)
        zeros = torch.zeros(n_steps * num_baselines, dtype=torch.float64)

        baselines = Baselines(
            st1=st,
            st2=st + 1,
            u=per_baseline(step),
            v=per_baseline(step),
            w=zeros,
            valid=torch.ones(n_steps * num_baselines),
            time=per_baseline(step),
            q1=zeros,
            q2=zeros,
            el1=zeros,
            el2=zeros,
        )
        baselines.scan_lengths = scan_lengths

        return baselines

    def test_get_valid_subset_no_window_across_scans(self, device: str) -> None:
        num_baselines = 3
        baselines = self._stacked_scans([3, 2, 4], num_baselines)

        subset = baselines.get_valid_subset(num_baselines, device=device)

        # rows 0-1-2 | 3-4 | 5-6-7-8: windows (0,1) (1,2) | (3,4) | (5,6) (6,7) (7,8)
        assert len(subset.u_valid) == 6 * num_baselines
        starts = torch.unique(subset.u_start).tolist()
        stops = torch.unique(subset.u_stop).tolist()
        assert starts == [0.0, 1.0, 3.0, 5.0, 6.0, 7.0]
        assert stops == [1.0, 2.0, 4.0, 6.0, 7.0, 8.0]

    def test_get_valid_subset_without_scan_lengths(self, device: str) -> None:
        """Without the scan lengths, all consecutive time steps are paired."""
        num_baselines = 3
        baselines = self._stacked_scans([3, 2], num_baselines)
        del baselines.scan_lengths

        subset = baselines.get_valid_subset(num_baselines, device=device)

        assert len(subset.u_valid) == 4 * num_baselines

    def test_get_valid_subset_single_scan(self, device: str) -> None:
        num_baselines = 3
        baselines = self._stacked_scans([5], num_baselines)

        subset = baselines.get_valid_subset(num_baselines, device=device)

        assert len(subset.u_valid) == 4 * num_baselines

    def test_add_baseline_to_empty(self, baselines: Baselines) -> None:
        empty = Baselines(
            st1=torch.tensor([]),
            st2=torch.tensor([]),
            u=torch.tensor([]),
            v=torch.tensor([]),
            w=torch.tensor([]),
            valid=torch.tensor([]),
            time=torch.tensor([]),
            q1=torch.tensor([]),
            q2=torch.tensor([]),
            el1=torch.tensor([]),
            el2=torch.tensor([]),
        )

        empty.add_baseline(baselines)

        assert empty.u.shape == baselines.u.shape


class TestValidBaselineSubset:
    def test_get_item(self, subset: ValidBaselineSubset) -> None:
        idx = 0

        subset_item = subset[idx]

        assert isinstance(subset_item, ValidBaselineSubset)

        for f in fields(ValidBaselineSubset):
            assert (getattr(subset_item, f.name) == getattr(subset, f.name)[idx]).all()

    def test_get_slice(self, subset: ValidBaselineSubset) -> None:
        subset_items = subset[2:-2]

        assert isinstance(subset_items, ValidBaselineSubset)
        assert subset_items.u_start.shape == torch.Size([6])

    def test_get_timerange(self, subset: ValidBaselineSubset, subset_data) -> None:
        dates = subset_data["date"]

        subset_items = subset.get_timerange(t_start=dates[2], t_stop=dates[-3])

        assert isinstance(subset_items, ValidBaselineSubset)
        assert subset_items.u_start.shape == torch.Size([6])
        np.testing.assert_array_equal(
            subset_items.date.detach().cpu(), dates[2:-2].detach().cpu()
        )

    @pytest.mark.parametrize(
        "fov,freq,img_size,expected",
        [
            (0.1, 15e9, 32, 8),
            (0.1, 15e9, 64, 10),
            (3600, 1.4e9, 32, 2),
            (0.1, 15e9, 35, 8),  # also test for odd image size
            (1e-8, 1.4e9, 32, 1),  # very small fov
        ],
    )
    def test_get_unique_grid(
        self,
        fov: float,
        freq: float,
        img_size: int,
        expected: int,
        subset_data: dict[str, torch.Tensor],
        device: str,
    ) -> None:
        size = 10
        dev = torch.device(device)

        subset_data["u_valid"] = torch.linspace(-1e6, 1e6, size, device=dev)
        subset_data["v_valid"] = torch.linspace(-1e6, 1e6, size, device=dev)

        subset = ValidBaselineSubset(**subset_data)

        unique = subset.get_unique_grid(
            fov=fov, ref_frequency=freq, img_size=img_size, device=device
        )

        assert isinstance(unique, ValidBaselineSubset)
        assert unique.u_valid.shape == torch.Size([expected])

    def test_lexsort(self, subset: ValidBaselineSubset):
        vals = torch.rand(4)
        arr = vals[(torch.rand(3, 9) * 4).long()]

        expected = np.lexsort(arr.detach().cpu().numpy())
        result = subset._lexsort(arr)

        np.testing.assert_array_equal(result, expected)


class TestScan:
    def test_get_num_timesteps(self, scan: Scan) -> None:
        assert scan.get_num_timesteps() == 181

    def test_get_timesteps(self, scan: Scan) -> None:
        timesteps = scan.get_timesteps()

        assert isinstance(timesteps, Time)
        assert len(timesteps) == scan.get_num_timesteps()
        assert timesteps[0] == scan.start
        assert timesteps[-1] == scan.stop

    @pytest.mark.parametrize(
        "duration,integration_time,expected",
        [
            (360, 60, 7),  # divisible, the endpoint is included
            (400, 60, 8),  # not divisible, short last interval
            (3, 0.1, 31),  # floating point integration time
            (60, 60, 2),
            (30, 60, 2),  # shorter than one integration time
        ],
    )
    def test_timesteps_cover_whole_scan(
        self, duration: float, integration_time: float, expected: int
    ) -> None:
        start = Time("2026-01-21T00:00:00", format="isot", scale="utc")
        stop = start + duration * un.second
        scan = Scan(
            start=start, stop=stop, separation=0, integration_time=integration_time
        )

        timesteps = scan.get_timesteps()
        steps = (timesteps - start).to(un.second).value

        assert scan.get_num_timesteps() == expected
        assert len(timesteps) == expected
        assert timesteps[0] == start
        assert timesteps[-1] == stop
        # no duplicates, and no step is longer than the integration time
        assert (np.diff(steps) > 0).all()
        assert (np.diff(steps) <= integration_time + 1e-6).all()

    def test_post_init_cast_to_astropy_quantity(self):
        start = Time("2026-01-21T00:00:00", format="isot", scale="utc")
        stop = Time("2026-01-21T01:00:00", format="isot", scale="utc")

        scan = Scan(start=start, stop=stop, separation=10, integration_time=10)

        assert isinstance(scan.separation, un.Quantity)
        assert isinstance(scan.integration_time, un.Quantity)
        assert scan.separation.unit == un.second
        assert scan.integration_time.unit == un.second

    def test_post_init_unit_conversion(self):
        start = Time("2026-01-21T00:00:00", format="isot", scale="utc")
        stop = Time("2026-01-21T01:00:00", format="isot", scale="utc")

        scan = Scan(
            start=start,
            stop=stop,
            separation=10 * un.minute,
            integration_time=10 * un.minute,
        )

        assert scan.separation.unit == un.second
        assert scan.integration_time.unit == un.second
        assert scan.separation == 600 * un.second
        assert scan.integration_time == 600 * un.second


class TestCreateScans:
    @pytest.fixture
    def start(self) -> Time:
        return Time("2026-01-21T00:00:00", format="isot", scale="utc")

    def test_scan_starts_after_previous_scan_and_separation(self, start: Time) -> None:
        scans = create_scans(
            start=start,
            scan_duration=400,
            num_scans=3,
            scan_separation=120,
            integration_time=60,
        )

        starts = [(scan.start - start).to(un.second).value for scan in scans]
        stops = [(scan.stop - start).to(un.second).value for scan in scans]

        assert len(scans) == 3
        np.testing.assert_allclose(starts, [0, 520, 1040], atol=1e-5)
        np.testing.assert_allclose(stops, [400, 920, 1440], atol=1e-5)
        # no waiting time after the last scan
        assert scans[-1].separation == 0 * un.second

    def test_per_scan_arrays(self, start: Time) -> None:
        scans = create_scans(
            start=start,
            scan_duration=[100, 200],
            num_scans=2,
            scan_separation=[50],
            integration_time=[10, 20],
        )

        assert [scan.integration_time.value for scan in scans] == [10, 20]
        np.testing.assert_allclose(
            [(scan.start - start).to(un.second).value for scan in scans],
            [0, 150],
            atol=1e-5,
        )

    def test_observation_uses_create_scans(self, obs_params: dict) -> None:
        obs = Observation(**{**obs_params, "num_scans": 3, "scan_separation": 120})

        expected = create_scans(
            start=obs.start,
            scan_duration=obs.scan_duration,
            num_scans=obs.num_scans,
            scan_separation=obs.scan_separation,
            integration_time=obs.int_time,
        )

        assert len(obs.scans) == len(expected)
        for scan, scan_expected in zip(obs.scans, expected):
            assert scan.start == scan_expected.start
            assert scan.stop == scan_expected.stop
            assert scan.integration_time == scan_expected.integration_time


class TestObservation:
    def test_init(self, obs: Observation, obs_params: dict) -> None:
        assert obs.ra == obs_params["src_ra"]
        assert obs.dec == obs_params["src_dec"]
        assert obs.ra.dtype == torch.double
        assert obs.dec.dtype == torch.double

        assert obs.num_scans == obs_params["num_scans"]
        assert len(obs.scans) == obs_params["num_scans"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_device_cuda(self, obs_params: dict) -> None:
        obs_params["device"] = "cuda"
        obs = Observation(**obs_params)

        assert obs.rd.device.type == "cuda"
        assert obs.lm.device.type == "cuda"

        subset = obs.baselines.get_valid_subset(obs.num_baselines, "cuda")

        assert subset.u_valid.device.type == "cuda"
        assert subset.v_valid.device.type == "cuda"

    def test_create_scans(self, obs_params: dict) -> None:
        obs_params["num_scans"] = 2
        obs_params["scan_duration"] = [400, 600]
        obs_params["scan_separation"] = [60]
        obs_params["integration_time"] = [30, 60]
        obs = Observation(**obs_params)

        assert len(obs.scans) == 2

    def test_valid_subset_stays_within_scans(self, obs_params: dict) -> None:
        """No uv point may be created in the gap between two scans."""
        # the source has to be visible, otherwise there are no valid baselines
        params = {
            **obs_params,
            "src_dec": -35.0,
            "array_layout": "meerkat",
            "ref_frequency": 1.284e9,
            "num_scans": 3,
            "scan_duration": 400,
            "scan_separation": 120,
            "integration_time": 60,
            "device": "cpu",
        }
        obs = Observation(**params)

        assert obs.baselines.scan_lengths == [
            sc.get_num_timesteps() for sc in obs.scans
        ]

        subset = obs.baselines.get_valid_subset(obs.num_baselines, "cpu")
        date = subset.date.numpy()
        in_scan = np.zeros(date.shape, dtype=bool)

        for scan in obs.scans:
            in_scan |= (date >= scan.start.jd) & (date <= scan.stop.jd)

        assert subset.u_valid.numel() > 0
        assert in_scan.all()

    def test_create_lm_grid(self, obs: Observation, device) -> None:
        img_size = obs.img_size

        lm = obs.create_lm_grid()
        dev = torch.device(device)

        assert lm.shape == (img_size, img_size, 2)
        assert lm.device.type == dev.type

    @pytest.mark.parametrize(
        "fov_deg,warns",
        [
            (1.0, False),
            (80.0, False),  # sqrt(2) rad = 81.05 deg
            (82.0, True),
            (120.0, True),
        ],
    )
    def test_create_lm_grid_warns_for_too_large_fov(
        self, obs_params: dict, caplog, fov_deg: float, warns: bool
    ) -> None:
        params = {**obs_params, "fov": fov_deg * 3600, "device": "cpu"}

        with caplog.at_level(logging.WARNING):
            obs = Observation(**params)

        l_m = obs.lm.numpy()
        outside = (l_m[..., 0] ** 2 + l_m[..., 1] ** 2 > 1).any()

        assert ("too large for the SIN projection" in caplog.text) == warns
        assert outside == warns

    def test_create_rd_grid(self, obs: Observation, device) -> None:
        img_size = obs.img_size

        rd = obs.create_rd_grid()
        dev = torch.device(device)

        assert rd.shape == (img_size, img_size, 2)
        assert rd.device.type == dev.type

    def test_calc_direction_cosines(self, obs: Observation) -> None:
        ha = torch.rand(1)
        el_st = torch.rand(10)
        delta_x = torch.rand((1, 10))
        delta_y = torch.rand((1, 10))
        delta_z = torch.rand((1, 10))

        u, v, w = obs.calc_direction_cosines(ha, el_st, delta_x, delta_y, delta_z)

        assert u.shape == v.shape == w.shape
        assert u.shape[0] == 10

    def test_calc_ref_elev(self, obs: Observation) -> None:
        times = ["1999-01-01T10:00:00", "2010-12-02T18:00:00", "2026-01-21T12:00:00"]

        expected_gha = Angle(
            [
                "70:36:53 degrees",
                "161:28:22 degrees",
                "120:52:6 degrees",
            ]
        )

        gha, ha_local, el_st = obs.calc_ref_elev(Time(times))

        assert gha.shape[0] == ha_local.shape[0] == el_st.shape[0] == len(times)
        np.testing.assert_allclose(gha, expected_gha.value, rtol=1e-2)

        assert (
            ha_local.shape
            == el_st.shape
            == torch.Size([len(times), len(obs.array_earth_loc)])
        )

    def test_calc_ref_elev_scalar_time(self, obs: Observation) -> None:
        time = obs.scans[0].start

        gha, ha_local, el_st = obs.calc_ref_elev(time)

        assert gha.shape[0] == ha_local.shape[0] == el_st.shape[0] == 1

    def test_calc_feed_rotation(self, mocker, obs: Observation) -> None:
        ha = torch.tensor([-30, 0, 30])

        mock_array_loc = mocker.MagicMock()
        mock_array_loc.lat = 45

        obs.array_earth_loc = mock_array_loc
        obs.dec = torch.tensor([45])

        q = obs.calc_feed_rotation(torch.deg2rad(ha))

        np.testing.assert_allclose(
            torch.rad2deg(q), [-79.2714, 0.0, 79.2714], rtol=1e-5
        )

    def test_get_baselines2(self, obs_params: dict) -> None:
        obs = Observation(**obs_params)
        times = obs.scans[0].get_timesteps()

        baselines = obs.get_baselines(times)

        assert isinstance(baselines, Baselines)
        assert baselines.u.shape[0] > 0

    def test_get_baselines(self, mocker, obs_params: dict) -> None:
        obs = Observation(**obs_params)
        times = obs.scans[0].get_timesteps()

        mock_calc_rev_elev = mocker.patch.object(
            obs,
            "calc_ref_elev",
            return_value=(
                torch.rand(len(times)),
                torch.rand((len(times), len(obs.array_earth_loc))),
                torch.rand((len(times), len(obs.array_earth_loc))),
            ),
        )

        mock_calc_feed_rotation = mocker.patch.object(
            obs,
            "calc_feed_rotation",
            return_value=torch.rand((len(times), len(obs.array_earth_loc))),
        )

        mock_uvw = torch.ones(obs.num_baselines)
        mock_calc_direction_cosines = mocker.patch.object(
            obs, "calc_direction_cosines", return_value=(mock_uvw, mock_uvw, mock_uvw)
        )

        baselines = obs.get_baselines(times)

        assert isinstance(baselines, Baselines)
        assert mock_calc_rev_elev.called
        assert mock_calc_feed_rotation.called
        assert mock_calc_direction_cosines.called
        assert baselines.u.shape[0] == len(times) * obs.num_baselines
        np.testing.assert_array_equal(baselines.u[0], mock_uvw)

    def test_get_baselines_scalar_time(self, obs_params: dict) -> None:
        obs = Observation(**obs_params)
        time = obs.scans[0].start

        baselines = obs.get_baselines(time)

        assert isinstance(baselines, Baselines)
        assert baselines.u.shape[0] == obs.num_baselines

    def test_calc_baselines(self, obs_params: dict) -> None:
        obs = Observation(**obs_params)

        assert hasattr(obs, "baselines")
        assert hasattr(obs.baselines, "num")
        assert hasattr(obs.baselines, "times_unique")
        assert hasattr(obs.baselines, "u")
        assert obs.baselines.u.shape[0] > 0

    def test_frequencies(self, obs_params: dict) -> None:
        obs = Observation(**obs_params)

        assert obs.ref_frequency == obs_params["ref_frequency"]
        assert obs.waves_low.shape == obs.waves_high.shape
        assert torch.all(obs.waves_high <= obs.waves_high)

    def test_multiple_frequency_offsets(self, obs_params: dict) -> None:
        obs_params["frequency_offsets"] = [0.0, 1.28e8, 2.56e8]
        obs_params["bandwidths"] = [1.28e8, 1.28e8, 1.28e8]

        obs = Observation(**obs_params)

        assert obs.waves_low.shape[0] == obs.waves_high.shape[0] == 3

    def test_single_scan(self, obs_params: dict) -> None:
        obs_params["num_scans"] = 1
        obs = Observation(**obs_params)

        assert len(obs.scans) == 1

    def test_polarization_attr(self, obs_params: dict) -> None:
        obs_params["polarization"] = "linear"
        obs_params["pol_kwargs"] = {"delta": 45, "amp_ratio": 0.5, "random_state": 42}
        obs_params["field_kwargs"] = {
            "order": [0.1, 0.1],
            "scale": [0, 1],
            "threshold": None,
            "random_state": 42,
        }

        obs = Observation(**obs_params)

        assert obs.polarization == obs_params["polarization"]
        assert obs.pol_kwargs == obs_params["pol_kwargs"]
        assert obs.field_kwargs == obs_params["field_kwargs"]

    def test_dense(self, obs_params: dict, device: str) -> None:
        obs_params["dense"] = True
        obs = Observation(**obs_params)

        assert isinstance(obs.waves_low, list)
        assert isinstance(obs.waves_high, list)
        assert hasattr(obs, "dense_baselines_gpu")
        assert isinstance(obs.dense_baselines_gpu, ValidBaselineSubset)
        assert obs.ra.device.type == obs.dec.device.type == device
