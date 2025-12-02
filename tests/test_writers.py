from pathlib import Path

import numpy as np

from pyvisgen.io import PTWriter, WDSShardWriter


OUTPUT_PATH = Path("./tests/build/wds")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


OUTPUT_PATH_PT = Path("./tests/build/pt")
OUTPUT_PATH_PT.mkdir(parents=True)

NUM_BUNDLES = 5
SAMP_PER_BUNDLE = 10
TOTAL_SAMPLES = NUM_BUNDLES * SAMP_PER_BUNDLE


def test_wds_writer_amp_phase():
    rng = np.random.default_rng(42)

    with WDSShardWriter(
        output_path=OUTPUT_PATH,
        dataset_type="train",
        total_samples=TOTAL_SAMPLES,
        shard_pattern="train-%06d.tar",
        amp_phase=True,
    ) as writer:
        x_data = rng.uniform(size=(NUM_BUNDLES, SAMP_PER_BUNDLE, 2, 256, 256))
        y_data = rng.uniform(size=(NUM_BUNDLES, SAMP_PER_BUNDLE, 2, 256, 256))

        for bundle_id, (x, y) in enumerate(zip(x_data, y_data)):
            writer.write(x, y, index=bundle_id, overlap=5)


def test_wds_writer_real_imag():
    rng = np.random.default_rng(42)

    with WDSShardWriter(
        output_path=OUTPUT_PATH,
        dataset_type="train",
        total_samples=TOTAL_SAMPLES,
        shard_pattern="train-%06d.tar",
        amp_phase=False,
    ) as writer:
        x_data = rng.uniform(size=(NUM_BUNDLES, SAMP_PER_BUNDLE, 2, 256, 256))
        y_data = rng.uniform(size=(NUM_BUNDLES, SAMP_PER_BUNDLE, 2, 256, 256))

        for bundle_id, (x, y) in enumerate(zip(x_data, y_data)):
            writer.write(x, y, index=bundle_id, overlap=5)


def test_wds_writer_compress():
    rng = np.random.default_rng(42)

    with WDSShardWriter(
        output_path=OUTPUT_PATH,
        dataset_type="train",
        total_samples=TOTAL_SAMPLES,
        shard_pattern="train-%06d.tar",
        compress=True,
        amp_phase=True,
    ) as writer:
        x_data = rng.uniform(size=(NUM_BUNDLES, SAMP_PER_BUNDLE, 2, 256, 256))
        y_data = rng.uniform(size=(NUM_BUNDLES, SAMP_PER_BUNDLE, 2, 256, 256))

        for bundle_id, (x, y) in enumerate(zip(x_data, y_data)):
            writer.write(x, y, index=bundle_id, overlap=5)


def test_pt_writer_amp_phase():
    rng = np.random.default_rng(42)

    with PTWriter(
        output_path=OUTPUT_PATH_PT, dataset_type="train", amp_phase=True
    ) as writer:
        x_data = rng.uniform(size=(5, 10, 2, 256, 256))
        y_data = rng.uniform(size=(5, 10, 2, 256, 256))

        for bundle_id, (x, y) in enumerate(zip(x_data, y_data)):
            writer.write(x, y, index=bundle_id, bundle_length=len(x_data))


def test_pt_writer_real_imag():
    rng = np.random.default_rng(42)

    with PTWriter(
        output_path=OUTPUT_PATH_PT, dataset_type="train", amp_phase=False
    ) as writer:
        x_data = rng.uniform(size=(5, 10, 2, 256, 256))
        y_data = rng.uniform(size=(5, 10, 2, 256, 256))

        for bundle_id, (x, y) in enumerate(zip(x_data, y_data)):
            writer.write(x, y, index=bundle_id, bundle_length=len(x_data))
