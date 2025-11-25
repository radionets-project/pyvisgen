from io import BytesIO
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import webdataset as wds

    _WDS_AVAIL = True
except ImportError:
    _WDS_AVAIL = False


class WDSShardWriter:
    def __init__(
        self,
        output_path: str | Path,
        *,
        total_samples: int,
        shard_pattern: str,
        compress: bool = False,
    ):
        if not isinstance(output_path, Path):
            output_path = Path(output_path)

        self.output_path = output_path
        self.total_samples = total_samples
        self.shard_pattern = shard_pattern
        self.compress = compress

        if self.compress and not shard_pattern.endswith(".gz"):
            self.shard_pattern = self.shard_pattern.replace(".tar", ".tar.gz")

        # keeping track of IDs
        self.current_shard_id = 0
        self.total_samples_written = 0
        self.shards_written = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def write_shard(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        bundle_id: int,
        mode: str,
    ) -> None:
        samples_in_shard = inputs.shape[0]

        shard_path = str(
            self.output_path / (self.shard_pattern % self.current_shard_id)
        )

        with wds.TarWriter(shard_path, compress=self.compress) as tarwriter:
            for x, y in zip(inputs, targets):
                sample = {
                    "__key__": f"{mode}_{self.total_samples_written:08d}",
                    "input.npy": self._serialize_numpy(x),
                    "target.npy": self._serialize_numpy(y),
                }

                tarwriter.write(sample)

                self.total_samples_written += 1

            metadata = pd.DataFrame(
                {
                    "total_samples_in_dataset": self.total_samples,
                    "samples_in_shard": samples_in_shard,
                    "shard_idx": self.current_shard_id,
                    "bundle_id": bundle_id,
                },
                index=[0],
            )
            metadata_table = pa.Table.from_pandas(metadata)
            metadata_path = (
                f"{shard_path}".replace(".tar", ".parquet")
                if shard_path.endswith(".tar")
                else f"{shard_path}".replace(".tar.gz", ".parquet")
            )
            pq.write_table(metadata_table, metadata_path)

        self.current_shard_id += 1
        self.shards_written += 1

    def _serialize_numpy(self, array: np.ndarray) -> bytes:
        buffer = BytesIO()
        np.save(buffer, array)

        return buffer.getvalue()
