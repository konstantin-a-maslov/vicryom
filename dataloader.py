import grain
import numpy as np
import rasterio
import rasterio.windows
import json
import pathlib
import functools
from tqdm import tqdm
import sys


def get_dataloader(
    manifest_path,
    subset,
    sample_size=512,
    mask_patch_size=32,
    masking_rate=0.6,
    batch_size=16,
    num_records=None,
    num_epochs=None,
    worker_count=8,
    worker_buffer_size=1,
    name=None,
    rng_seed=42,
):
    if subset not in {"train", "val"}:
        raise ValueError("Subset must be either train or val")

    manifest_path = pathlib.Path(manifest_path)
    with open(manifest_path, "r") as src:
        manifest = json.load(src)
    tile_info = manifest[subset]
    for item in tile_info:
        item["tile_path"] = manifest_path.parent / item["tile_path"]
        item["index_path"] = manifest_path.parent / item["index_path"]

    if num_records is None:
        num_records = manifest[f"{subset}_count"]

    source = grain.sources.RangeDataSource(0, num_records, 1)
    sampler = grain.samplers.IndexSampler(
        num_records=num_records,
        num_epochs=(num_epochs if subset == "train" else 1),
        shuffle=False,
        seed=rng_seed,
    )
    sample_op = Sample(
        tile_info, 
        sample_size=sample_size, 
        training=(subset == "train"),
        name=name,
    )
    augmentation_op = Augmentation()
    random_mask_op = RandomMask(patch_size=mask_patch_size, rate=masking_rate)

    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            sample_op,
        ] + ([
            augmentation_op,
        ] if subset == "train" else []) + [
            random_mask_op,
            grain.transforms.Batch(
                batch_size=batch_size, 
                drop_remainder=(subset == "train"),
            ),
        ],
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
    )

    if sys.platform == "win32" and worker_count > 0:
        loader._dataset = grain.experimental.WithOptionsIterDataset(
            loader._dataset,
            grain.experimental.DatasetOptions(
                min_shm_size=sys.maxsize,
            ),
        )

    return loader


@functools.lru_cache(maxsize=None)
def load_valid_centres(npy_path):
    return np.load(npy_path, mmap_mode="r")


class Sample(grain.transforms.RandomMap):
    def __init__(self, tile_info, sample_size, training=True, name=None):
        self.name = name
        self.sample_size = sample_size
        self.training = training
        self.tiles = []
        counts = []

        for tile in tqdm(tile_info, desc=f"{f'{self.name}:' if self.name is not None else ''} Counting data..."):
            centres = load_valid_centres(tile["index_path"])
            count = tile["count"]
            if count > 0:
                self.tiles.append(tile)
                counts.append(count)

        weights = np.asarray(counts, dtype=np.float32)
        self.probs = weights / weights.sum()
        self.cumulative_counts = np.cumsum(counts, dtype=np.int64)

    def random_map(self, sample_id, rng):
        if self.training:
            tile_id = rng.choice(len(self.tiles), p=self.probs)
            tile = self.tiles[tile_id]
            centres = load_valid_centres(tile["index_path"])
            row, col = centres[rng.integers(len(centres))]

        else:
            tile_id = np.searchsorted(self.cumulative_counts, sample_id, side="right")
            tile = self.tiles[tile_id]
            offset = (self.cumulative_counts[tile_id - 1] if tile_id > 0 else 0)
            centres = load_valid_centres(tile["index_path"])
            row, col = centres[sample_id - offset]

        with rasterio.open(tile["tile_path"]) as src:
            window = rasterio.windows.Window(
                col - self.sample_size // 2,
                row - self.sample_size // 2,
                self.sample_size,
                self.sample_size,
            )
            sample = src.read(window=window, out_dtype=np.float32)
            sample /= 255.0

        return np.ascontiguousarray(sample)


class Augmentation(grain.transforms.RandomMap):
    def __init__(
        self,
        flip_p=0.5,
        intensity_p=0.8,
        gain_range=(0.9, 1.1),
        brightness_range=(-0.05, 0.05),
        gamma_p=0.5,
        gamma_range=(0.8, 1.25),
    ):
        self.flip_p = flip_p
        self.intensity_p = intensity_p
        self.gain_range = gain_range
        self.brightness_range = brightness_range
        self.gamma_p = gamma_p
        self.gamma_range = gamma_range

    def random_map(self, image, rng):
        # rotation
        k = rng.integers(4)
        if k > 0:
            image = np.rot90(image, k=k, axes=(1, 2))

        # reflection
        if rng.uniform() < self.flip_p:
            image = np.flip(image, axis=1)

        # linear intensity augmentation
        if rng.random() < self.intensity_p:
            gain = np.float32(rng.uniform(*self.gain_range))
            brightness = np.float32(rng.uniform(*self.brightness_range))
            image *= gain
            image += brightness
            np.clip(image, 0.0, 1.0, out=image)

        # gamma transform
        if rng.random() < self.gamma_p:
            log_gamma = rng.uniform(
                np.log(self.gamma_range[0]),
                np.log(self.gamma_range[1]),
            )
            gamma = np.float32(np.exp(log_gamma))
            np.power(image, gamma, out=image)

        return image


class RandomMask(grain.transforms.RandomMap):
    def __init__(self, patch_size, rate):
        self.patch_size = patch_size
        self.rate = rate

    def random_map(self, image, rng):
        _, h, w = image.shape
        mask_size = (1, h // self.patch_size, w // self.patch_size)
        mask = rng.uniform(size=mask_size)
        mask = (mask < self.rate).astype(np.uint8)
        mask = np.repeat(
            np.repeat(mask, self.patch_size, axis=1),
            self.patch_size, axis=2,
        )
        return image, mask
