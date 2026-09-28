import grain
import numpy as np
import rasterio
import rasterio.windows
import json
import functools
from tqdm import tqdm


def get_dataloader(
    manifest_path,
    subset,
    sample_size=512,
    mask_patch_size=32,
    masking_rate=0.6,
    batch_size=16,
    worker_count=8,
    worker_buffer_size=2,
    name=None,
    rng_seed=42,
):
    with open(manifest_path, "r") as src:
        manifest = json.load(src)
    tile_info = manifest[subset]

    source = grain.sources.RangeDataSource(0, len(tile_info), 1)
    sampler = grain.samplers.IndexSampler(
        num_records=len(tile_info),
        num_epochs=None,
        shuffle=False,
        seed=rng_seed,
    )
    random_sample = RandomSample(tile_info, sample_size=sample_size, name=name)
    augmentation = Augmentation()
    random_mask = RandomMask(patch_size=mask_patch_size, rate=masking_rate)

    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            random_sample,
            augmentation,
            random_mask,
            grain.transforms.Batch(batch_size=batch_size, drop_remainder=True),
        ],
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
    )

    return loader


@functools.lru_cache(maxsize=64)
def load_valid_centres(npy_path):
    return np.load(npy_path, mmap_mode="r")


class RandomSample(grain.transforms.RandomMap):
    def __init__(self, tile_info, sample_size, name=None):
        self.name = name
        self.sample_size = sample_size
        self.tiles = []
        counts = []

        for tile in tqdm(tile_info, desc=f"{f'{self.name}:' if self.name is not None else ''} Counting data..."):
            centres = load_valid_centres(tile["index_path"])
            count = len(centres)
            if count > 0:
                self.tiles.append(tile)
                counts.append(count)

        weights = np.asarray(counts, dtype=np.float32)
        self.probs = weights / weights.sum()

    def random_map(self, sample_id, rng):
        tile_id = rng.choice(len(self.tiles), p=self.probs)
        tile = self.tiles[tile_id]

        centres = load_valid_centres(tile["index_path"])
        row, col = centres[rng.integers(len(centres))]

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
        if rng.uniform() < 0.5:
            image = np.flip(image, axis=1)

        # linear intensity augmentation
        if rng.random() < self.flip_p:
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


########################### TO REMOVE
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import gc

    loader = get_dataloader("data/manifest.json", "train", name="Train dataloader")

    _, axs = plt.subplots(nrows=2, ncols=16, figsize=(16, 3))

    for i, batch in enumerate(loader):
        if i < 32:
            continue
        
        for i, (image, mask) in enumerate(zip(*batch)):
            axs[0][i].imshow(image[0], cmap="gray")
            axs[1][i].imshow(mask[0], cmap="cividis")
        break

    del loader
    gc.collect()

    for ax in axs.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    