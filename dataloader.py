import grain
import numpy as np
import rasterio
import rasterio.windows
import json
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
    worker_count=8,
    worker_buffer_size=1,
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

    if sys.platform == "win32" and worker_count > 0:
        loader._dataset = grain.experimental.WithOptionsIterDataset(
            loader._dataset,
            grain.experimental.DatasetOptions(
                min_shm_size=sys.maxsize,
            ),
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
    # import gc

    loader = get_dataloader("data/manifest.json", "train", name="Train dataloader")

    for i, batch in enumerate(loader):
        
        _, axs = plt.subplots(nrows=2, ncols=16, figsize=(16, 3))

        for j, (image, mask) in enumerate(zip(*batch)):
            axs[0][j].imshow(image[0], cmap="gray")
            axs[1][j].imshow(mask[0], cmap="cividis")

        for ax in axs.flatten():
            ax.axis("off")
    
        plt.tight_layout()
        plt.show()

        if i >= 8:
            break

    # del loader
    # gc.collect()


# confirmed ~linear scaling with worker_count, and worker_buffer_size=1 is good
# import gc
# import time


# def benchmark_loader(
#     manifest_path="data/manifest.json",
#     subset="train",
#     worker_counts=(0, 1, 2, 4, 8),
#     worker_buffer_size=2,
#     batch_size=16,
#     warmup_batches=16,
#     benchmark_batches=1024,
# ):
#     results = []

#     for worker_count in worker_counts:
#         print(f"\n{'=' * 60}")
#         print(
#             f"worker_count={worker_count}, "
#             f"worker_buffer_size={worker_buffer_size}"
#         )
#         print(f"{'=' * 60}")

#         loader = get_dataloader(
#             manifest_path,
#             subset,
#             batch_size=batch_size,
#             worker_count=worker_count,
#             worker_buffer_size=worker_buffer_size,
#             name=f"Workers={worker_count}",
#         )

#         iterator = iter(loader)

#         # Warm-up:
#         # - worker process startup
#         # - imports in spawned workers
#         # - Rasterio/GDAL initialization
#         # - initial queue filling
#         print(f"Warming up ({warmup_batches} batches)...")
#         for _ in range(warmup_batches):
#             next(iterator)

#         print(f"Benchmarking ({benchmark_batches} batches)...")

#         start = time.perf_counter()

#         for _ in range(benchmark_batches):
#             batch = next(iterator)

#         elapsed = time.perf_counter() - start

#         batches_per_second = benchmark_batches / elapsed
#         samples_per_second = (
#             benchmark_batches * batch_size / elapsed
#         )
#         ms_per_batch = elapsed / benchmark_batches * 1000

#         print(f"Elapsed:       {elapsed:8.2f} s")
#         print(f"Batch time:    {ms_per_batch:8.2f} ms")
#         print(f"Throughput:    {batches_per_second:8.2f} batches/s")
#         print(f"               {samples_per_second:8.2f} samples/s")

#         results.append(
#             {
#                 "worker_count": worker_count,
#                 "worker_buffer_size": worker_buffer_size,
#                 "elapsed_s": elapsed,
#                 "ms_per_batch": ms_per_batch,
#                 "batches_per_s": batches_per_second,
#                 "samples_per_s": samples_per_second,
#             }
#         )

#         del batch
#         del iterator
#         del loader
#         gc.collect()

#     print("\n\nRESULTS")
#     print(
#         f"{'workers':>8} "
#         f"{'seconds':>10} "
#         f"{'ms/batch':>12} "
#         f"{'batch/s':>12} "
#         f"{'samples/s':>12}"
#     )

#     for r in results:
#         print(
#             f"{r['worker_count']:>8} "
#             f"{r['elapsed_s']:>10.2f} "
#             f"{r['ms_per_batch']:>12.2f} "
#             f"{r['batches_per_s']:>12.2f} "
#             f"{r['samples_per_s']:>12.2f}"
#         )

#     return results


# if __name__ == "__main__":
#     # results = benchmark_loader(
#     #     worker_counts=(0, 1, 2, 4, 8),
#     #     worker_buffer_size=2,
#     #     batch_size=16,
#     #     warmup_batches=16,
#     #     benchmark_batches=1024,
#     # )
#     for buffer_size in (1, 2, 4, 8):
#         benchmark_loader(
#             worker_counts=(8,),
#             worker_buffer_size=buffer_size,
#             benchmark_batches=1024,
#         )
    