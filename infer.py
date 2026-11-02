import jax
import equinox
from model.vicryom import ViCryoMTiny
import numpy as np
import rioxarray
import rasterio
import affine
import grain
import pathlib


def run(cfg):
    model = get_model(cfg)
    dataloader, scene = get_dataloader(cfg)
    batch_predict = jax.jit(jax.vmap(get_predict_fn(model)))
    predictions = run_inference(dataloader, scene, batch_predict)
    
    dst_folder = pathlib.Path(cfg.output_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)
    h, w = scene["shape"]

    for i, pred in enumerate(predictions[::-1]):
        c, ph, pw = pred.shape
        transform = scene["transform"] * affine.Affine.scale(w / pw, h / ph)
        dst_path = dst_folder / f"embds_level{i}.tiff"

        with rasterio.open(
            dst_path,
            "w",
            driver="COG",
            height=ph,
            width=pw,
            count=c,
            dtype="float32",
            crs=scene["crs"],
            transform=transform,
            BLOCKSIZE=512,
            COMPRESS="DEFLATE",
            PREDICTOR="FLOATING_POINT",
            OVERVIEW_RESAMPLING="AVERAGE",
            BIGTIFF="IF_SAFER",
            NUM_THREADS="ALL_CPUS",
        ) as dst:
            dst.write(pred)
    


def get_model(cfg):
    model = ViCryoMTiny(1, 1, jax.random.key(42))
    model = equinox.tree_deserialise_leaves(cfg.checkpoint, model)
    return model


def get_dataloader(cfg):
    da = rioxarray.open_rasterio(cfg.input) / 255.0
    da, padding = pad_for_inference(da, patch_size=cfg.patch_size)

    image = da.values.astype(np.float32)
    _, h, w = image.shape

    p = cfg.patch_size
    step = p // 2

    rows = list(range(0, h - p + 1, step))
    cols = list(range(0, w - p + 1, step))

    def get_patch(idx):
        row = rows[idx // len(cols)]
        col = cols[idx % len(cols)]

        patch = image[:, row:row + p, col:col + p]

        return {
            "image": np.nan_to_num(patch),
            "row": np.int32(row),
            "col": np.int32(col),
        }

    loader = (
        grain.MapDataset.range(len(rows) * len(cols))
            .map(get_patch)
            .batch(cfg.batch_size, drop_remainder=False)
            .to_iter_dataset(
                read_options=grain.ReadOptions(num_threads=0)
            )
    )

    loader = grain.experimental.ThreadPrefetchIterDataset(
        loader,
        prefetch_buffer_size=True,
    )

    scene = {
        "shape": (h, w),
        "transform": da.rio.transform(),
        "crs": da.rio.crs,
        "padding": padding,
    }

    return loader, scene


def run_inference(loader, scene, predict_fn, sigma_scale=0.25):
    h, w = scene["shape"]

    preds_sums, weights_sums, kernels, patch_size = [], [], [], None

    for batch in loader:
        preds = predict_fn(batch["image"])

        if patch_size is None:
            patch_size = preds[-1].shape[-1]

            for pred in preds:
                # preds_sum init
                _, c, p, _ = pred.shape
                sh, sw = patch_size // p, patch_size // p
                preds_sum = np.zeros((c, h // sh, w // sw), dtype=np.float32)
                preds_sums.append(preds_sum)
                # weights_sum init
                weights_sum = np.zeros((h // sh, w // sw), dtype=np.float32)
                weights_sums.append(weights_sum)
                # kernel init
                sigma = p * sigma_scale
                coords = np.arange(p) - (p - 1) / 2
                gaussian = np.exp(-0.5 * (coords / sigma) ** 2)
                kernel = np.outer(gaussian, gaussian).astype(np.float32)
                kernels.append(kernel)

        for i in range(len(preds)):
            p = preds[i].shape[-1]
            sh, sw = patch_size // p, patch_size // p

            for pred, row, col in zip(preds[i], batch["row"], batch["col"]):
                r, c = int(row) // sh, int(col) // sw
                preds_sums[i][:, r:r + p, c:c + p] += pred * kernels[i][None]
                weights_sums[i][r:r + p, c:c + p] += kernels[i]

    results = [
        preds_sum / weights_sum[None]
        for preds_sum, weights_sum in zip(preds_sums, weights_sums)
    ]

    return results


def get_predict_fn(model):
    def predict(image):
        embds, _ = model(image)
        return embds
    return predict


def pad_for_inference(da, patch_size, min_border=0):
    h, w = da.rio.shape

    extra_x = (-(w + 2 * min_border)) % patch_size
    extra_y = (-(h + 2 * min_border)) % patch_size

    pad_left = min_border + extra_x // 2
    pad_right = min_border + extra_x - extra_x // 2
    pad_top = min_border + extra_y // 2
    pad_bottom = min_border + extra_y - extra_y // 2

    res_x, res_y = da.rio.resolution()
    left, bottom, right, top = da.rio.bounds()

    padded = da.rio.pad_box(
        minx=left - pad_left * res_x,
        miny=bottom - pad_bottom * abs(res_y),
        maxx=right + pad_right * res_x,
        maxy=top + pad_top * abs(res_y),
        constant_values=0.0,
    )

    padding = {
        "left": pad_left,
        "right": pad_right,
        "top": pad_top,
        "bottom": pad_bottom,
    }

    return padded, padding
