import jax
import jax.numpy as jnp
import equinox
import dataloader
from model.vicryom import ViCryoMTiny
import optax
import numpy as np
from logger import CSVLogger
import datetime
from tqdm import tqdm


def run(cfg):
    train_data, val_data = get_dataloaders(cfg)
    model = get_model(cfg)
    optimiser = get_optimiser(cfg)
    opt_state = optimiser.init(equinox.filter(model, equinox.is_inexact_array))

    steps_per_epoch = cfg.num_records // cfg.batch_size
    train_iter = iter(train_data)
    best_loss, best_epoch, logger = np.inf, None, None

    def compute_loss(model, images, masks):
        loss = loss_fn(model, images, masks, embd_lambda=cfg.embd_lambda, remat=cfg.remat_model)
        return loss

    loss_and_grad = equinox.filter_value_and_grad(compute_loss)
        
    @equinox.filter_jit
    def train_step(model, opt_state, images, masks, running_loss):
        loss, grads = loss_and_grad(model, images, masks)
        params = equinox.filter(model, equinox.is_inexact_array)
        updates, opt_state = optimiser.update(grads, opt_state, params)
        model = equinox.apply_updates(model, updates)
        return model, opt_state, running_loss + loss

    @equinox.filter_jit
    def val_step(model, images, masks):
        loss = compute_loss(model, images, masks)
        return loss

    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}")

        # Train loop
        running_loss = jnp.zeros((), dtype=jnp.float32)
        with tqdm(total=steps_per_epoch, desc="Train: ") as pbar:
            for step in range(steps_per_epoch):
                images, masks = next(train_iter)
                model, opt_state, running_loss = train_step(model, opt_state, images, masks, running_loss)
                pbar.update(1)
        train_loss = float(jax.device_get(running_loss / steps_per_epoch))

        # Val check
        print("Val:", end=' ')
        val_loss, num_val_batches = 0.0, 0
        for images, masks in val_data:
            val_loss += val_step(model, images, masks)
            num_val_batches += 1

        val_loss /= num_val_batches
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            equinox.tree_serialise_leaves(cfg.checkpoint, model)

        log_record = dict(
            timestamp=str(datetime.datetime.now()),
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        if logger is None:
            logger = CSVLogger("logs/test_log.csv", log_record.keys())
        logger.log(log_record)

        print(f"val_loss={val_loss:.4f}, best_val_loss={best_loss:.4f} at Epoch {best_epoch}, train_loss={train_loss:.4f}")


def get_dataloaders(cfg):
    train_data = dataloader.get_dataloader(
        cfg.manifest,
        subset="train",
        sample_size=cfg.patch_size,
        num_records=cfg.num_records,
        batch_size=cfg.batch_size,
        name="Train dataloader",
        worker_count=cfg.worker_count,
        worker_buffer_size=cfg.worker_buffer_size,
    )
    val_data = dataloader.get_dataloader(
        cfg.manifest,
        subset="val",
        sample_size=cfg.patch_size,
        batch_size=cfg.batch_size,
        name="Val dataloader",
        worker_count=cfg.worker_count,
        worker_buffer_size=cfg.worker_buffer_size,
    )
    return train_data, val_data


def get_model(cfg):
    model = ViCryoMTiny(1, 1, jax.random.key(42))
    return model


def get_optimiser(cfg):
    total_steps = cfg.num_records // cfg.batch_size * cfg.num_epochs
    lr_schedule = optax.cosine_decay_schedule(
        init_value=cfg.learning_rate,
        decay_steps=total_steps,
        alpha=0.0,
    )
    opt_chain = [
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    ]
    optimiser = optax.chain(*opt_chain)
    return optimiser


def loss_fn(model, images, masks, embd_lambda, remat=False):
    def forward(model, image, mask):
        return model(x=image, mask=mask)

    if remat:
        forward = equinox.filter_checkpoint(forward)

    def single_loss(image, mask):
        embds, preds = forward(model, image, mask)

        image_rot = jnp.rot90(image, k=1, axes=(-2, -1))
        mask_rot = jnp.rot90(mask, k=1, axes=(-2, -1))
        embds_rot, _ = forward(model, image_rot, mask_rot)

        image_flip = jnp.flip(image, axis=-1)
        mask_flip = jnp.flip(mask, axis=-1)
        embds_flip, _ = forward(model, image_flip, mask_flip)

        embds_rot = [jnp.rot90(x, k=-1, axes=(-2, -1)) for x in embds_rot]
        embds_flip = [jnp.flip(x, axis=-1) for x in embds_flip]

        mse = masked_pyramid_mse(preds, image, mask)
        rot_loss = embedding_similarity_loss(embds, embds_rot)
        flip_loss = embedding_similarity_loss(embds, embds_flip)

        loss = mse + embd_lambda * (rot_loss + flip_loss)
        return loss

    loss = jax.vmap(single_loss)(images, masks)
    loss = jnp.mean(loss)
    return loss


def masked_pyramid_mse(xs, y, mask):
    valid = (mask == 0).astype(jnp.float32)
    _, H, W = y.shape
    mse = 0.0

    for x in xs:
        _, h, w = x.shape
        sh, sw = H // h, W // w

        def downsample(z):
            C = z.shape[0]
            return z.reshape(C, h, sh, w, sw).mean(axis=(2, 4))

        weights = (downsample(valid) == 1.0).astype(jnp.float32)
        target = downsample(y * valid)
        squared_error = (x - target) ** 2

        pyramid_mse = jnp.sum(squared_error * weights) / jnp.maximum(jnp.sum(weights), 1.0)
        mse += pyramid_mse

    mse /= len(xs)
    return mse


def embedding_similarity_loss(xs, ys):
    loss = 0.0
    for x, y in zip(xs, ys):
        distance = optax.losses.cosine_distance(x, y, axis=0, epsilon=1e-6)
        loss += jnp.mean(distance)
    return loss / len(xs)
