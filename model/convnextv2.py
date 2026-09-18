import jax
import jax.numpy as jnp
import equinox as eqx
import utils


class GlobalResponseNormalisation(eqx.Module):
    gamma: jax.Array
    beta: jax.Array
    eps: float

    def __init__(self, dim, eps=1e-6):
        self.gamma = jnp.zeros((dim, 1, 1))
        self.beta = jnp.zeros((dim, 1, 1))
        self.eps = eps

    def __call__(self, x, mask=None):
        if mask is None:
            mask = jnp.ones_like(x[:1])

        gx = jnp.sqrt(
            jnp.sum(x * x * mask, axis=(1, 2), keepdims=True)
            + self.eps # to avoid potential singularity
        )
        nx = gx / (gx.mean(axis=0, keepdims=True) + self.eps)

        return x + self.gamma * (x * nx) + self.beta


class ConvNeXtV2Block(eqx.Module):
    conv: eqx.nn.Conv2d
    ln: eqx.nn.LayerNorm
    proj1: eqx.nn.Conv2d
    grn: GlobalResponseNormalisation
    proj2: eqx.nn.Conv2d

    def __init__(self, dim, key):
        keys = utils.key_gen(key)

        self.conv = eqx.nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            padding=3,
            groups=dim, # depth-wise
            key=next(keys),
        )
        self.ln = eqx.nn.LayerNorm(shape=dim)
        self.proj1 = eqx.nn.Conv2d(
            in_channels=dim,
            out_channels=4 * dim,
            kernel_size=1,
            key=next(keys),
        )
        self.grn = GlobalResponseNormalisation(dim=4 * dim)
        self.proj2 = eqx.nn.Conv2d(
            in_channels=4 * dim,
            out_channels=dim,
            kernel_size=1,
            key=next(keys),
        )

    def __call__(self, x, mask=None):
        if mask is None:
            mask = jnp.ones_like(x[:1])

        x = x * mask
        y = self.conv(x) * mask
        y = utils.spatial_vmap(self.ln)(y)
        y = self.proj1(y)
        y = jax.nn.gelu(y)
        y = self.grn(y, mask)
        y = self.proj2(y)

        return (x + y) * mask


class DownsampleBlock(eqx.Module):
    ln: eqx.nn.LayerNorm
    conv: eqx.nn.Conv2d
    mask_pool: eqx.nn.MaxPool2d

    def __init__(self, in_channels, out_channels, key):
        self.ln = eqx.nn.LayerNorm(shape=in_channels)
        self.conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            key=key,
        )
        self.mask_pool = eqx.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def __call__(self, x, mask=None):
        if mask is None:
            mask = jnp.ones_like(x[:1])

        y = utils.spatial_vmap(self.ln)(x)
        y = y * mask
        y = self.conv(y)
        mask = 1.0 - self.mask_pool(1.0 - mask)

        return y, mask


class ConvNeXtV2(eqx.Module):
    stages: int
    blocks: tuple[tuple[ConvNeXtV2Block, ...], ...]
    downs: tuple[eqx.Module, ...]
    
    def __init__(self, depths, dims, key):
        keys = utils.key_gen(key)
        self.stages = len(depths)

        blocks, downs = [], []

        for stage in range(self.stages):
            stage_blocks = tuple(
                ConvNeXtV2Block(
                    dim=dims[stage],
                    key=next(keys),
                )
                for _ in range(depths[stage])
            )
            blocks.append(stage_blocks)

            if stage < self.stages - 1:
                down = DownsampleBlock(
                    in_channels=dims[stage],
                    out_channels=dims[stage + 1],
                    key=next(keys),
                )
                downs.append(down)

        self.blocks = tuple(blocks)
        self.downs = tuple(downs)

    def __call__(self, x, mask=None):
        if mask is None:
            mask = jnp.ones_like(x[:1])

        y, outputs = x, []

        for stage in range(self.stages):
            for block in self.blocks[stage]:
                y = block(y, mask)
            outputs.append(y)

            if stage < self.stages - 1:
                y, mask = self.downs[stage](y, mask)

        return outputs


def ConvNeXtV2Atto(key):
    return ConvNeXtV2(
        depths=[2, 2, 6, 2], 
        dims=[40, 80, 160, 320], 
        key=key,
    )


def ConvNeXtV2Tiny(key):
    return ConvNeXtV2(
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        key=key,
    )


def ConvNeXtV2Base(key):
    return ConvNeXtV2(
        depths=[3, 3, 27, 3], 
        dims=[128, 256, 512, 1024], 
        key=key,
    )
