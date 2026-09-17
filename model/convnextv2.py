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

    def __call__(self, x, mask=1.0):
        gx = jnp.sqrt(
            jnp.sum(x * x * mask, axis=(1, 2), keepdims=True)
            + self.eps
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

    def __call__(self, x, mask=1.0):
        x = x * mask
        y = self.conv(x) * mask
        y = utils.spatial_vmap(self.ln)(y)
        y = self.proj1(y)
        y = jax.nn.gelu(y)
        y = self.grn(y, mask)
        y = self.proj2(y)
        return (x + y) * mask


class ConvNeXtV2(eqx.Module):
    blocks: tuple[ConvNeXtV2Block]
    downs: tuple[eqx.nn.Conv2d]

    def __init__(self, depths, dims, key):
        keys = utils.key_gen(key)

    def __call__(self, x, mask=1.0):
        pass


def ConvNeXtV2Tiny(key):
    return ConvNeXtV2(
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        key=key
    )
