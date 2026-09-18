import jax
import jax.numpy as jnp
import equinox as eqx
import utils


class Stem(eqx.Module):
    stages: int
    convs: tuple[eqx.nn.Conv2d, ...]
    lns: tuple[eqx.nn.LayerNorm, ...]
    mask_pool: eqx.nn.MaxPool2d

    def __init__(self, in_channels, dims, key):
        keys = utils.key_gen(key)
        self.stages = len(dims)

        convs, lns = [], []

        for stage in range(self.stages):
            conv = eqx.nn.Conv2d(
                in_channels=in_channels if stage == 0 else dims[stage - 1],
                out_channels=dims[stage],
                kernel_size=2,
                stride=2,
                key=next(keys),
            )
            convs.append(conv)

            ln = eqx.nn.LayerNorm(shape=dims[stage])
            lns.append(ln)

        self.convs = tuple(convs)
        self.lns = tuple(lns)

        self.mask_pool = eqx.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def __call__(self, x, mask=None):
        if mask is None:
            mask = jnp.ones_like(x[:1])

        y = x * mask
        outputs = [y]

        for stage in range(self.stages):
            mask = 1.0 - self.mask_pool(1.0 - mask)
            y = self.convs[stage](y)
            y = y * mask
            y = utils.spatial_vmap(self.lns[stage])(y)
            y = jax.nn.gelu(y)
            y = y * mask
            outputs.append(y)

        return outputs, mask
    