import jax
import equinox as eqx
import model.convnextv2
import utils


class DecoderBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    ln: eqx.nn.LayerNorm

    def __init__(self, in_channels, out_channels, key):
        keys = utils.key_gen(key)

        self.conv1 = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1, 
            groups=in_channels,
            key=next(keys),
        )
        self.conv2 = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            key=next(keys),
        )
        self.ln = eqx.nn.LayerNorm(shape=out_channels)

    def __call__(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = utils.spatial_vmap(self.ln)(y)
        y = jax.nn.gelu(y)
        return y


class Decoder(eqx.Module):
    stages: int
    dec_blocks: tuple[model.convnextv2.ConvNeXtV2Block, ...]
    emd_blocks: tuple[DecoderBlock, ...]
    skip_projs: tuple[eqx.nn.Conv2d, ...]
    out_projs: tuple[eqx.nn.Conv2d, ...]

    def __init__(self, width, depth, skip_channels, emb_channels, out_channels, key):
        keys = utils.key_gen(key)
        self.stages = len(emb_channels)

        dec_blocks = []
        for _ in range(depth):
            dec_block = model.convnextv2.ConvNeXtV2Block(
                dim=width,
                key=next(keys),
            )
            dec_blocks.append(dec_block)
        self.dec_blocks = tuple(dec_blocks)

        skip_projs = []
        for stage in range(self.stages - 1):
            skip_proj = eqx.nn.Conv2d(
                in_channels=skip_channels[stage],
                out_channels=emb_channels[stage],
                kernel_size=1,
                key=next(keys),
            )
            skip_projs.append(skip_proj)
        self.skip_projs = tuple(skip_projs)

        emd_blocks, out_projs = [], [], 

        for stage in range(self.stages):
            emd_block = DecoderBlock(
                in_channels=width if stage == 0 else emb_channels[stage - 1],
                out_channels=emb_channels[stage],
                key=next(keys),
            )
            emd_blocks.append(emd_block)

            out_proj = eqx.nn.Conv2d(
                in_channels=emb_channels[stage],
                out_channels=out_channels,
                kernel_size=1,
                key=next(keys),
            )
            out_projs.append(out_proj)

        self.emd_blocks = tuple(emd_blocks)
        self.out_projs = tuple(out_projs)

    def __call__(self, x, skips):
        y = x
        for dec_block in self.dec_blocks:
            y = dec_block(y)

        embds, outputs = [], []

        for stage in range(self.stages):
            y = self.emd_blocks[stage](y)
            embds.append(y)

            output = self.out_projs[stage](y)
            outputs.append(output)

            if stage < self.stages - 1:
                c, h, w = y.shape
                y = jax.image.resize(y, shape=(c, 2 * h, 2 * w), method="bilinear")

                skip = self.skip_projs[stage](skips[stage])
                y = y + skip

        return embds, outputs
