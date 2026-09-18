import jax.numpy as jnp
import equinox as eqx
import model.stem
import model.convnextv2
import model.decoder
import utils


class ViCryoM(eqx.Module):
    # encoder
    stem: model.stem.Stem
    enc_proj: eqx.nn.Conv2d
    bkbn: model.convnextv2.ConvNeXtV2
    # decoder
    dec_proj: eqx.nn.Conv2d
    dec: model.decoder.Decoder

    def __init__(
            self, 
            in_channels,
            stem_dims, 
            bkbn_depths, bkbn_dims, 
            dec_width, dec_depth, 
            emb_channels, 
            out_channels, 
            key,
        ):
        keys = utils.key_gen(key)

        # encoder
        self.stem = model.stem.Stem(
            in_channels=in_channels,
            dims=stem_dims,
            key=next(keys),
        )
        self.enc_proj = eqx.nn.Conv2d(
            in_channels=stem_dims[-1],
            out_channels=bkbn_dims[0],
            kernel_size=1,
            key=next(keys),
        )
        self.bkbn = model.convnextv2.ConvNeXtV2(
            depths=bkbn_depths,
            dims=bkbn_dims,
            key=next(keys),
        )

        # decoder
        self.dec_proj = eqx.nn.Conv2d(
            in_channels=bkbn_dims[-1],
            out_channels=dec_width,
            kernel_size=1,
            key=next(keys),
        )
        skip_channels = [in_channels, *stem_dims[:-1], *bkbn_dims[:-1]]
        self.dec = model.decoder.Decoder(
            width=dec_width, 
            depth=dec_depth, 
            skip_channels=skip_channels[::-1],
            emb_channels=emb_channels, 
            out_channels=out_channels,
            key=next(keys),
        )

    def __call__(self, x, mask=None):
        if mask is None:
            mask = jnp.ones_like(x[:1])

        stem_outputs, mask = self.stem(x, mask)
        bkbn_in = self.enc_proj(stem_outputs[-1])
        bkbn_outputs = self.bkbn(bkbn_in, mask)

        enc_outputs = [*stem_outputs[:-1], *bkbn_outputs]
        dec_input = self.dec_proj(enc_outputs[-1])
        skips = enc_outputs[:-1][::-1]

        embds, outputs = self.dec(dec_input, skips)
        
        return embds, outputs


def ViCryoMTiny(in_channels, out_channels, key):
    return ViCryoM(
        in_channels=in_channels, 
        stem_dims=[32, 64], 
        bkbn_depths=[3, 3, 9, 3], 
        bkbn_dims=[96, 192, 384, 768], 
        dec_width=96,
        dec_depth=2, 
        emb_channels=[64, 64, 64, 32, 32, 32],
        out_channels=out_channels, 
        key=key,
    )
