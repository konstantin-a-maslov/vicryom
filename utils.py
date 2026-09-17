import jax


def key_gen(key):
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def spatial_vmap(f):
    return jax.vmap(
        jax.vmap(f, in_axes=1, out_axes=1), 
        in_axes=1, out_axes=1,
    )
