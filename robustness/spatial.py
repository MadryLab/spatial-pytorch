import torch as ch

_MESHGRIDS = {}

def make_meshgrid(x):
    bs, _, _, dim = x.shape
    device = x.get_device()

    key = (dim, bs, device)
    if key in _MESHGRIDS:
        return _MESHGRIDS[key]

    space = ch.linspace(-1, 1, dim)
    meshgrid = ch.meshgrid([space, space])
    gridder = ch.cat([meshgrid[1][..., None], meshgrid[0][..., None]], dim=2)
    grid = gridder[None, ...].repeat(bs, 1, 1, 1)
    ones = ch.ones(grid.shape[:3] + (1,))
    final_grid = ch.cat([grid, ones], dim=3)
    expanded_grid = final_grid[..., None].cuda()

    _MESHGRIDS[key] = expanded_grid

    return expanded_grid

def unif(size, mini, maxi):
    args = {"from": mini, "to":maxi}
    return ch.cuda.FloatTensor(size=size).uniform_(**args)

def make_slice(a, b, c):
    to_cat = [a[None, ...], b[None, ...], c[None, ...]]
    return ch.cat(to_cat, dim=0)

def make_mats(rots, txs):
    # rots: degrees
    # txs: % of image dim

    rots = rots * 0.01745327778 # deg to rad
    txs = txs * 2

    cosses = ch.cos(rots)
    sins = ch.sin(rots)

    top_slice = make_slice(cosses, -sins, txs[:, 0])[None, ...].permute([2, 0, 1])
    bot_slice = make_slice(sins, cosses, txs[:, 1])[None, ...].permute([2, 0, 1])

    mats = ch.cat([top_slice, bot_slice], dim=1)

    mats = mats[:, None, None, :, :]
    mats = mats.repeat(1, 224, 224, 1, 1)
    return mats

def transform(x, rots, txs):
    assert x.shape[2] == x.shape[3]

    with ch.no_grad():
        meshgrid = make_meshgrid(x)
        tfm_mats = make_mats(rots, txs)

        new_coords = ch.matmul(tfm_mats, meshgrid)
        new_coords = new_coords.squeeze_(-1)

        new_image = ch.nn.functional.grid_sample(x, new_coords)
        return new_image
