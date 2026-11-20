"""Download a subset of cells from Xaira's X-Atlas/Orion h5ad files without the full file.

The figshare h5ad files (HCT116 ~209 GB, HEK293T ~350 GB) store X as an *uncompressed*
chunked CSR matrix, so each cell's counts are a few contiguous byte ranges. This script:
  1. opens the remote file over HTTP range requests and reads obs, var and X/indptr;
  2. picks cells (all cells of the requested targets + a sample of Non-Targeting controls);
  3. maps each picked cell's slice of X/data and X/indices to file byte ranges using the
     HDF5 chunk index, fetches those ranges in parallel, and writes a local h5ad.
Progress is saved per batch, so a rerun with the same arguments resumes.

Example (VCC 2026 test-panel targets + 20k controls from HCT116):
    python scripts/xaira_subset.py --line HCT116 \
        --targets data/vcc_2026_controls_final/pert_counts.csv \
        --n-controls 20000 --out data/xaira/HCT116_vcc_test_subset.h5ad
"""
import argparse, io, os, struct, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

FILES = {  # figshare.plus article 29190726
    "HCT116": ("https://ndownloader.figshare.com/files/55021257", 209354246272),
    "HEK293T": ("https://ndownloader.figshare.com/files/55074802", None),
}
GUIDE_CALLS = {  # per-cell guide tables, same cell order as the h5ad obs (checked at runtime)
    "HCT116": "https://ndownloader.figshare.com/files/59490731",
    "HEK293T": "https://ndownloader.figshare.com/files/59490734",
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_no_redirect = urllib.request.build_opener(_NoRedirect)
_signed = {}  # figshare url -> (S3 link, time resolved)
_signed_lock = threading.Lock()


def _s3_link(url, max_age=6.0):
    """figshare redirects to a signed S3 link that expires after 10 s. Reuse one link for a
    few seconds across threads instead of paying the redirect round trip on every request."""
    with _signed_lock:
        link, t = _signed.get(url, (None, 0.0))
        if link and time.time() - t < max_age:
            return link
        try:
            _no_redirect.open(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60)
            raise IOError("expected a redirect from figshare")
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308):
                raise
            link = e.headers["Location"]
        _signed[url] = (link, time.time())
        return link


def http_range(url, start, end, tries=10):
    """Bytes [start, end) of url, via a recently signed S3 link; retried with backoff."""
    for attempt in range(tries):
        try:
            link = _s3_link(url) if "figshare" in url else url
            req = urllib.request.Request(link, headers={"Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            if len(data) == end - start:
                return data
        except Exception:
            with _signed_lock:
                _signed.pop(url, None)  # link may have expired; resolve a fresh one
        time.sleep(min(2 ** attempt, 60))
    raise IOError(f"range {start}-{end} of {url} failed after {tries} tries")


def remote_size(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[1])


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests with a block cache (for h5py)."""

    def __init__(self, url, size, block=1 << 18, max_cache=4096):
        self.url, self.size, self.block, self.max_cache = url, size, block, max_cache
        self.pos, self.cache = 0, {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            if len(self.cache) >= self.max_cache:
                self.cache.pop(next(iter(self.cache)))
            s = i * self.block
            self.cache[i] = http_range(self.url, s, min(s + self.block, self.size))
        return self.cache[i]

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        out, got = memoryview(b), 0
        while got < n:
            i, o = divmod(self.pos + got, self.block)
            blk = self._block(i)
            k = min(len(blk) - o, n - got)
            out[got:got + k] = blk[o:o + k]
            got += k
        self.pos += n
        return n


def parallel_get(url, ranges, workers, piece=8 << 20):
    """Concatenated bytes of many (start, end) ranges, split into pieces and fetched in parallel."""
    jobs = [(a, min(a + piece, b)) for a0, b in ranges for a in range(a0, b, piece)]
    with ThreadPoolExecutor(workers) as pool:
        return b"".join(pool.map(lambda r: http_range(url, *r), jobs))


def read_contiguous(url, dset, workers):
    """A contiguous, uncompressed numeric dataset, fetched in parallel."""
    off = dset.id.get_offset()
    assert off is not None and dset.compression is None
    nbytes = dset.size * dset.dtype.itemsize
    return np.frombuffer(parallel_get(url, [(off, off + nbytes)], workers), dtype=dset.dtype)


def read_chunked(url, dset, workers):
    """A 1-D chunked, uncompressed numeric dataset, fetched chunk by chunk in parallel."""
    assert dset.compression is None
    offs = chunk_offsets(dset)
    ranges = byte_ranges(0, dset.shape[0], dset.chunks[0], dset.dtype.itemsize, offs)
    return np.frombuffer(parallel_get(url, ranges, workers), dtype=dset.dtype)


def read_categorical(url, grp, workers):
    """An anndata categorical column; codes fetched in parallel, categories via h5py (small)."""
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in grp["categories"][:]], dtype=object)
    codes = read_contiguous(url, grp["codes"], workers)
    return np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)


def read_column(grp):
    """An anndata obs/var column: plain dataset, or categorical group (codes + categories)."""
    if isinstance(grp, h5py.Group):
        cats = grp["categories"][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats], dtype=object)
        codes = grp["codes"][:]
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)
    v = grp[:]
    return np.array([x.decode() for x in v], dtype=object) if v.dtype.kind in "SO" else v


def btree_root(url, group, name):
    """Address of a chunked dataset's v1 B-tree chunk index, read from its object header.
    Avoids h5py/HDF5 metadata calls that traverse the whole index one request at a time."""
    addr = group.id.links.get_info(name.encode()).u
    b = http_range(url, addr, addr + 4096)
    ver, _, nmsg = struct.unpack_from("<BBH", b, 0)
    assert ver == 1, "expected a version-1 object header"
    p = 16
    for _ in range(nmsg):
        mtype, msize = struct.unpack_from("<HH", b, p)
        body = b[p + 8:p + 8 + msize]
        if mtype == 8:  # data layout message
            assert body[0] == 3 and body[1] == 2, "expected a v3 chunked layout"
            return struct.unpack_from("<Q", body, 3)[0], body[2]  # (B-tree address, ndims + 1)
        p += 8 + msize
    raise ValueError(f"no layout message in the first 4 KB of the {name} header")


def btree_chunk_offsets(url, group, name, workers, cache=None):
    """File offset of every chunk of a 1-D chunked dataset, read straight from its v1 B-tree,
    one tree level at a time with all nodes of a level fetched in parallel."""
    if cache and os.path.exists(cache):
        return np.load(cache)
    dset = group[name]
    clen, nbytes = dset.chunks[0], dset.chunks[0] * dset.dtype.itemsize
    off = np.full(-(-dset.shape[0] // clen), -1, dtype=np.int64)
    root, nd = btree_root(url, group, name)
    entry = 8 + 8 * nd + 8  # key (size, filter mask, nd offsets) + child address

    def node(a):
        b = http_range(url, a, a + 4096)
        n = struct.unpack_from("<H", b, 6)[0]
        need = 24 + n * entry + 8 + 8 * nd
        return b if need <= len(b) else http_range(url, a, a + need)

    level = [root]
    with ThreadPoolExecutor(max(workers, 32)) as pool:
        while level:
            nxt = []
            for b in pool.map(node, level):
                assert b[:4] == b"TREE" and b[4] == 1, "not a chunk B-tree node"
                lvl, n = b[5], struct.unpack_from("<H", b, 6)[0]
                for i in range(n):
                    q = 24 + i * entry
                    size, mask = struct.unpack_from("<II", b, q)
                    elem0 = struct.unpack_from("<Q", b, q + 8)[0]
                    child = struct.unpack_from("<Q", b, q + 8 + 8 * nd)[0]
                    if lvl == 0:
                        assert mask == 0 and elem0 % clen == 0
                        off[elem0 // clen] = child
                    else:
                        nxt.append(child)
            print(f"  {name}: read {len(level)} index nodes", flush=True)
            level = nxt
    assert (off >= 0).all(), "chunk index incomplete"
    if cache:
        np.save(cache, off)
    return off


def chunk_offsets(dset):
    """File byte offset of every chunk of a 1-D chunked dataset, indexed by chunk number."""
    n_chunks = -(-dset.shape[0] // dset.chunks[0])
    off = np.full(n_chunks, -1, dtype=np.int64)
    def cb(info):
        off[info.chunk_offset[0] // dset.chunks[0]] = info.byte_offset
    dset.id.chunk_iter(cb)
    assert (off >= 0).all(), "missing chunks"
    return off


def byte_ranges(start, end, chunk_len, itemsize, offsets):
    """File byte ranges covering elements [start, end) of a chunked 1-D dataset."""
    out = []
    while start < end:
        c = start // chunk_len
        stop = min(end, (c + 1) * chunk_len)
        b0 = offsets[c] + (start - c * chunk_len) * itemsize
        out.append((b0, b0 + (stop - start) * itemsize))
        start = stop
    return out


def fetch_cell(url, s, e, meta):
    """(indices, data) of one cell: elements [s, e) of X/indices and X/data."""
    parts = {}
    for name in ("indices", "data"):
        clen, dtype, offs = meta[name]
        buf = b"".join(http_range(url, a, b) for a, b in byte_ranges(s, e, clen, dtype.itemsize, offs))
        parts[name] = np.frombuffer(buf, dtype=dtype)
    return parts["indices"].astype(np.int32), parts["data"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--line", choices=FILES, default="HCT116")
    p.add_argument("--targets", required=True, help="CSV with a target_gene column, or comma-separated genes")
    p.add_argument("--n-controls", type=int, default=20000)
    p.add_argument("--max-cells-per-target", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--batch", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    url, size = FILES[a.line]
    size = size or remote_size(url)
    work = a.out + ".parts"
    os.makedirs(work, exist_ok=True)

    t0 = time.time()
    h = h5py.File(RangeFile(url, size), "r")
    meta_path = os.path.join(work, "meta.npz")
    if os.path.exists(meta_path):
        m = np.load(meta_path, allow_pickle=True)
        barcodes, target, sample, indptr, var_names = (m[k] for k in
            ("barcodes", "target", "sample", "indptr", "var_names"))
    else:
        print("reading obs, var and indptr ...", flush=True)
        target = read_categorical(url, h["obs"]["gene_target"], a.workers)
        sample = read_categorical(url, h["obs"]["sample"], a.workers)
        indptr = read_chunked(url, h["X"]["indptr"], a.workers)
        print(f"  obs codes + indptr ({time.time() - t0:.0f}s)", flush=True)
        var_names = read_column(h["var"]["_index"])
        print(f"  var names ({time.time() - t0:.0f}s)", flush=True)
        # obs/_index is variable-length strings (slow remotely): take barcodes from the guide
        # table instead and confirm its row order matches the h5ad's targets.
        g = pd.read_csv(GUIDE_CALLS[a.line], usecols=["cell_barcode", "gene_target"], compression="gzip")
        assert len(g) == len(target) and (g["gene_target"].to_numpy() == target).all(), \
            "guide table order does not match h5ad obs"
        barcodes = g["cell_barcode"].to_numpy(dtype=object)
        np.savez(meta_path, barcodes=barcodes, target=target, sample=sample,
                 indptr=indptr, var_names=var_names)
    print(f"{len(barcodes):,} cells x {len(var_names):,} genes ({time.time() - t0:.0f}s)", flush=True)

    if os.path.exists(a.targets):
        wanted = set(pd.read_csv(a.targets)["target_gene"])
    else:
        wanted = set(a.targets.split(","))
    rng = np.random.default_rng(a.seed)
    tgt = pd.Series(target)
    pick = []
    for g, idx in tgt[tgt.isin(wanted)].groupby(tgt).groups.items():
        idx = np.asarray(idx)
        if a.max_cells_per_target and len(idx) > a.max_cells_per_target:
            idx = rng.choice(idx, a.max_cells_per_target, replace=False)
        pick.append(idx)
    ctrl = np.flatnonzero(target == "Non-Targeting")
    pick.append(rng.choice(ctrl, min(a.n_controls, len(ctrl)), replace=False))
    pick = np.sort(np.concatenate(pick))
    missing = sorted(wanted - set(tgt.unique()))
    nnz = int((indptr[pick + 1] - indptr[pick]).sum())
    print(f"picked {len(pick):,} cells ({len(wanted) - len(missing)} targets, "
          f"missing {missing}); {nnz * 12 / 1e9:.2f} GB to fetch", flush=True)

    print("reading HDF5 chunk index ...", flush=True)
    meta = {k: (h["X"][k].chunks[0], h["X"][k].dtype,
                btree_chunk_offsets(url, h["X"], k, a.workers, os.path.join(work, f"chunks_{k}.npy")))
            for k in ("indices", "data")}
    print(f"chunk index read ({time.time() - t0:.0f}s)", flush=True)

    batches = [pick[i:i + a.batch] for i in range(0, len(pick), a.batch)]
    with ThreadPoolExecutor(a.workers) as pool:
        for bi, rows in enumerate(batches):
            path = os.path.join(work, f"batch_{bi:05d}.npz")
            if os.path.exists(path):
                continue
            tb = time.time()
            res = list(pool.map(lambda r: fetch_cell(url, indptr[r], indptr[r + 1], meta), rows))
            ind = np.concatenate([r[0] for r in res])
            dat = np.concatenate([r[1] for r in res])
            ptr = np.concatenate([[0], np.cumsum([len(r[0]) for r in res])])
            np.savez(path, rows=rows, indices=ind, data=dat, indptr=ptr)
            mb = len(ind) * 12 / 1e6
            print(f"batch {bi + 1}/{len(batches)}: {mb:.0f} MB in {time.time() - tb:.0f}s "
                  f"({mb / (time.time() - tb):.1f} MB/s)", flush=True)

    print("assembling h5ad ...", flush=True)
    mats, rows = [], []
    for bi in range(len(batches)):
        z = np.load(os.path.join(work, f"batch_{bi:05d}.npz"))
        mats.append(sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                                  shape=(len(z["rows"]), len(var_names))))
        rows.append(z["rows"])
    rows = np.concatenate(rows)
    obs = pd.DataFrame({"target_gene": target[rows], "sample": sample[rows]},
                       index=pd.Index(barcodes[rows].astype(str), name="cell_barcode"))
    obs["target_gene"] = obs["target_gene"].replace({"Non-Targeting": "non-targeting"}).astype("category")
    obs["sample"] = obs["sample"].astype("category")
    out = ad.AnnData(X=sp.vstack(mats).tocsr(), obs=obs,
                     var=pd.DataFrame(index=pd.Index(var_names.astype(str), name="gene")))
    out.uns["source"] = f"X-Atlas/Orion {a.line}, figshare.plus 29190726 (non-commercial license)"
    out.write_h5ad(a.out, compression="gzip")
    print(f"wrote {a.out}: {out.n_obs:,} cells, {out.n_vars:,} genes ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
