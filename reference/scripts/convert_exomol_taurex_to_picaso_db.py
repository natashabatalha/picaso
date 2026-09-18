#!/usr/bin/env python3
"""Insert ExoMolOP TauREx cross-section into a PICASO SQLite opacity DB."""

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

from picaso.opacity_factory import open_local


FLOOR = 1e-50


def bracket(axis, value):
    value = float(np.clip(value, axis[0], axis[-1]))
    hi = int(np.searchsorted(axis, value, side="right"))
    if hi == 0:
        return 0, 0, 0.0
    if hi >= len(axis):
        return len(axis) - 1, len(axis) - 1, 0.0
    lo = hi - 1
    weight = (value - axis[lo]) / (axis[hi] - axis[lo])
    return lo, hi, float(weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xsec-h5", default="28Si-1H4__OY2T.R15000_0.3-50mu.xsec.TauREx.h5")
    parser.add_argument("--base-db", default="reference/opacities/opacities.db")
    parser.add_argument("--output-db", default="reference/opacities/opacities_with_SiH4.db")
    parser.add_argument("--molecule", default="SiH4")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    xsec_h5 = Path(args.xsec_h5)
    base_db = Path(args.base_db)
    output_db = Path(args.output_db)

    if not xsec_h5.exists():
        raise FileNotFoundError(xsec_h5)
    if not base_db.exists():
        raise FileNotFoundError(base_db)
    if output_db.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_db} exists; pass --overwrite to replace {args.molecule} rows")
    else:
        print(f"Copying {base_db} -> {output_db}")
        shutil.copy2(base_db, output_db)

    cur, conn = open_local(str(output_db))
    cur.execute("SELECT wavenumber_grid FROM header LIMIT 1")
    picaso_wno = cur.fetchone()[0]
    cur.execute("SELECT DISTINCT ptid, pressure, temperature FROM molecular ORDER BY ptid")
    pt_pairs = cur.fetchall()

    cur.execute("DELETE FROM molecular WHERE molecule = ?", (args.molecule,))
    conn.commit()

    with h5py.File(xsec_h5, "r") as h5:
        exo_wno = h5["bin_edges"][:]
        exo_p = h5["p"][:]
        exo_t = h5["t"][:]
        xsec = h5["xsecarr"]

        if xsec.shape != (len(exo_p), len(exo_t), len(exo_wno)):
            raise ValueError(f"Unexpected xsecarr shape: {xsec.shape}")

        print(f"ExoMol grid: {len(exo_p)} P x {len(exo_t)} T x {len(exo_wno)} wavenumbers")
        print(f"PICASO target: {len(pt_pairs)} PT rows x {len(picaso_wno)} wavenumbers")

        logp_axis = np.log10(exo_p)
        cache = {}

        def resampled_log_xsec(ip, it):
            key = (ip, it)
            if key not in cache:
                values = np.asarray(xsec[ip, it, :], dtype=float)
                values = np.interp(picaso_wno, exo_wno, values, left=FLOOR, right=FLOOR)
                cache[key] = np.log10(np.maximum(values, FLOOR))
            return cache[key]

        total = len(pt_pairs)
        for n, (ptid, pressure, temperature) in enumerate(pt_pairs, start=1):
            ip0, ip1, wp = bracket(logp_axis, np.log10(pressure))
            it0, it1, wt = bracket(exo_t, temperature)

            ll = resampled_log_xsec(ip0, it0)
            lh = resampled_log_xsec(ip0, it1)
            hl = resampled_log_xsec(ip1, it0)
            hh = resampled_log_xsec(ip1, it1)

            log_xsec = (
                (1.0 - wp) * (1.0 - wt) * ll
                + (1.0 - wp) * wt * lh
                + wp * (1.0 - wt) * hl
                + wp * wt * hh
            )
            opacity = 10.0**log_xsec

            cur.execute(
                "INSERT INTO molecular (ptid, molecule, pressure, temperature, opacity) VALUES (?, ?, ?, ?, ?)",
                (int(ptid), args.molecule, float(pressure), float(temperature), opacity),
            )

            if n % 100 == 0 or n == total:
                conn.commit()
                print(f"Inserted {n}/{total} {args.molecule} PT rows")

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM molecular WHERE molecule = ?", (args.molecule,))
    print(f"{args.molecule} row count:", cur.fetchone()[0])
    conn.close()
    print(f"Wrote {output_db}")


if __name__ == "__main__":
    main()
