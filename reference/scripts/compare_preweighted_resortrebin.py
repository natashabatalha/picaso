"""
Runs the same brown dwarf climate model twice, once with preweighted correlated-k tables (chemistry
built into the table) and once with resortrebin (per-molecule tables mixed on the fly with the same
Visscher 2121 chemistry), and plots the two pressure-temperature profiles.

Everything except the opacity method is identical: T_int, gravity, pressure grid, starting guess,
convective zone guess, metallicity and C/O. At high metallicity any error in how resortrebin weights
gases by abundance shows up as a temperature offset between the two profiles.

Needs the full reference data: reference/opacities/preweighted/sonora_2121grid_feh*_co*.hdf5,
reference/opacities/resortrebin/*_1460.hdf5 and reference/sonora_grids/bobcat/.

    python compare_preweighted_resortrebin.py --feh 2.0 --co 0.46 --out pt_compare.png

--with-fix adds a third run, resortrebin with each layer's mixed k multiplied by the summed mixing
ratio of the gases that were mixed (sum x_i k_i rather than sum x_i k_i / sum x_i). It patches the
method for that run only and does not change picaso.
"""
import argparse
import os
import warnings

import astropy.units as u
import numpy as np

warnings.filterwarnings('ignore')

SOLAR_CO = 0.458  # absolute C/O that cto_relative=1 maps to


def _patch_abundance_weighting():
    """
    Wraps RetrieveCKs.mix_my_opacities_gasesfly so the mixed opacity is multiplied by the total
    mixing ratio of the mixed gases in each layer. The total is the same at all four grid corners,
    so this equals returning kmix_bin * mix_t from deq_chem.do_mixing_mono_gasesfly.
    Returns a function that undoes the patch.
    """
    from picaso.atmsetup import is_opacity_excluded
    from picaso.optics import RetrieveCKs
    original = RetrieveCKs.mix_my_opacities_gasesfly

    def mix_times_total(self, atmosphere, exclude_mol=1):
        original(self, atmosphere, exclude_mol=exclude_mol)
        mixed = [m for m in atmosphere.molecules if not is_opacity_excluded(exclude_mol, m, 'line')]
        total = atmosphere.layer['mixingratios'][mixed].values.sum(axis=1)
        self.molecular_opa = self.molecular_opa * total[:, np.newaxis, np.newaxis]

    RetrieveCKs.mix_my_opacities_gasesfly = mix_times_total
    return lambda: setattr(RetrieveCKs, 'mix_my_opacities_gasesfly', original)


def run_climate(method, feh=2.0, co_absolute=0.46, tint=700, gravity=316, rcb_guess=79):
    """
    Converges one climate model and returns (pressure [bar], temperature [K]).

    method : 'preweighted' or 'resortrebin'
    feh : log10 metallicity relative to solar, must match a preweighted table (e.g. 2.0)
    co_absolute : absolute C/O, must match a preweighted table (e.g. 0.46 for solar)
    """
    from picaso import justdoit as jdi
    refdata = os.environ['picaso_refdata']

    cl_run = jdi.inputs(calculation='browndwarf', climate=True)
    cl_run.gravity(gravity=gravity, gravity_unit=u.Unit('m/(s**2)'))
    cl_run.effective_temp(tint)

    # same starting guess for both runs: the solar metallicity Sonora Bobcat profile
    guess = os.path.join(refdata, 'sonora_grids', 'bobcat', f't{tint}g{gravity}nc_m0.0.cmp.gz')
    pressure, temp_guess = np.loadtxt(guess, usecols=[1, 2], unpack=True, skiprows=1)
    cl_run.inputs_climate(temp_guess=temp_guess, pressure=pressure, rcb_guess=rcb_guess, rfacv=0.0)

    undo_patch = None
    if method == 'resortrebin (fixed)':
        undo_patch = _patch_abundance_weighting()
        method = 'resortrebin'

    if method == 'preweighted':
        ck_db = os.path.join(refdata, 'opacities', 'preweighted',
                             f'sonora_2121grid_feh{feh:.1f}_co{co_absolute:.2f}.hdf5')
        opacity = jdi.opannection(ck_db=ck_db, method='preweighted')
    elif method == 'resortrebin':
        opacity = jdi.opannection(method='resortrebin')  # all available molecules
        cl_run.atmosphere(mh=10**feh, cto_relative=co_absolute / SOLAR_CO, chem_method='visscher')
    else:
        raise ValueError(f'unknown method {method}')

    try:
        out = cl_run.climate(opacity)
    finally:
        if undo_patch:
            undo_patch()
    return np.asarray(out['pressure']), np.asarray(out['temperature'])


def plot_profiles(profiles, title, out_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ink, muted, surface = '#0b0b0b', '#52514e', '#fcfcfb'
    styles = {'preweighted': dict(color='#2a78d6', linestyle='-'),
              'resortrebin': dict(color='#eb6834', linestyle='--'),
              'resortrebin (fixed)': dict(color='#1baf7a', linestyle=':')}

    fig, ax = plt.subplots(figsize=(6, 7), facecolor=surface)
    ax.set_facecolor(surface)
    for name, (pressure, temperature) in profiles.items():
        ax.plot(temperature, pressure, linewidth=2, label=name, **styles[name])
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_xlabel('Temperature [K]', color=ink)
    ax.set_ylabel('Pressure [bar]', color=ink)
    ax.set_title(title, color=ink, fontsize=11)
    ax.grid(True, which='major', color='#e4e3df', linewidth=0.8)
    ax.tick_params(colors=muted)
    for spine in ax.spines.values():
        spine.set_color('#c3c2b7')
    ax.legend(frameon=False, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, facecolor=surface)
    print(f'saved {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--feh', type=float, default=2.0)
    parser.add_argument('--co', type=float, default=0.46, help='absolute C/O')
    parser.add_argument('--out', default='pt_preweighted_vs_resortrebin.png')
    parser.add_argument('--with-fix', action='store_true', help='also run resortrebin with sum x_i k_i mixing')
    args = parser.parse_args()

    methods = ['preweighted', 'resortrebin'] + (['resortrebin (fixed)'] if args.with_fix else [])
    profiles = {method: run_climate(method, feh=args.feh, co_absolute=args.co) for method in methods}
    p_ref, t_ref = profiles['preweighted']
    for method in methods[1:]:
        p, t = profiles[method]
        dt = t - np.interp(np.log10(p), np.log10(p_ref), t_ref)
        print(f'{method} - preweighted temperature: median {np.median(dt):+.1f} K, '
              f'max |dT| {np.max(np.abs(dt)):.1f} K at {p[np.argmax(np.abs(dt))]:.3g} bar')
    plot_profiles(profiles, f'T$_{{int}}$ = 700 K, g = 316 m/s$^2$, [M/H] = {args.feh:+.1f}, C/O = {args.co}',
                  args.out)
