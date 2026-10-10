"""Generate the spectrum icons shown on the docs landing page.

Computes a reflected-light, thermal-emission and transmission spectrum, a
pressure-temperature profile, and a model-vs-data eclipse spectrum with PICASO
(same setups as the A_basics tutorials) and draws each as minimal, transparent
SVG line art for the cards in docs/index.md. The phase crescents are cut from
phases_source.png (the bottom strip of the original PICASO use-cases figure).

    python docs/_scripts/make_landing_icons.py

Needs picaso_refdata and an opacity file covering 0.3-12 um.
"""
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import picaso.justdoit as jdi

warnings.filterwarnings('ignore')

DOCS = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(DOCS, '_static', 'landing')
R = 120   # resampling resolution: enough to show bands, smooth enough for an icon

# colours follow the original use-cases figure, brightened slightly to read in dark mode too
BLUE, MAGENTA, RED, ORANGE, TEAL = '#3b7dd8', '#c2368f', '#d23a48', '#e07b28', '#2a9d8f'


def hot_jupiter(opa):
    case = jdi.inputs()
    case.phase_angle(0)
    case.gravity(mass=1, mass_unit=jdi.u.Unit('M_jup'),
                 radius=1.2, radius_unit=jdi.u.Unit('R_jup'))
    case.star(opa, 4000, 0.0122, 4.437, radius=0.7, radius_unit=jdi.u.Unit('R_sun'))
    case.atmosphere(filename=jdi.HJ_pt(), sep=r'\s+')
    return case


def reflected():
    opa = jdi.opannection(wave_range=[0.3, 1])
    case = jdi.inputs()
    case.phase_angle(0)
    case.gravity(gravity=25, gravity_unit=jdi.u.Unit('m/(s**2)'))
    case.star(opa, 5000, 0, 4.0)
    case.atmosphere(filename=jdi.jupiter_pt(), sep=r'\s+')
    df = case.spectrum(opa, calculation='reflected')
    wno, alb = jdi.mean_regrid(df['wavenumber'], df['albedo'], R=R)
    return 1e4 / wno, alb


def thermal():
    opa = jdi.opannection(wave_range=[1, 12])
    case = hot_jupiter(opa)
    df = case.spectrum(opa, calculation='thermal')
    wno, fp = jdi.mean_regrid(df['wavenumber'], df['thermal'], R=R)
    return 1e4 / wno, np.log10(fp)


def transmission():
    opa = jdi.opannection(wave_range=[0.6, 5])
    case = hot_jupiter(opa)
    case.approx(p_reference=10)
    df = case.spectrum(opa, calculation='transmission')
    wno, depth = jdi.mean_regrid(df['wavenumber'], df['transit_depth'], R=R)
    return 1e4 / wno, depth


def eclipse():
    opa = jdi.opannection(wave_range=[2.5, 5])
    case = hot_jupiter(opa)
    case.star(opa, 4000, 0.0122, 4.437, radius=0.7, radius_unit=jdi.u.Unit('R_sun'),
              semi_major=0.03, semi_major_unit=jdi.u.Unit('au'))
    df = case.spectrum(opa, calculation='thermal')
    wno, fpfs = jdi.mean_regrid(df['wavenumber'], df['fpfs_thermal'], R=60)
    return 1e4 / wno, fpfs


def draw_pt(name, colors):
    """A family of Guillot profiles at increasing Teq, like a climate grid."""
    case = jdi.inputs()
    case.gravity(gravity=25, gravity_unit=jdi.u.Unit('m/(s**2)'))
    fig, ax = plt.subplots(figsize=(4, 1.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.axis('off')
    for teq, color in zip([500, 800, 1100, 1400, 1700], colors):
        df = case.guillot_pt(teq, T_int=200, nlevel=80, logg1=-1, alpha=0.5)
        ax.plot(df['temperature'], np.log10(df['pressure']), color=color, lw=2.0,
                solid_joinstyle='round', solid_capstyle='round')
    ax.set_ylim(1.6, -6.2)                     # pressure increases downward
    ax.margins(x=0.03)
    fig.subplots_adjust(0.02, 0.04, 0.98, 0.96)
    save(fig, name)


def crescents(name, phases=(0, 2, 4, 6, 8), pitch=235, pad=20):
    """Cut the phase crescents (with their reflections) out of phases_source.png."""
    src = os.path.join(os.path.dirname(__file__), 'phases_source.png')
    strip = np.asarray(Image.open(src).convert('RGB')).astype(float)
    ink = strip.min(axis=2) < 235
    cols = np.where(ink.any(axis=0))[0]
    blobs, start = [], cols[0]
    for a, b in zip(cols[:-1], cols[1:]):
        if b - a > 8:
            blobs.append((start, a)); start = b
    blobs.append((start, cols[-1]))
    # slot centres in the original are evenly spaced; keep each crescent's offset
    slot0, slot_pitch = (blobs[0][0] + blobs[0][1]) / 2, (blobs[-1][1] - blobs[0][0] - (blobs[0][1] - blobs[0][0])) / (len(blobs) - 1)
    h = strip.shape[0]
    canvas = np.full((h + 2 * pad, pitch * len(phases) + 2 * pad, 3), 255.0)
    for k, i in enumerate(phases):
        x0, x1 = blobs[i]
        offset = x0 - (slot0 + slot_pitch * i)
        cx = pad + pitch * k + pitch / 2
        dst = int(round(cx + offset))
        canvas[pad:pad + h, dst:dst + (x1 - x0 + 1)] = strip[:, x0:x1 + 1]
    # Each crescent is a single colour faded toward white (edges, reflection).
    # Recover that as solid colour + alpha, so it works on light and dark cards.
    rgba = np.zeros(canvas.shape[:2] + (4,))
    for k in range(len(phases)):
        x0, x1 = pad + pitch * k, pad + pitch * (k + 1)
        tile = canvas[:, x0:x1]
        dist = (255 - tile).sum(axis=2)
        base = np.median(tile[dist > 0.9 * dist.max()], axis=0)   # the crescent's colour
        frac = ((255 - tile) * (255 - base)).sum(axis=2) / ((255 - base) ** 2).sum()
        rgba[:, x0:x1, :3] = base
        rgba[:, x0:x1, 3] = np.clip(frac, 0, 1) * 255
    rgba = rgba.astype(np.uint8)
    img = Image.fromarray(rgba, 'RGBA')
    # match the 4:1.5 aspect of the SVG icons, then size for 2x displays
    w, h = img.size
    target_h = int(w * 1.5 / 4)
    if target_h > h:
        framed = Image.new('RGBA', (w, target_h), (255, 255, 255, 0))
        framed.paste(img, (0, (target_h - h) // 2))
        img = framed
    img = img.resize((800, 300), Image.LANCZOS)
    path = os.path.join(OUT, f'{name}.png')
    img.save(path, optimize=True)
    print('wrote', os.path.relpath(path))


def save(fig, name):
    path = os.path.join(OUT, f'{name}.svg')
    fig.savefig(path, format='svg', transparent=True, metadata={'Date': None})
    plt.close(fig)
    print('wrote', os.path.relpath(path))


def draw(name, wave, flux, color, logx=False, data=False):
    order = np.argsort(wave)
    wave, flux = wave[order], flux[order]
    fig, ax = plt.subplots(figsize=(4, 1.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.axis('off')
    if logx:
        ax.set_xscale('log')
    lo = flux.min() - 0.12 * np.ptp(flux)
    ax.fill_between(wave, flux, lo, color=color, alpha=0.12, linewidth=0)
    if data:
        # simulated observations along the model, like the original figure
        rng = np.random.default_rng(4)
        idx = np.linspace(4, len(wave) - 5, 20).astype(int)
        err = 0.07 * np.ptp(flux)
        obs = flux[idx] + rng.normal(0, err * 0.7, len(idx))
        ax.plot(wave, flux, color=color, lw=1.6, alpha=0.5)
        ax.errorbar(wave[idx], obs, yerr=err, fmt='o', ms=5.5, color=color,
                    mfc='none', mew=1.6, elinewidth=1.4, capsize=0)
    else:
        ax.plot(wave, flux, color=color, lw=2.0, solid_joinstyle='round')
    ax.set_xlim(wave.min(), wave.max())
    ax.set_ylim(lo, flux.max() + 0.08 * np.ptp(flux))
    fig.subplots_adjust(0.02, 0.04, 0.98, 0.96)
    save(fig, name)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['svg.hashsalt'] = 'picaso'   # stable ids, so reruns don't churn git
    draw('reflected', *reflected(), BLUE)
    draw('thermal', *thermal(), MAGENTA, logx=True)
    draw('transmission', *transmission(), RED, logx=True)
    draw_pt('climate', ['#f2b134', '#ec9a2c', ORANGE, '#d45f27', '#c44326'])
    crescents('phases')
    draw('fit', *eclipse(), TEAL, data=True)
