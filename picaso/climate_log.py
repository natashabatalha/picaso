"""
Progress reporting for the 1D climate solver (climate.py).

A climate run gives two independent kinds of progress:

- the printout (``verbose``): a readable account of each stage of the solver,
  one line per outer iteration (``verbose=True``), plus every Newton step
  (``verbose=2``);
- the watch stream (``watch``): structured events (dicts) sent to one or more
  callables while the run is going, e.g. to draw the profile as it converges
  (``watch='plot'``), write a log file that can be followed from another
  process (``watch='climate_log.jsonl'``), or feed a web page (the PICASO UI).

With ``verbose=False`` and no ``watch``, every hook returns after a single
attribute check, so nothing is formatted, copied or stored.

Usage::

    out = cl.climate(opa)                              # standard printout
    out = cl.climate(opa, verbose=2)                   # also every Newton step
    out = cl.climate(opa, verbose=False, watch='plot') # live plot only
    events = []
    out = cl.climate(opa, watch=events.append)         # keep every event
"""
import json
import time

import numpy as np

# =======================================
# NEWTON DIAGNOSTICS
# =======================================
# climate.newton_raphson_solver fills one row of this table per Newton step
NEWTON_COLUMNS = ("step", "tmin", "tmax", "flux_err", "lambda", "dt_rel", "n_clipped", "capped", "status")
N_NEWTON_COLUMNS = len(NEWTON_COLUMNS)
# how a Newton step ended (the `status` column)
NEWTON_CONTINUE = 0  # took a step, not converged yet
NEWTON_CONVERGED = 1  # flux residual or temperature change below tolerance
NEWTON_STALLED = 2  # the line search could not lower the residual further; accepted as converged
NEWTON_AT_ROOT = 3  # the profile already balanced the fluxes, no step taken
NEWTON_NAN = 4  # temperature went NaN and was reset to the previous step

NEWTON_STATUS = {NEWTON_CONTINUE: "max", NEWTON_CONVERGED: "ok", NEWTON_STALLED: "stall",
                 NEWTON_AT_ROOT: "root", NEWTON_NAN: "nan"}


def new_newton_diagnostics(max_steps):
    """Table for climate.t_start to fill, one row per Newton step (unused rows stay NaN)."""
    return np.full((max(int(max_steps), 1), N_NEWTON_COLUMNS), np.nan)


def newton_rows(diagnostics):
    """The filled rows of a Newton diagnostics table, as a list of dicts."""
    if diagnostics is None:
        return []
    rows = diagnostics[~np.isnan(diagnostics[:, 0])]
    return [dict(zip(NEWTON_COLUMNS, row)) for row in rows.tolist()]


def convective_zones(nstr, nofczns):
    """[(top level, bottom level)] of each convective zone described by the nstr vector."""
    return [(int(nstr[3 * k + 1]), int(nstr[3 * k + 2]) + 1) for k in range(int(nofczns))]


# =======================================
# REPORTER
# =======================================
RULE = "=" * 78
THIN_RULE = "-" * 78
HOW_TO_READ = """\
 How to read this log: the solver works in STAGES. In each stage the convective
 zone(s) are held fixed and the solver repeats outer iterations until the T(P)
 profile stops changing. One outer iteration = a Newton solve of T(P) with
 opacities frozen, then chemistry, opacities (and kzz/clouds) are recomputed.
   newton   Newton steps used/allowed, and how the solve ended:
            ok=converged, stall=line search could not improve, max=hit the limit
   T range  coldest-hottest temperature of the current profile [K]
   flux err largest net-flux imbalance in a radiative layer, in units of sigma*Teff^4
   dT       mean |T_new - T_old| / 1.5 [K]; a stage converges when dT < tol
   notes    warning signs: T clipped to the opacity grid, small line search
            steps, dT growing, NaNs
 verbose=2 also prints every Newton step; verbose=False silences this log."""


class ClimateReporter:
    """
    Prints and streams the progress of one climate run. The solver calls its hooks
    (start, stage, profile_start, iteration, profile_end, convection, finish);
    each returns immediately when ``verbose`` is off and nothing is watching.

    Parameters
    ----------
    verbose : bool or int
        False/0 prints nothing, True/1 prints stages and outer iterations,
        2 (or 'debug') also prints every Newton step and the messages of the
        chemistry, kzz and cloud updates.
    watch : None, str, callable, or list of those
        Where to send events while the run is going. 'plot' draws a live
        figure (LiveClimatePlot), a path ending in .jsonl appends one JSON
        event per line (JSONLinesLog), and a callable is called with each
        event dict.
    """

    def __init__(self, verbose=True, watch=None):
        if verbose == "debug":
            verbose = 2
        self.level = int(verbose) if verbose else 0
        self.sinks = _make_sinks(watch)
        self.on = bool(self.level or self.sinks)
        self.pressure = None
        self.stage_number = 0
        self.n_stages = None
        self.stage_title = ""
        self.outer_total = 0
        self.newton_total = 0
        self.stage_dts = []
        self.last_newton = []
        self.zones = []
        self._header_printed = False
        self._t0 = time.time()

    @property
    def debug(self):
        """True if the solver's internal messages (chemistry, kzz, clouds) should print."""
        return self.level >= 2

    # ---------- output ----------
    def say(self, text, level=1):
        if self.level >= level:
            print(text, flush=True)
        if self.sinks:
            self.emit("log", text=text, level=level)

    def emit(self, event, **data):
        data["event"] = event
        data["elapsed"] = time.time() - self._t0
        for sink in self.sinks:
            sink(data)

    # ---------- hooks ----------
    def start(self, pressure, temperature, summary=(), n_stages=None):
        """Beginning of the run. `summary` is a list of 'label: value' strings describing the setup."""
        if not self.on:
            return
        self._t0 = time.time()
        self.pressure = np.asarray(pressure, dtype=float)
        self.n_stages = n_stages
        self.say(RULE)
        self.say(" PICASO 1D climate | " + " | ".join(summary[:4]))
        if len(summary) > 4:
            self.say(" " + " | ".join(summary[4:]))
        self.say(RULE)
        self.say(HOW_TO_READ)
        self.say(RULE)
        if self.sinks:
            self.emit("start", pressure=self.pressure.copy(), temperature=np.array(temperature, dtype=float),
                      summary=list(summary))

    def stage(self, title, description=None):
        """A new stage of the solver (a series of profile calls with one purpose)."""
        if not self.on:
            return
        self.stage_number += 1
        self.stage_title = title
        self.stage_dts = []
        self._header_printed = False
        of = f"/{self.n_stages}" if self.n_stages else ""
        self.say("")
        self.say(f"STAGE {self.stage_number}{of}  {title}")
        if description:
            self.say(f"   {description}")
        if self.sinks:
            self.emit("stage", stage=self.stage_number, title=title, description=description)

    def profile_start(self, convergence_criteria, nstr, nofczns):
        """A profile() call begins: its tolerances and convective zones."""
        if not self.on:
            return
        self.tol = convergence_criteria.outer_threshold
        self.max_outer = convergence_criteria.max_outer_iterations
        self.max_newton = convergence_criteria.max_inner_iterations
        self.prev_dt = None
        self.zones = convective_zones(nstr, nofczns)
        self.say(f"   zones: {self.describe_zones(nstr, nofczns)}   "
                 f"(tol dT < {self.tol:g} K, up to {self.max_outer} iterations x {self.max_newton} Newton steps)")
        if not self._header_printed:
            self.say("    iter  newton      T range [K]   flux err    dT [K]   time  notes")
            self._header_printed = True

    def iteration(self, outer, temperature, dt, newton_diagnostics, nstr, nofczns,
                  converged=False, taudif=None, taudif_tol=None):
        """One outer iteration of profile(): a Newton solve then a chemistry/opacity update."""
        if not self.on:
            return
        self.outer_total += 1
        rows = newton_rows(newton_diagnostics)
        self.newton_total += len(rows)
        self.last_newton = rows
        self.stage_dts.append(dt)

        last = rows[-1] if rows else {}
        status = NEWTON_STATUS.get(int(last.get("status", NEWTON_CONTINUE)), "?")
        if status == "max" and len(rows) < self.max_newton:
            status = "?"
        newton = f"{len(rows):>2}/{self.max_newton:<2} {status:<5}"
        flux_err = last.get("flux_err", np.nan)
        notes = self._notes(rows, dt, taudif, taudif_tol)

        if self.debug:
            for row in rows:
                self.say(f"         newton step {int(row['step']):>2}: T {row['tmin']:7.1f}-{row['tmax']:<7.1f} "
                         f"flux err {row['flux_err']:.2e}  line-search step {row['lambda']:.2f}  "
                         f"max|dT|/T {row['dt_rel']:.1e}  [{NEWTON_STATUS.get(int(row['status']), '?')}]", level=2)
        done = "  <- converged" if converged else ""
        self.say(f"    {outer + 1:>4}  {newton}  {np.min(temperature):6.0f} - {np.max(temperature):<6.0f} "
                 f"{flux_err:9.2e}  {dt:8.2f}  {self._clock():>5}  {', '.join(notes)}{done}")
        self.prev_dt = dt

        if self.sinks:
            self.emit("iteration", stage=self.stage_number, stage_title=self.stage_title,
                      iteration=outer + 1, total_iteration=self.outer_total,
                      temperature=np.array(temperature, dtype=float), dt=float(dt), tol=float(self.tol),
                      converged=bool(converged), flux_err=float(flux_err),
                      newton_steps=len(rows), newton_status=status, newton=rows,
                      zones=convective_zones(nstr, nofczns), notes=notes,
                      taudif=None if taudif is None else float(taudif))

    def profile_end(self, converged, n_iterations):
        """A profile() call ends."""
        if not self.on:
            return
        if converged:
            self.say(f"   -> converged after {n_iterations} iterations")
        else:
            last = f" (dT {self.prev_dt:.2f} K > tol {self.tol:g} K)" if self.prev_dt is not None else ""
            self.say(f"   -> stopped at the iteration limit{last}; moving on")

    def convection(self, message, nstr, nofczns):
        """The convective zones changed."""
        if not self.on:
            return
        self.zones = convective_zones(nstr, nofczns)
        self.say(f"   * {message} -> zones: {self.describe_zones(nstr, nofczns)}")
        if self.sinks:
            self.emit("convection", message=message, zones=convective_zones(nstr, nofczns))

    def finish(self, converged, temperature):
        """End of the run."""
        if not self.on:
            return
        verdict = "CONVERGED" if converged else "NOT CONVERGED"
        self.say("")
        self.say(RULE)
        self.say(f" {verdict} | {self.outer_total} outer iterations, {self.newton_total} Newton steps, "
                 f"{self._clock()}")
        self.say(f" final zones: {self._describe(self.zones)} | "
                 f"T range {np.min(temperature):.0f}-{np.max(temperature):.0f} K")
        if not converged:
            for hint in self._hints():
                self.say(f" hint: {hint}")
        self.say(RULE)
        if self.sinks:
            self.emit("finish", converged=bool(converged), temperature=np.array(temperature, dtype=float),
                      zones=self.zones, outer_iterations=self.outer_total,
                      newton_steps=self.newton_total)

    # ---------- helpers ----------
    def describe_zones(self, nstr, nofczns):
        return self._describe(convective_zones(nstr, nofczns))

    def _describe(self, zones):
        if not zones:
            return "none"
        if self.pressure is None:
            return ", ".join(f"levels {top}-{bottom}" for top, bottom in zones)
        last = len(self.pressure) - 1
        return ", ".join(f"levels {top}-{bottom} ({self.pressure[min(top, last)]:.3g}-"
                         f"{self.pressure[min(bottom, last)]:.3g} bar)" for top, bottom in zones)

    def _clock(self):
        seconds = int(time.time() - self._t0)
        return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"

    def _notes(self, rows, dt, taudif, taudif_tol):
        notes = []
        if any(int(r["status"]) == NEWTON_NAN for r in rows):
            notes.append("NaN T reset")
        clipped = max((int(r["n_clipped"]) for r in rows), default=0)
        if clipped:
            notes.append(f"{clipped} lvl clipped to T grid")
        small = [r["lambda"] for r in rows if r["lambda"] < 0.25 and int(r["status"]) != NEWTON_AT_ROOT]
        if small:
            notes.append(f"line search x{min(small):.2f}")
        if self.prev_dt is not None and dt > self.prev_dt:
            notes.append("dT grew")
        if taudif is not None and taudif_tol is not None and taudif > 0:
            notes.append(f"cloud dtau {taudif:.2g}" + (" > tol" if taudif >= taudif_tol else ""))
        return notes

    def _hints(self):
        hints = ["the final stage hit its iteration limit. A closer initial guess (e.g. a Sonora profile), "
                 "a different rcb_guess, or a finer pressure grid often help."]
        dts = self.stage_dts[-6:]
        if len(dts) >= 4 and all((b - a) * (c - b) < 0 for a, b, c in zip(dts, dts[1:], dts[2:])):
            hints.append("dT alternates up and down, a sign of an oscillation between two states; "
                         "for disequilibrium chemistry try damping=True.")
        if any(int(r["n_clipped"]) for r in self.last_newton):
            hints.append("temperatures were clipped to the edge of the opacity grid; check that the "
                         "opacity tables cover the temperatures of this object.")
        return hints


# =======================================
# WATCHERS
# =======================================
def _make_sinks(watch):
    if watch is None or watch is False:
        return []
    if isinstance(watch, (list, tuple)):
        return [sink for w in watch for sink in _make_sinks(w)]
    if watch is True or watch == "plot":
        return [LiveClimatePlot()]
    if isinstance(watch, str):
        return [JSONLinesLog(watch)]
    if callable(watch):
        return [watch]
    raise ValueError(f"Could not interpret watch={watch!r}. Use 'plot', a .jsonl file path, or a callable.")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class JSONLinesLog:
    """
    Appends each event as one line of JSON, flushed immediately, so the run can be
    followed from another terminal (``tail -f``) or process (``read_climate_log``).
    """

    def __init__(self, path, mode="w"):
        self.path = path
        open(path, mode).close()

    def __call__(self, event):
        with open(self.path, "a") as f:
            f.write(json.dumps(_jsonable(event)) + "\n")


def read_climate_log(path):
    """Events written by JSONLinesLog (works while the run is still going)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class ConvergenceHistory:
    """
    Keeps what is needed to draw the convergence of a run from its events: the
    pressure grid, the guess, every outer-iteration profile and its dT.
    Used by LiveClimatePlot and the PICASO UI.
    """

    def __init__(self):
        self.pressure = None
        self.guess = None
        self.profiles = []  # (total iteration, temperature)
        self.dts = []  # (total iteration, dT, tol, stage)
        self.stages = []  # (total iteration at which the stage began, title)
        self.zones = []
        self.log = []
        self.finished = None

    def __call__(self, event):
        kind = event["event"]
        if kind == "start":
            self.pressure, self.guess = event["pressure"], event["temperature"]
        elif kind == "stage":
            self.stages.append((len(self.dts), event["title"]))
        elif kind == "iteration":
            self.profiles.append((event["total_iteration"], event["temperature"]))
            self.dts.append((event["total_iteration"], event["dt"], event["tol"], event["stage"]))
            self.zones = event["zones"]
        elif kind == "convection":
            self.zones = event["zones"]
        elif kind == "log":
            self.log.append(event["text"])
        elif kind == "finish":
            self.finished = event


class LiveClimatePlot(ConvergenceHistory):
    """
    Live figure of a climate run: every outer-iteration T(P) profile (older ones
    fade) with the convective zones shaded, and dT per iteration against each
    stage's tolerance. Redraws in place in a Jupyter notebook; elsewhere it
    opens a matplotlib window.
    """

    def __init__(self, figsize=(11, 5)):
        super().__init__()
        self.figsize = figsize
        self.fig = None
        self._handle = None

    def __call__(self, event):
        super().__call__(event)
        if event["event"] in ("iteration", "convection", "finish"):
            self.draw()

    def draw(self):
        import matplotlib.pyplot as plt
        if self.fig is None:
            self.fig, (self.ax_pt, self.ax_dt) = plt.subplots(1, 2, figsize=self.figsize)
        draw_convergence(self, self.ax_pt, self.ax_dt)
        self.fig.tight_layout()
        self._show()

    def _show(self):
        try:
            from IPython import get_ipython
            from IPython.display import display
            in_notebook = get_ipython() is not None and "IPKernelApp" in get_ipython().config
        except ImportError:
            in_notebook = False
        if in_notebook:
            import matplotlib.pyplot as plt
            if self._handle is None:
                self._handle = display(self.fig, display_id=True)
                plt.close(self.fig)  # stop the notebook from drawing it a second time at the end of the cell
            else:
                self._handle.update(self.fig)
        else:
            import matplotlib.pyplot as plt
            plt.pause(0.01)


def draw_convergence(history, ax_pt, ax_dt):
    """Draws a ConvergenceHistory on two matplotlib axes."""
    ax_pt.clear()
    ax_dt.clear()
    pressure = history.pressure
    if pressure is not None:
        if history.guess is not None:
            ax_pt.semilogy(history.guess, pressure, "k--", lw=1, label="initial guess")
        n = len(history.profiles)
        for i, (_, temperature) in enumerate(history.profiles[:-1]):
            ax_pt.semilogy(temperature, pressure, color="tab:blue", lw=1, alpha=0.1 + 0.4 * (i + 1) / n)
        if history.profiles:
            k, temperature = history.profiles[-1]
            label = "converged" if history.finished else f"iteration {k}"
            ax_pt.semilogy(temperature, pressure, color="tab:red", lw=2.5, label=label)
        last = len(pressure) - 1
        for top, bottom in history.zones:
            ax_pt.axhspan(pressure[min(top, last)], pressure[min(bottom, last)], color="tab:orange", alpha=0.15,
                          lw=0, label="convective" if (top, bottom) == tuple(history.zones[0]) else None)
        ax_pt.set_ylim(pressure.max(), pressure.min())
        ax_pt.set_xlabel("Temperature [K]")
        ax_pt.set_ylabel("Pressure [bar]")
        ax_pt.legend(loc="upper right", fontsize=8)

    if history.dts:
        k, dt, tol, stage = map(np.array, zip(*history.dts))
        ax_dt.semilogy(k, np.maximum(dt, 1e-3), "o-", color="tab:blue", ms=4, label="dT")
        ax_dt.step(k, tol, where="mid", color="tab:green", ls="--", label="stage tolerance")
        for start, title in history.stages:
            ax_dt.axvline(start + 0.5, color="0.7", lw=0.8)
        ax_dt.set_xlabel("outer iteration")
        ax_dt.set_ylabel("mean |dT| / 1.5 [K]")
        ax_dt.legend(loc="upper right", fontsize=8)
        title = history.stages[-1][1] if history.stages else ""
        if history.finished:
            title = "CONVERGED" if history.finished["converged"] else "NOT CONVERGED"
        ax_dt.set_title(title, fontsize=10)
