"""
Information Content computations: Jacobians, per-case statistics (SVD degrees
of freedom, Shannon information) and the dashboard figures.
"""
import copy
import os

import numpy as np
import plotly.graph_objects as go

from picaso import information_content as ic
from picaso import justdoit as jdi

EXAMPLE_PARAMS = ["cto_absolute", "log_mh", "fsed", "Teq", "phase"]


def example_jacobian():
    """Jacobian of a Jupiter-like reflected light case with chemical equilibrium and virga clouds (slow)."""
    refdata = jdi.__refdata__
    config = {
        "OpticalProperties": {"opacity_file": os.path.join(refdata, "opacities", "opacities_0.3_15_R15000.db"),
                              "opacity_kwargs": {"wave_range": [0.3, 2.0]},
                              "opacity_method": "resampled",
                              "virga_mieff": os.path.join(refdata, "virga/")},
        "calc_type": "spectrum",
        "irradiated": True,
        "geometry": {"phase": {"unit": "radian", "value": np.pi / 2},
                     "phase_kwargs": {"num_tangle": 6, "num_gangle": 6}},
        "object": {"distance": {"unit": "parsec", "value": 8.3},
                   "gravity": {"unit": "cm/s**2", "value": 100000.0},
                   "mass": {"unit": "Mjup", "value": 1.2},
                   "radius": {"unit": "Rjup", "value": 1.2},
                   "teff": {"unit": "Kelvin", "value": 5400},
                   "teq": {"unit": "Kelvin", "value": 500}},
        "observation_type": "albedo",
        "star": {"grid": {"database": "ck04models", "feh": 0, "logg": 4, "teff": 5400},
                 "radius": {"unit": "Rsun", "value": 1},
                 "semi_major": {"unit": "AU", "value": 200}},
        "temperature": {
            "profile": "guillot",
            "pressure": {"reference": {"value": 1e1, "unit": "bar"}, "min": {"value": 1e-5, "unit": "bar"},
                         "max": {"value": 1e3, "unit": "bar"}, "nlevel": 60, "spacing": "log"},
            "guillot": {"T_int": 100, "Teq": 200, "alpha": 0.5, "logKir": -1.5, "logg1": -1},
        },
        "chemistry": {"method": "chemeq_on_the_fly", "chemeq_on_the_fly": {"cto_absolute": 0.55, "log_mh": 2}},
        "clouds": {"cloud1_type": "virga",
                   "cloud1": {"virga": {"condensates": ["H2O"], "fsed": 2, "kzz": 1e8, "mh": 100,
                                        "mmw": 2.2, "sig": 2}}},
    }
    spectrum = ic.run(driver_dict=config)
    return {"wno": spectrum["wavenumber"], "jacobian": ic.jacobian(driver_dict=config, params=EXAMPLE_PARAMS),
            "params": list(EXAMPLE_PARAMS)}


def load_npz(file):
    """Jacobian data from an uploaded .npz with 'wno', 'jacobian' and 'params' arrays."""
    data = np.load(file)
    if not all(key in data for key in ("wno", "jacobian", "params")):
        raise ValueError("Uploaded file must contain 'wno', 'jacobian', and 'params' keys.")
    return {"wno": data["wno"], "jacobian": data["jacobian"], "params": [str(p) for p in data["params"]]}


def analyzer_for(case, wno, jacobian):
    """ic.Analyze for one observation case (a manual grid, or wavelength/error columns from a CSV)."""
    if case["method"] == "Manual":
        new_wno = ic.create_grid(case["min_wave"], case["max_wave"], case["res"])
        return ic.Analyze(wno, jacobian, case["error"], new_wno=new_wno)

    df = case["csv"]
    wavelength, error = df["wavelength"].values, df["error"].values
    # ic.Analyze expects the error on the original wno grid for rebinning; np.interp needs increasing x
    xp, fp = 1e4 / wavelength[::-1], error[::-1]
    if np.all(np.diff(wno) < 0):
        interpolated_error = np.interp(wno[::-1], xp, fp)[::-1]
    else:
        interpolated_error = np.interp(wno, xp, fp)
    new_wno = 1e4 / wavelength
    order = np.argsort(new_wno)[::-1]  # ic.Analyze wants descending wno
    analyzer = ic.Analyze(wno, jacobian, interpolated_error, new_wno=new_wno[order])
    analyzer.error = error[order]  # use exactly the errors from the CSV
    return analyzer


def analyze(cases, data, priors):
    """Statistics for every case. Returns {names, svd, shannon, loss_h, loss_ci, analyzers}."""
    results = {"names": [], "svd": [], "shannon": [], "loss_h": [], "loss_ci": [], "analyzers": []}
    for case in cases:
        analyzer = analyzer_for(case, data["wno"], data["jacobian"])
        results["names"].append(case["name"])
        results["svd"].append(analyzer.degrees_of_freedom_svd()[0])  # (dfs, s, vh)
        results["shannon"].append(analyzer.shannon_ic(priors))
        loss_h, loss_ci = analyzer.loss_by_wave()  # uses the prior set by shannon_ic
        results["loss_h"].append(loss_h)
        results["loss_ci"].append(loss_ci)
        results["analyzers"].append(copy.copy(analyzer))
    return results


# =======================================
# FIGURES
# =======================================
def _case_wno(analyzer, wno):
    return analyzer.new_wno if analyzer.new_wno is not None else wno


def jacobian_figure(data):
    """Normalized |Jacobian| of each parameter on the native wavelength grid."""
    wavelength = 1e4 / np.asarray(data["wno"])
    jacobian = np.asarray(data["jacobian"])  # (N_wavelengths, N_parameters)
    fig = go.Figure()
    for i, param in enumerate(data["params"]):
        y = np.abs(jacobian[:, i])
        peak = np.max(y)
        fig.add_trace(go.Scatter(x=wavelength, y=y / peak if peak != 0 else y, name=param, mode="lines"))
    fig.update_layout(xaxis_title="Wavelength [um]", yaxis_title="Normalized |Jacobian|", title="Jacobian",
                      legend_title="Parameters")
    return fig


def case_figures(results, params, wno, index):
    """Binned Jacobian and constraint-interval loss for one case."""
    name, analyzer = results["names"][index], results["analyzers"][index]
    wavelength = 1e4 / _case_wno(analyzer, wno)

    jacobian_fig = go.Figure()
    for i, param in enumerate(params):
        y = np.array(analyzer.jacobian[i, :]).flatten()
        peak = np.max(np.abs(y))
        jacobian_fig.add_trace(go.Scatter(x=wavelength, y=np.abs(y) / peak if peak != 0 else y, name=param, mode="lines"))
    jacobian_fig.update_layout(xaxis_title="Wavelength [um]", yaxis_title="Normalized |Jacobian|",
                               title=f"Binned Jacobian for {name}", legend_title="Parameters")

    loss_ci_fig = go.Figure()
    for i, param in enumerate(params):
        loss_ci_fig.add_trace(go.Scatter(x=wavelength, y=np.array(results["loss_ci"][index])[:, i], mode="lines",
                                         name=param))
    loss_ci_fig.update_layout(xaxis_title="Wavelength [um]", yaxis_title="Delta Constraint Interval/um",
                              title=f"Loss in 1-sigma Constraint Interval vs. W for {name}")
    return jacobian_fig, loss_ci_fig


def comparison_figures(results, params, wno):
    """{name: figure} comparing all cases."""
    names = results["names"]
    loss_h = go.Figure()
    for name, loss, analyzer in zip(names, results["loss_h"], results["analyzers"]):
        loss_h.add_trace(go.Scatter(x=1e4 / _case_wno(analyzer, wno), y=loss, mode="lines", name=name))
    loss_h.update_layout(xaxis_title="Wavelength [um]", yaxis_title="Delta IC/um", title="H loss vs. W (All Cases)")

    def per_case(values, title, yaxis, marker):
        fig = go.Figure(go.Scatter(x=names, y=values, mode="lines+markers", name=title, marker=marker))
        fig.update_layout(xaxis_title="Case", yaxis_title=yaxis, title=title)
        return fig

    def per_param(key, title, yaxis):
        fig = go.Figure()
        for i, param in enumerate(params):
            fig.add_trace(go.Scatter(x=names, y=[r[key][i] for r in results["shannon"]], mode="lines+markers",
                                     name=param, marker=dict(size=6)))
        fig.update_layout(xaxis_title="Case", yaxis_title=yaxis, title=title)
        return fig

    return {
        "loss_h": loss_h,
        "dof": per_case([r["DOF"] for r in results["shannon"]], "Shannon DOF", "DOF",
                        dict(symbol="square", color="green")),
        "h": per_case([r["H"] for r in results["shannon"]], "Shannon Information (H)", "H [bits]",
                      dict(symbol="triangle-up", color="red")),
        "averaging_kernel": per_param("AveragingKernel", "Averaging Kernel vs. Case", "Averaging Kernel Diagonal"),
        "constraint_interval": per_param("constraint_interval", "1-sigma Constraint Interval vs. Case",
                                         "Constraint Interval"),
        "svd": per_case(results["svd"], "SVD Degrees of Freedom for Signal", "SVD DFS",
                        dict(symbol="circle", color="blue")),
    }
