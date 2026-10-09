"""Retrieval cards of the Spectrum & Retrieval Setup page, and the actions behind their buttons."""
import picaso.driver as go
from picaso.driver_ui.core import runs
from picaso.driver_ui.core.config_schema import Bool, Choice, Number, Section, Text, TextList, build_fields
from picaso.driver_ui.core.plots import plotly_plot, prior_sample_figures
from picaso.driver_ui.core.retrieval_setup import (
    PRIOR_TYPES, SAMPLER_CODES, SYSTEMATICS, SYSTEMATICS_LABELS, data_name, default_prior, find_prior,
    free_parameters, parse_kwargs)
from picaso.driver_ui.views.cards import Card, set_messages
from picaso.driver_ui.views.spectrum_session import (
    model_config, retrieval_base_config, retrieval_priors, selected_parameters, wave_range)


def sync_priors(sess):
    """Keeps one prior per selected parameter: new selections get defaults, deselected ones are dropped."""
    config_priors = sess.config.get("retrieval")
    sess.ui["priors"] = {
        key: sess.ui["priors"].get(key) or default_prior(value, find_prior(config_priors, key))
        for key, value in selected_parameters(sess).items()
    }


class RetrievalCard(Card):
    """Base for the cards shown once retrievals are switched on."""
    clears_previews = False

    def visible(self, sess):
        return sess.ui["retrieval"]

    def after_update(self, sess):
        sync_priors(sess)


class RetrievalToggleCard(RetrievalCard):
    name, title = "retrieval", "Retrievals"

    def visible(self, sess):
        return True

    def fields(self, sess):
        return Section(("retrieval_card",), [
            Bool(("ui", "retrieval"), sess.ui["retrieval"], label="Do you want to do a retrieval?")])


class ObservationDataCard(RetrievalCard):
    name, title = "observation_data", "Observational Data Configuration"
    extra = "spectrum/extras/data_check.html"
    clears_previews = True

    def fields(self, sess):
        observations = sess.config["ObservationData"]
        filenames = observations["filenames"]
        children = [TextList(("ObservationData", "filenames"), filenames, multiline=True,
                             hint="Path(s) to your observation data, one per line")]
        other = {k: v for k, v in observations.items() if k not in ("filenames", "instruments")}
        children += build_fields(other, ("ObservationData",)).children

        instruments = go.get_instrument_options()
        current = observations.get("instruments", [])
        for i, filename in enumerate(filenames):
            name = data_name(filename)
            per_file = []
            if instruments:
                value = current[i] if i < len(current) and current[i] in instruments else None
                per_file.append(Choice(("ObservationData", "instruments", str(i)), value, [None, *instruments],
                                       label="Instrument", hint="Resolution file to convolve with"))
            per_file += [Bool(("ui", "systematics", kind, name), sess.ui["systematics"][kind].get(name, False),
                              label=SYSTEMATICS_LABELS[kind]) for kind in SYSTEMATICS]
            children.append(Section(("observation_file", name), per_file, label=name))
        return Section(("observation_card",), children)

    def notes(self, sess):
        return [("info", "Specify column, coord, or data_var names and units. Units are only required if not "
                         "specified through xarray, and should be in astropy format (enter unitless for e.g. "
                         "albedo or transit depth).")]

    def after_update(self, sess):
        observations = sess.config["ObservationData"]
        if go.get_instrument_options():
            count = len(observations["filenames"])
            observations["instruments"] = (observations.get("instruments", []) + [None] * count)[:count]
        sync_priors(sess)


class FreeParametersCard(RetrievalCard):
    name, title = "free_parameters", "Free Parameters"

    def fields(self, sess):
        groups = {}
        for key, value in free_parameters(retrieval_base_config(sess)).items():
            groups.setdefault(key.split(".")[0], []).append(
                Bool(("ui", "free", key), sess.ui["free"].get(key, False), label=key, hint=f"current value {value:g}"))
        return Section(("free_card",), [Section(("free_group", group), items, label=group)
                                        for group, items in groups.items()])

    def notes(self, sess):
        return [("info", "Select which available free parameters you'd like to do a retrieval on.")]


class PriorsCard(RetrievalCard):
    name, title = "priors", "Priors"

    def visible(self, sess):
        return sess.ui["retrieval"] and bool(sess.ui["priors"])

    def fields(self, sess):
        sections = []
        for key, prior in sess.ui["priors"].items():
            path = ("ui", "priors", key)
            items = [Choice(path + ("prior",), prior["prior"], PRIOR_TYPES), Bool(path + ("log",), prior["log"])]
            params = ("min", "max") if prior["prior"] == "uniform" else ("mean", "std")
            items += [Number(path + (param,), prior[param]) for param in params]
            sections.append(Section(path, items, label=key))
        return Section(("priors_card",), sections)

    def notes(self, sess):
        return [("error", f"{key}: min must be less than max") for key, p in sess.ui["priors"].items()
                if p["prior"] == "uniform" and p["min"] >= p["max"]]


class SamplerCard(RetrievalCard):
    name, title = "sampler", "Sampler Options"

    def fields(self, sess):
        sampler = sess.ui["sampler"]
        return Section(("sampler_card",), [
            Choice(("ui", "sampler", "code"), sampler["code"], SAMPLER_CODES, label="Bayesian code"),
            Text(("ui", "sampler", "sampler_kwargs"), sampler["sampler_kwargs"],
                 hint="Parsable dictionary, e.g. {'live_points': 700}"),
            Text(("ui", "sampler", "run_kwargs"), sampler["run_kwargs"],
                 hint="Parsable dictionary, e.g. {'max_iter': 10000}"),
            Text(("InputOutput", "retrieval_output"), sess.config["InputOutput"]["retrieval_output"],
                 hint="Retrieval run output path"),
        ])

    def notes(self, sess):
        notes = []
        for key in ("sampler_kwargs", "run_kwargs"):
            try:
                parse_kwargs(sess.ui["sampler"][key])
            except (ValueError, SyntaxError) as e:
                notes.append(("error", f"{key}: {e}"))
        return notes


class PriorTestCard(RetrievalCard):
    name, title = "prior_test", "Set and Test Your Prior Bounds"
    extra = "spectrum/extras/prior_test.html"

    def visible(self, sess):
        return sess.ui["retrieval"] and bool(sess.ui["priors"])

    def fields(self, sess):
        return Section(("prior_test_card",), [
            Number(("ui", "nsamples"), sess.ui["nsamples"], minimum=1, label="Number of samples")])

    def notes(self, sess):
        return [("info", "Run samples drawn from the prior ranges to visualize the chemistry, pressure-temperature "
                         "profiles, and spectra they produce.")]


CARDS = [RetrievalToggleCard(), ObservationDataCard(), FreeParametersCard(), PriorsCard(), SamplerCard(),
         PriorTestCard()]


# =======================================
# ACTIONS
# =======================================
def check_data(sess, with_reference=False):
    reference = sess.results.get("spectrum") if with_reference else None
    try:
        messages, fig = runs.check_data(model_config(sess), reference, wave_range(sess))
    except Exception as e:
        set_messages(sess.results, "observation_data", [("error", f"Error parsing data: {e}")])
        return
    sess.results["data_plot"] = plotly_plot(fig)
    set_messages(sess.results, "observation_data", messages)


def sample_priors(sess):
    try:
        configs, profiles = runs.sample_priors(retrieval_base_config(sess), retrieval_priors(sess), sess.ui["nsamples"])
        figures = prior_sample_figures(profiles, include_clouds=sess.ui["include_clouds"])
    except Exception as e:
        set_messages(sess.results, "prior_test", [("error", f"Could not sample the priors: {e}")])
        return
    sess.results["prior_samples"] = {"configs": configs, "plots": [plotly_plot(f) for f in figures]}


def sample_spectra(sess):
    samples = sess.results.get("prior_samples")
    if samples is None:
        return
    try:
        warnings, fig = runs.prior_sample_spectra(samples["configs"], retrieval_base_config(sess), wave_range(sess))
    except Exception as e:
        set_messages(sess.results, "prior_test", [("error", f"Could not run the sample spectra: {e}")])
        return
    samples["spectrum_plot"] = plotly_plot(fig)
    set_messages(sess.results, "prior_test", [("warning", w) for w in warnings])


ACTIONS = {
    "data-check": check_data,
    "data-check-reference": lambda sess: check_data(sess, with_reference=True),
    "sample": sample_priors,
    "sample-spectra": sample_spectra,
}
