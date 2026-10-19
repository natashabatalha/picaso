"""
The model-builder sidebar on the card pages (Spectrum, 1D climate).

One step per model component (star, object, P-T, chemistry, ...) sits between a
Setup and an Export step, above a planet badge that gains a layer as each
component is done, so students see what they are building.

A step is done once the student has edited one of its cards or run its preview;
the defaults alone do not count. A step that is a model run (the spectrum, the
climate model) is done once it has been run. Optional steps show as switched off
until enabled. Which step is in view is tracked in the browser (static/app.js),
not here.

Each page describes its steps with a Builder subclass (SpectrumBuilder below,
ClimateBuilder in views/climate.py) and passes `builder.state(...)` to
templates/_cards.html, which adds the sidebar.
"""
from dataclasses import dataclass

import picaso.driver as go
from picaso.driver_ui.core.config_ops import is_irradiated


@dataclass(frozen=True)
class Step:
    key: str  # also picks the step's icon (templates/_builder.html) and colour (--step-<key> in style.css)
    label: str
    summary: str  # completes "<label>: ..." in the step's tooltip
    cards: tuple  # names of the page's cards that belong to this step
    bookend: bool = False  # Setup and Export frame the model steps but are not part of the planet
    toggle: str = None  # id of the switch that turns an optional step on; clicking the switched-off step jumps to it


# the badge's caption for each kind of calculation in go.OBSERVATION_CALC_MAP
GEOMETRY_LABELS = {"transmission": "Transmission", "thermal": "Emission", "reflected": "Reflected"}


class Builder:
    """
    A page's steps. Subclasses set `steps` and `run_step` (the step that is done
    once its model has been run, and lights up the badge's spectrum layers), and
    say where the page keeps its state.
    """
    steps = ()
    run_step = None

    def __init__(self):
        self.step_of_card = {card: step.key for step in self.steps for card in step.cards}

    # ---------- the page's state; override these ----------
    def store(self, sess):
        """The page's state dict, which holds the set of touched cards."""
        raise NotImplementedError

    def enabled(self, sess, key):
        return True

    def optional(self, sess, key):
        """Whether to label a switched-off step "optional"."""
        return self.step(key).toggle is not None

    def ran(self, sess):
        """Whether `run_step`'s model has been run."""
        return False

    def kinds(self, sess):
        """The kinds of calculation the badge shows, a subset of GEOMETRY_LABELS."""
        return {"thermal"}

    def badge_label(self, sess):
        kinds = self.kinds(sess)
        return " + ".join(GEOMETRY_LABELS[k] for k in GEOMETRY_LABELS if k in kinds) or "Spectrum"

    def subs(self, sess):
        """{name: done} for a step drawn with a dot per sub-task (the retrieval), or None."""
        return None

    def done(self, sess, key):
        if key == self.run_step:
            return self.ran(sess)
        return any(card in self.touched(sess) for card in self.step(key).cards)

    # ---------- recording progress ----------
    def touched(self, sess):
        """Names of the cards the student has edited or run something on."""
        return self.store(sess).setdefault("touched", set())

    def touch(self, sess, card_name):
        self.touched(sess).add(card_name)

    def step(self, key):
        return next(s for s in self.steps if s.key == key)

    # ---------- state for the template ----------
    def state(self, sess, card_views, error_card=None):
        """
        What templates/_builder.html needs. `card_views` are the page's rendered
        cards (views/cards.py): a card showing an error note marks its step.
        `error_card` is a card whose submitted values failed validation.
        """
        with_errors = {self.step_of_card.get(view["card"].name) for view in card_views
                       if any(level == "error" for level, _ in view["notes"])}
        with_errors.add(self.step_of_card.get(error_card))

        steps = []
        for step in self.steps:
            on = self.enabled(sess, step.key)
            steps.append({
                "step": step,
                "state": "off" if not on else "done" if self.done(sess, step.key) else "todo",
                "optional": self.optional(sess, step.key),
                "error": on and step.key in with_errors,
            })
        return {
            "steps": steps,
            "subs": self.subs(sess),
            "card_steps": self.step_of_card,
            "run_step": self.run_step,
            "layers": self._layers({s["step"].key: s["state"] for s in steps}, self.kinds(sess),
                                   self.enabled(sess, "star")),
            "geometry_label": self.badge_label(sess),
        }

    def _layers(self, states, kinds, star_on):
        """{badge layer id: "off" | "ghost" | "on"}. The star is drawn dashed until its step is done; the rest fade in."""
        def solid(key, shown=True):
            return "on" if shown and states.get(key) == "done" else "off"

        star = "on" if states.get("star") == "done" else "ghost"
        transit = "transmission" in kinds
        return {
            "star-transit": star if star_on and transit else "off",
            "star-side": star if star_on and not transit else "off",
            "object": solid("object"),
            "pt": solid("pt"),
            "chemistry": solid("chemistry"),
            "clouds": solid("clouds"),
            "lit": "on" if star_on and "reflected" in kinds and states.get("object") == "done" else "off",
            "spec-transit": solid(self.run_step, transit),
            "spec-emit": solid(self.run_step, "thermal" in kinds),
            "spec-reflect": solid(self.run_step, "reflected" in kinds),
            "retrieval": solid("retrieval"),
        }


# =======================================
# SPECTRUM PAGE
# =======================================
RETRIEVAL_CARDS = ("observation_data", "free_parameters", "priors", "sampler", "prior_test")


class SpectrumBuilder(Builder):
    steps = (
        Step("setup", "Setup", "opacities and the kind of observation you are modeling.",
             ("upload", "optical", "setup"), bookend=True),
        Step("star", "Star", "the host star that lights the planet. Skip it for brown dwarfs and free-floating planets.",
             ("star",), toggle="f-irradiated"),
        Step("object", "Object", "bulk properties of the planet: mass, radius and gravity.", ("object",)),
        Step("pt", "P–T", "how temperature changes with pressure, from the deep atmosphere to the top.",
             ("temperature",)),
        Step("chemistry", "Chemistry", "which molecules are in the atmosphere, and how much of each.", ("chemistry",)),
        Step("clouds", "Clouds", "condensate layers that block or scatter light.", ("clouds",),
             toggle="f-ui-include_clouds"),
        Step("spectrum", "Spectrum", "the wavelength range and resolution of the computed spectrum.",
             ("geometry", "spectrum")),
        Step("retrieval", "Retrieval", "fit the model to data: free parameters, priors and sampler.",
             ("retrieval",) + RETRIEVAL_CARDS, toggle="f-ui-retrieval"),
        Step("export", "Export", "download the driver TOML and run this model from Python.",
             ("configuration",), bookend=True),
    )
    run_step = "spectrum"

    def store(self, sess):
        return sess.ui

    def enabled(self, sess, key):
        if key == "star":
            return is_irradiated(sess.config)
        if key == "clouds":
            return sess.ui["include_clouds"]
        if key == "retrieval":
            return sess.ui["retrieval"]
        return True

    def optional(self, sess, key):
        # a star is needed for anything but thermal emission
        return super().optional(sess, key) and not (key == "star" and sess.config["observation_type"] != "thermal")

    def ran(self, sess):
        return "spectrum" in sess.results

    def kinds(self, sess):
        """The kinds of calculation in the observation type, e.g. {"transmission", "thermal"} for transit_depth+thermal."""
        kinds = {go.OBSERVATION_CALC_MAP.get(part) for part in sess.config["observation_type"].split("+")}
        return kinds & set(GEOMETRY_LABELS)

    def subs(self, sess):
        return self.retrieval_progress(sess) if sess.ui["retrieval"] else None

    def retrieval_progress(self, sess):
        """{retrieval card name: done}. The prior test counts once its samples have been drawn."""
        edited = self.touched(sess)
        progress = {name: name in edited for name in RETRIEVAL_CARDS}
        progress["prior_test"] = "prior_samples" in sess.results
        return progress

    def done(self, sess, key):
        if key == "retrieval":
            return all(self.retrieval_progress(sess).values())
        return super().done(sess, key)


SPECTRUM = SpectrumBuilder()
