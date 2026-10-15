"""
The model-builder strip at the top of the Spectrum page.

One step per model component (star, object, P-T, chemistry, clouds, spectrum,
retrieval) sits between a Setup and an Export step, next to a planet badge that
gains a layer as each component is done, so students see what they are building.

A step is done once the student has edited one of its cards or run its preview;
the defaults alone do not count. The spectrum is done once it has been run, and a
retrieval once each of its cards has been worked through. Star, clouds and
retrieval are optional: they show as switched off until enabled. Which step is in
view is tracked in the browser (static/app.js), not here.
"""
from dataclasses import dataclass

import picaso.driver as go
from picaso.driver_ui.core import clouds
from picaso.driver_ui.core.config_ops import is_irradiated


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    summary: str  # completes "<label>: ..." in the caption under the strip
    cards: tuple  # names of the cards (views/spectrum.py) that belong to this step
    bookend: bool = False  # Setup and Export frame the model steps but are not part of the planet


RETRIEVAL_CARDS = ("observation_data", "free_parameters", "priors", "sampler", "prior_test")

STEPS = (
    Step("setup", "Setup", "opacities and the kind of observation you are modeling.",
         ("upload", "optical", "setup"), bookend=True),
    Step("star", "Star", "the host star that lights the planet. Skip it for brown dwarfs and free-floating planets.",
         ("star",)),
    Step("object", "Object", "bulk properties of the planet: mass, radius and gravity.", ("object",)),
    Step("pt", "P–T", "how temperature changes with pressure, from the deep atmosphere to the top.",
         ("temperature",)),
    Step("chemistry", "Chemistry", "which molecules are in the atmosphere, and how much of each.", ("chemistry",)),
    Step("clouds", "Clouds", "condensate layers that block or scatter light.", ("clouds",)),
    Step("spectrum", "Spectrum", "the wavelength range and resolution of the computed spectrum.",
         ("geometry", "spectrum")),
    Step("retrieval", "Retrieval", "fit the model to data: free parameters, priors and sampler.",
         ("retrieval",) + RETRIEVAL_CARDS),
    Step("export", "Export", "download the driver TOML and run this model from Python.",
         ("configuration",), bookend=True),
)
STEP_OF_CARD = {card: step.key for step in STEPS for card in step.cards}

# id of the switch that turns each optional step on; clicking the switched-off step jumps to it
TOGGLES = {"star": "f-irradiated", "clouds": "f-ui-include_clouds", "retrieval": "f-ui-retrieval"}

# the badge's caption for each kind of calculation in go.OBSERVATION_CALC_MAP
GEOMETRY_LABELS = {"transmission": "Transmission", "thermal": "Emission", "reflected": "Reflected"}


# =======================================
# RECORDING PROGRESS
# =======================================
def touched(sess):
    """Names of the cards the student has edited or run something on."""
    return sess.ui.setdefault("touched", set())


def touch(sess, card_name):
    touched(sess).add(card_name)


# =======================================
# STATE FOR THE TEMPLATE
# =======================================
def geometry(config):
    """The kinds of calculation in the observation type, e.g. {"transmission", "thermal"} for transit_depth+thermal."""
    kinds = {go.OBSERVATION_CALC_MAP.get(part) for part in config["observation_type"].split("+")}
    return kinds & set(GEOMETRY_LABELS)


def enabled(sess, key):
    if key == "star":
        return is_irradiated(sess.config)
    if key == "clouds":
        return sess.ui["include_clouds"]
    if key == "retrieval":
        return sess.ui["retrieval"]
    return True


def retrieval_progress(sess):
    """{retrieval card name: done}. The prior test counts once its samples have been drawn."""
    edited = touched(sess)
    progress = {name: name in edited for name in RETRIEVAL_CARDS}
    progress["prior_test"] = "prior_samples" in sess.results
    return progress


def done(sess, key):
    edited = touched(sess)
    if key == "spectrum":
        return "spectrum" in sess.results
    if key == "retrieval":
        return all(retrieval_progress(sess).values())
    step = next(s for s in STEPS if s.key == key)
    return any(card in edited for card in step.cards)


def state(sess, card_views, error_card=None):
    """
    What templates/spectrum/_builder.html needs. `card_views` are the page's
    rendered cards (views/cards.py): a card showing an error note marks its step.
    `error_card` is a card whose submitted values failed validation.
    """
    with_errors = {STEP_OF_CARD.get(view["card"].name) for view in card_views
                   if any(level == "error" for level, _ in view["notes"])}
    with_errors.add(STEP_OF_CARD.get(error_card))

    steps = []
    for step in STEPS:
        on = enabled(sess, step.key)
        steps.append({
            "step": step,
            "state": "off" if not on else "done" if done(sess, step.key) else "todo",
            "optional": step.key in TOGGLES and not (step.key == "star" and sess.config["observation_type"] != "thermal"),
            "error": on and step.key in with_errors,
            "toggle": TOGGLES.get(step.key),
        })
    by_key = {s["step"].key: s["state"] for s in steps}

    kinds = geometry(sess.config)
    star_on = enabled(sess, "star")
    cloud_count = len(clouds.cloud_ids(sess.config["clouds"])) if sess.ui["include_clouds"] else 0
    return {
        "steps": steps,
        "subs": retrieval_progress(sess) if sess.ui["retrieval"] else None,
        "card_steps": STEP_OF_CARD,
        "layers": _layers(by_key, kinds, star_on),
        "geometry_label": " + ".join(GEOMETRY_LABELS[k] for k in GEOMETRY_LABELS if k in kinds) or "Spectrum",
        "chips": [
            ("with star" if star_on else "no star", star_on),
            (f"{cloud_count} cloud{'s' if cloud_count != 1 else ''}" if cloud_count else "no clouds", bool(cloud_count)),
            ("retrieval" if sess.ui["retrieval"] else "no retrieval", sess.ui["retrieval"]),
        ],
    }


def _layers(states, kinds, star_on):
    """{badge layer id: "off" | "ghost" | "on"}. The star is drawn dashed until its step is done; the rest fade in."""
    def solid(key, shown=True):
        return "on" if shown and states[key] == "done" else "off"

    star = "on" if states["star"] == "done" else "ghost"
    transit = "transmission" in kinds
    return {
        "star-transit": star if star_on and transit else "off",
        "star-side": star if star_on and not transit else "off",
        "object": solid("object"),
        "pt": solid("pt"),
        "chemistry": solid("chemistry"),
        "clouds": solid("clouds"),
        "lit": "on" if star_on and "reflected" in kinds and states["object"] == "done" else "off",
        "spec-transit": solid("spectrum", transit),
        "spec-emit": solid("spectrum", "thermal" in kinds),
        "spec-reflect": solid("spectrum", "reflected" in kinds),
        "retrieval": solid("retrieval"),
    }
