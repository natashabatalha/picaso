"""
Free-chemistry panel: which molecules are included and how each one's
abundance profile is parameterized.

The molecule picker writes the selected molecules to a temporary
`chemistry.free.selected` key; `normalize` consumes it and stores them the way
PICASO expects: background gases in `background.gases`, the rest in `species`.
"""
from picaso.driver_ui.core.config_schema import Choice, MultiChoice, Number, NumberList, Section, Text

PATH = ("chemistry", "free")
LIST_PARAMS = ("P_knots", "vmr_knots")
TEXT_DEFAULTS = {"interpolation_method": "slinear"}
DEFAULT_BACKGROUND_FRACTION = 5.667
MAX_BACKGROUND_GASES = 2


def selected_molecules(free):
    return free.get("species", []) + free.get("background", {}).get("gases", [])


def profile_params(free, profile):
    params = free["profile_options"].get(profile)
    return params if isinstance(params, list) else []  # e.g. background=0 takes no parameters


def fields(free, molecules):
    """
    Parameters
    ----------
    free : dict
        The chemistry.free config section
    molecules : list
        Molecules that can be selected (from the opacity database)
    """
    selected = [mol for mol in selected_molecules(free) if mol in molecules]
    children = [MultiChoice(PATH + ("selected",), selected, molecules, hint="Molecules to include")]

    profiles = list(free["profile_options"])
    for mol in selected:
        mol_path = PATH + (mol,)
        profile = free[mol]["profile"]
        mol_fields = [Choice(mol_path + ("profile",), profile, profiles)]
        for param in profile_params(free, profile):
            mol_fields.append(_param_field(mol_path + (param,), free[mol][param]))
        children.append(Section(mol_path, mol_fields))

    gases = free.get("background", {}).get("gases", [])
    if len(gases) == 2:
        children.append(Number(PATH + ("background", "fraction"), free["background"]["fraction"],
                               hint=f"Fraction between {gases[0]}:{gases[1]}"))
    return Section(PATH, children)


def _param_field(path, value):
    if path[-1] in LIST_PARAMS:
        return NumberList(path, value, hint="comma separated")
    if path[-1] in TEXT_DEFAULTS:
        return Text(path, value)
    return Number(path, value)


def normalize(free):
    """
    Brings the section in line with the molecules the user selected: adds new
    molecules (constant profile), drops deselected ones, fills in parameters a
    newly chosen profile needs, and rebuilds the background gas settings.
    """
    selected = free.pop("selected") if "selected" in free else selected_molecules(free)
    selected = list(dict.fromkeys(selected))
    for key in list(free):
        if isinstance(free[key], dict) and "profile" in free[key] and key not in selected:
            del free[key]

    for mol in selected:
        mol_config = free.setdefault(mol, {"profile": "constant", "unit": "v/v"})
        for param in profile_params(free, mol_config["profile"]):
            mol_config.setdefault(param, [] if param in LIST_PARAMS else TEXT_DEFAULTS.get(param, 0.0))

    background = [mol for mol in selected if free[mol]["profile"] == "background"]
    free["species"] = [mol for mol in selected if mol not in background]
    if background:
        old_fraction = free.get("background", {}).get("fraction", DEFAULT_BACKGROUND_FRACTION)
        free["background"] = {"gases": background}
        if len(background) == 2:
            free["background"]["fraction"] = old_fraction
    else:
        free.pop("background", None)


def errors(free):
    if len(free.get("background", {}).get("gases", [])) > MAX_BACKGROUND_GASES:
        return [f"Only up to {MAX_BACKGROUND_GASES} background gases are supported"]
    return []
