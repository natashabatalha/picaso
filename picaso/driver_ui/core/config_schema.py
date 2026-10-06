"""
Schema layer for driver.toml.

`build_fields` turns a config section into a tree of typed fields for a UI to
render; `apply_form` writes submitted form values back into the config. Every
field is addressed by its dotted path (e.g. "temperature.pressure.min"), which
is also its form field name, so no labels ever have to be parsed back.

Conventions read from the TOML:

    foo = 'a'   with  foo_options = ['a', 'b']   -> Choice
    foo = {value = 1.2, unit = 'Rjup'}           -> Quantity (value editable, unit fixed)
    foo_kwargs = {...}                           -> Section, shown only while a sibling
                                                    Choice is set to 'foo' (e.g.
                                                    distribution = 'lognorm' shows
                                                    lognorm_kwargs); otherwise untouched
    selectors[path] = (key, options)             -> Selector: `key` names which sibling
                                                    table is active, e.g.
                                                    temperature.profile = 'guillot'
    choices[path] = options                      -> Choice with options supplied by code

Paths listed in `custom` become Custom placeholders, for sections that need a
hand-written panel (free chemistry, clouds, xarray grids, ...).
"""
import copy
from dataclasses import dataclass, field


# =======================================
# FIELD TYPES
# =======================================
@dataclass
class Field:
    path: tuple
    label: str = field(default="", kw_only=True)  # shown instead of the key when set
    hint: str = field(default="", kw_only=True)  # help text shown under the input
    links: list = field(default_factory=list, kw_only=True)  # [(label, url)] shown under the input

    @property
    def key(self):
        return self.path[-1]

    @property
    def name(self):
        return ".".join(self.path)

    @property
    def target(self):
        """Path in the config that a parsed form value is written to."""
        return self.path

    def read(self, form):
        """Raw submitted value, or None if this field was not submitted."""
        return form.get(self.name)


@dataclass
class Bool(Field):
    value: bool
    kind = "bool"

    def read(self, form):
        # an unchecked HTML checkbox is simply absent from the form
        return form.get(self.name) is not None

    def parse(self, raw):
        return raw


@dataclass
class Number(Field):
    value: float
    minimum: float = field(default=None, kw_only=True)
    maximum: float = field(default=None, kw_only=True)
    kind = "number"

    def parse(self, raw):
        number = _parse_number(raw, keep_int=isinstance(self.value, int))
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"must be at least {self.minimum:g}")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"must be at most {self.maximum:g}")
        return number


@dataclass
class Quantity(Number):
    unit: str = ""
    kind = "quantity"

    @property
    def target(self):
        return self.path + ("value",)


@dataclass
class Text(Field):
    value: str
    multiline: bool = field(default=False, kw_only=True)  # textarea instead of a single line
    kind = "text"

    def parse(self, raw):
        return raw.strip()


@dataclass
class Choice(Field):
    value: object
    options: list
    kind = "choice"

    def parse(self, raw):
        for option in self.options:
            if str(option) == raw:
                return option
        raise ValueError(f"'{raw}' is not one of {self.options}")


@dataclass
class Selector(Choice):
    branch: "Field | None" = None  # fields of the active sibling table
    kind = "selector"


@dataclass
class MultiChoice(Field):
    value: list
    options: list
    kind = "multi_choice"

    def read(self, form):
        # like checkboxes, an empty selection is absent from the form
        return form.getlist(self.name) if hasattr(form, "getlist") else form.get(self.name, [])

    def parse(self, raw):
        unknown = [item for item in raw if item not in map(str, self.options)]
        if unknown:
            raise ValueError(f"{unknown} not in options")
        return [option for option in self.options if str(option) in raw]


@dataclass
class NumberList(Field):
    value: list
    kind = "number_list"

    def parse(self, raw):
        keep_int = all(isinstance(v, int) for v in self.value)
        return [_parse_number(item, keep_int) for item in _split(raw, ",")]


@dataclass
class TextList(Field):
    value: list
    multiline: bool = field(default=False, kw_only=True)  # one item per line instead of comma separated
    kind = "text_list"

    def parse(self, raw):
        return _split(raw, "\n" if self.multiline else ",")


@dataclass
class FileInput(Field):
    """A file picker. Uploads are not config values, so the card reads them from the request itself."""
    accept: str = ""
    kind = "file"

    def read(self, form):
        return None


@dataclass
class Section(Field):
    children: list = field(default_factory=list)
    kind = "section"


@dataclass
class Custom(Field):
    value: object  # the raw config section, for the hand-written panel
    kind = "custom"


# =======================================
# CONFIG -> FIELDS
# =======================================
def build_fields(section, path=(), selectors=None, custom=frozenset(), choices=None):
    """
    Builds the field tree for one config section.

    Parameters
    ----------
    section : dict
        A (sub)table of the driver.toml config
    path : tuple
        Where `section` lives in the full config, e.g. ('temperature',)
    selectors : dict
        {section path: (selector key, options)}; options=None reads `<key>_options` from the TOML
    custom : set
        Paths to emit as Custom placeholders instead of generic fields
    choices : dict
        {field path: options} for Choice fields whose options are not in the TOML

    Return
    ------
    Section
    """
    selectors = selectors or {}
    choices = choices or {}
    selector_key, branch_names = None, []
    if path in selectors:
        selector_key, branch_names = selectors[path]
        if branch_names is None:
            branch_names = section.get(f"{selector_key}_options", [])

    children = []
    for key, value in section.items():
        if key.endswith("_options") or key in branch_names:
            continue
        if key.endswith("_kwargs") and not _kwargs_is_active(section, key):
            continue

        child_path = path + (key,)
        if key == selector_key:
            branch = None
            if isinstance(section.get(value), dict):
                branch = _field_for(section[value], path + (value,), None, selectors, custom, choices)
            children.append(Selector(child_path, value, list(branch_names), branch))
        else:
            options = choices.get(child_path, section.get(f"{key}_options"))
            options = options if isinstance(options, list) else None
            children.append(_field_for(value, child_path, options, selectors, custom, choices))

    return Section(path, children)


def _field_for(value, path, options, selectors, custom, choices):
    if path in custom:
        return Custom(path, value)
    if options is not None:
        return Choice(path, value, options)
    if isinstance(value, bool):  # before int: bool is a subclass of int
        return Bool(path, value)
    if isinstance(value, (int, float)):
        return Number(path, value)
    if isinstance(value, str):
        return Text(path, value)
    if isinstance(value, list):
        if value and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return NumberList(path, value)
        return TextList(path, value)
    if isinstance(value, dict):
        if "value" in value and "unit" in value:
            return Quantity(path, value["value"], value["unit"])
        return build_fields(value, path, selectors, custom, choices)
    raise TypeError(f"Unsupported value at {'.'.join(path)}: {value!r}")


def _kwargs_is_active(section, kwargs_key):
    """True if a sibling Choice is currently set to this kwargs block's name."""
    name = kwargs_key[: -len("_kwargs")]
    return any(
        value == name and f"{key}_options" in section
        for key, value in section.items()
    )


# =======================================
# FORM -> CONFIG
# =======================================
def apply_form(config, fields, form):
    """
    Writes submitted form values into a copy of the config.

    Fields missing from the form are left unchanged, except Bool and
    MultiChoice fields: unchecked boxes are not submitted, so a missing one
    means False / nothing selected. Only pass the fields that were actually
    rendered in the submitted form.

    Parameters
    ----------
    config : dict
        Full driver.toml config
    fields : Field
        Field tree that was rendered (usually from build_fields)
    form : Mapping
        Submitted values keyed by field name, e.g. Flask's request.form

    Return
    ------
    (dict, dict)
        Updated copy of the config, and {field name: error message} for values that failed to parse
    """
    config = copy.deepcopy(config)
    errors = {}
    for f in iter_inputs(fields):
        raw = f.read(form)
        if raw is None:
            continue
        try:
            set_path(config, f.target, f.parse(raw))
        except ValueError as e:
            errors[f.name] = str(e)
    return config, errors


def iter_inputs(fields):
    """Yields every field that maps to a single form input."""
    if isinstance(fields, Section):
        for child in fields.children:
            yield from iter_inputs(child)
    elif isinstance(fields, Custom):
        return  # hand-written panels parse their own inputs
    else:
        yield fields
        if isinstance(fields, Selector) and fields.branch is not None:
            yield from iter_inputs(fields.branch)


def find_field(fields, name):
    """The field with this dotted name (inputs, sections and placeholders), or None."""
    if fields.name == name:
        return fields
    children = []
    if isinstance(fields, Section):
        children = fields.children
    elif isinstance(fields, Selector) and fields.branch is not None:
        children = [fields.branch]
    for child in children:
        found = find_field(child, name)
        if found is not None:
            return found
    return None


def replace_field(fields, name, new):
    """Swaps the field with this dotted name (e.g. a Custom placeholder) for `new`, in place."""
    if isinstance(fields, Selector) and fields.branch is not None:
        if fields.branch.name == name:
            fields.branch = new
            return True
        return replace_field(fields.branch, name, new)
    if isinstance(fields, Section):
        for i, child in enumerate(fields.children):
            if child.name == name:
                fields.children[i] = new
                return True
            if replace_field(child, name, new):
                return True
    return False


# =======================================
# HELPERS
# =======================================
def set_path(config, path, value):
    """Sets a nested value, creating missing tables; numeric keys index into lists (padding them with None)."""
    for key in path[:-1]:
        config = config[int(key)] if isinstance(config, list) else config.setdefault(key, {})
    if isinstance(config, list):
        index = int(path[-1])
        config.extend([None] * (index + 1 - len(config)))
        config[index] = value
    else:
        config[path[-1]] = value


def _parse_number(raw, keep_int):
    raw = raw.strip()
    if keep_int:
        try:
            return int(raw)
        except ValueError:
            pass
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"'{raw}' is not a number") from None


def _split(raw, separator):
    return [item.strip() for item in raw.split(separator) if item.strip()]
