"""
The app's pages, grouped into the top-bar menus. The home page lists the same
pages, so a new feature only needs an entry here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    path: str
    label: str
    summary: str
    wip: bool = False
    visual: str = ""  # image in static/ shown on the Get started card; empty shows a placeholder


@dataclass(frozen=True)
class Group:
    label: str
    summary: str
    pages: tuple

    def contains(self, request_path):
        return any(is_active(page.path, request_path) for page in self.pages)


GROUPS = (
    Group("Setup", "Get PICASO's data in place.", (
        Page("/refdata/", "Set up reference data",
             "Point the app at your reference data, check your environment, and download opacities, "
             "stellar grids and other data products."),
    )),
    Group("Model", "Configure and run PICASO.", (
        Page("/spectrum/", "Run a spectrum",
             "Build a driver configuration (star, planet, pressure-temperature profile, chemistry, clouds) and "
             "compute transmission, emission or reflected light spectra. Add observational data, free parameters, "
             "priors and sampler options to export a ready-to-run retrieval."),
        Page("/climate/", "Calculate 1D climate",
             "Build a climate configuration (object, star, initial pressure-temperature guess, chemistry) and "
             "compute a self-consistent radiative-convective temperature profile for a brown dwarf or planet."),
    )),
    Group("Analyze", "Interpret retrievals and observations.", (
        Page("/analysis/", "Analyze a retrieval",
             "Load a finished retrieval to make corner plots, compare the maximum likelihood model to the data, "
             "generate banded profiles and spectra, and export the results."),
        Page("/info-content/", "Compare observations",
             "Compare observation cases by their degrees of freedom and Shannon information content.", wip=True),
    )),
    Group("Explore", "Look inside PICASO's inputs.", (
        Page("/opacities/", "View opacities",
             "Plot molecular and continuum cross sections from an opacity database at chosen temperatures, "
             "pressures and resolution."),
    )),
)

RESOURCES = (
    ("Documentation", "https://natashabatalha.github.io/picaso"),
    ("Tutorials", "https://natashabatalha.github.io/picaso/tutorials.html"),
    ("Installation guide", "https://natashabatalha.github.io/picaso/installation.html"),
    ("GitHub repository", "https://github.com/natashabatalha/picaso"),
)


def is_active(path, request_path):
    return request_path == path or (path != "/" and request_path.startswith(path))
