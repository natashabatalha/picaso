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


@dataclass(frozen=True)
class Group:
    label: str
    summary: str
    pages: tuple

    def contains(self, request_path):
        return any(is_active(page.path, request_path) for page in self.pages)


GROUPS = (
    Group("Model", "Configure and run PICASO.", (
        Page("/spectrum/", "Spectrum & Retrieval Setup",
             "Build a driver configuration (star, planet, pressure-temperature profile, chemistry, clouds) and "
             "compute transmission, emission or reflected light spectra. Add observational data, free parameters, "
             "priors and sampler options to export a ready-to-run retrieval."),
    )),
    Group("Analyze", "Interpret retrievals and observations.", (
        Page("/analysis/", "Retrieval Analysis",
             "Load a finished retrieval to make corner plots, compare the maximum likelihood model to the data, "
             "generate banded profiles and spectra, and export the results."),
        Page("/info-content/", "Information Content",
             "Compare observation cases by their degrees of freedom and Shannon information content.", wip=True),
    )),
    Group("Explore", "Look inside PICASO's inputs.", (
        Page("/opacities/", "Opacity Viewer",
             "Plot molecular and continuum cross sections from an opacity database at chosen temperatures, "
             "pressures and resolution."),
    )),
    Group("Setup", "Get PICASO's data in place.", (
        Page("/refdata/", "Reference Data",
             "Point the app at your reference data, check your environment, and download opacities, "
             "stellar grids and other data products."),
    )),
)

RESOURCES = (
    ("Documentation", "https://natashabatalha.github.io/picaso"),
    ("Tutorials", "https://natashabatalha.github.io/picaso/tutorials.html"),
    ("Installation Guide", "https://natashabatalha.github.io/picaso/installation.html"),
    ("GitHub Repository", "https://github.com/natashabatalha/picaso"),
)


def is_active(path, request_path):
    return request_path == path or (path != "/" and request_path.startswith(path))
