"""
Per-browser-session state.

The state is too large (and partly not serializable) for Flask's cookie
session, so the cookie only carries a session id and the state lives in
process memory. That is enough for a local single-user app; a hosted
deployment would swap this module for a shared store.
"""
import os
import tempfile
import uuid
from dataclasses import dataclass, field

from flask import session

from picaso.driver_ui.core.config_schema import apply_form


PAGE_STATE = ("ui", "analysis", "info_content")


@dataclass
class Session:
    config: dict = None  # the Spectrum page's working driver config
    ui: dict = field(default_factory=dict)  # UI settings that are not part of the config
    results: dict = field(default_factory=dict)  # computed outputs (figures, model runs, messages)
    analysis: dict = field(default_factory=dict)  # Retrieval Analysis page
    info_content: dict = field(default_factory=dict)  # Information Content page
    climate: dict = field(default_factory=dict)  # 1D Climate Calculations page; its cards apply forms themselves (views/climate.py)
    upload_dir: str = None  # where uploaded observation data files are saved (see save_uploads)

    def apply(self, fields, form):
        """
        Applies a submitted form. Fields under a page-state path ("ui",
        "analysis", "info_content") write to that dict; everything else writes
        to `config`. Returns {field name: error}.
        """
        root = {**(self.config or {}), **{key: getattr(self, key) for key in PAGE_STATE}}
        root, errors = apply_form(root, fields, form)
        for key in PAGE_STATE:
            setattr(self, key, root.pop(key))
        if self.config is not None:
            self.config = root
        return errors

    def save_uploads(self, files):
        """
        Saves uploaded files (werkzeug FileStorage) to this session's upload folder under their
        own file names, which name the data in the driver (e.g. systematics keys), so they are
        kept as is. A later upload with the same name replaces the earlier file. Returns the paths.
        """
        if self.upload_dir is None or not os.path.isdir(self.upload_dir):
            self.upload_dir = tempfile.mkdtemp(prefix="picaso_data_")
        paths = []
        for file in files:
            name = os.path.basename((file.filename or "").replace("\\", "/"))
            if name in ("", ".", ".."):
                continue
            path = os.path.join(self.upload_dir, name)
            file.save(path)
            paths.append(path)
        return paths


_sessions = {}


def current():
    sid = session.setdefault("sid", uuid.uuid4().hex)
    return _sessions.setdefault(sid, Session())
