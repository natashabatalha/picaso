"""
picaso.references on bibtexparser 2.x (#450).

tests/data/references_bibtexparser1.json holds what References.get_opa returned under
bibtexparser 1.4.4. The 2.x port has to give the same LaTeX table and the same entries.
Two things differ between the parsers by design, and are normalised before comparing:
  - 1.x expanded month macros (oct -> October); 2.x keeps the macro and writes it bare
    (month = oct), which BibTeX expands itself.
  - 1.x stripped the indentation of continuation lines inside a field; 2.x keeps it.
"""
import json
import re
from pathlib import Path

import bibtexparser
import pytest

from picaso import references as pref

SNAPSHOT = Path(__file__).parent / 'data' / 'references_bibtexparser1.json'

MONTHS = {m[:3].lower(): m for m in ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                                     'August', 'September', 'October', 'November', 'December']}


def _normalise(fields):
    """Lower-case the field names, expand month macros and collapse continuation-line indentation."""
    out = {}
    for key, value in fields.items():
        key = key.lower()
        value = re.sub(r'\n\s*', '\n', str(value))
        if key == 'month':
            value = MONTHS.get(value.lower(), value)
        out[key] = value
    return out


def _entry_fields(entry):
    """bibtexparser 2.x Entry -> the flat dict bibtexparser 1.x used, including ENTRYTYPE and ID."""
    fields = {f.key: f.value for f in entry.fields}
    fields['ENTRYTYPE'] = entry.entry_type
    fields['ID'] = entry.key
    return _normalise(fields)


with open(SNAPSHOT) as f:
    CASES = json.load(f)['cases']


@pytest.fixture(scope='module')
def refs():
    return pref.References()


def test_bib_parses_cleanly(refs):
    # references.bib has a few keys that appear twice with identical content; 2.x keeps one copy
    assert {'Bittner2018Lis', 'GharibNezhad2021', 'Ryabchikova2015'} <= set(refs.bib_dict)
    assert all(isinstance(e, bibtexparser.model.Entry) for e in refs.bib_dict.values())


def test_every_reference_list_id_is_in_the_bib(refs):
    missing = []
    for mol, ids in refs.reflist['opacities'].items():
        for ID in ([ids] if isinstance(ids, str) else ids):
            if ID not in refs.bib_dict:
                missing.append((mol, ID))
    assert missing == []


@pytest.mark.parametrize('case', sorted(CASES))
def test_get_opa_matches_bibtexparser1(refs, case):
    expected = CASES[case]
    tex, db = refs.get_opa(molecules=expected['molecules'])

    assert isinstance(db, bibtexparser.Library)
    assert tex == expected['tex']
    assert [e.key for e in db.entries] == [e['ID'] for e in expected['entries']]
    for got, want in zip(db.entries, expected['entries']):
        assert _entry_fields(got) == _normalise(want), got.key


@pytest.mark.parametrize('case', sorted(CASES))
def test_create_bib_round_trips(refs, case, tmp_path):
    _, db = refs.get_opa(molecules=CASES[case]['molecules'])
    filename = tmp_path / 'refs.bib'
    pref.create_bib(db, str(filename))
    # writing twice must not change the shared entries in refs.bib_dict (e.g. add extra braces)
    pref.create_bib(db, str(filename))

    reread = bibtexparser.parse_file(str(filename))
    assert reread.failed_blocks == []
    assert [e.key for e in reread.entries] == [e.key for e in db.entries]
    for got, want in zip(reread.entries, db.entries):
        assert _entry_fields(got) == _entry_fields(want), got.key


def test_get_opa_needs_input(refs):
    with pytest.raises(Exception, match='Need to either'):
        refs.get_opa()
