"""
Zero-overhead citation tracking for PICASO functions and opacity references.

Functions are tagged with DOIs at *definition* time via the `@cite`
decorator below. The decorator only attaches metadata to the function
object and records it in a module-level registry -- it never wraps the
function, so calling a cited function costs exactly what calling the
undecorated function would cost (no added runtime overhead).

To find out which citations apply to a given run, callers do not need to
execute anything: `driver.references(driver_file)` statically reads the
driver TOML config (the same config `driver.run` consumes) to determine
which pt_/chem_/cloud_ parameterization functions would be selected, then
looks up any DOIs registered against those function names.

To cite a new function anywhere in the codebase, decorate it:

    from .references import cite

    @cite('10.1051/0004-6361/200913396')
    def pt_guillot(self, ...):
        ...

A function may carry more than one DOI (e.g. the framework it implements
plus the paper defining its exact equations):

    @cite('10.1093/mnras/stx1246', '10.1093/mnras/stad670')
    def cloud_brewster_mie(self, ...):
        ...
"""

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
import os
import json 
import numpy as np

CITATIONS = {}


def cite(*dois):
    """
    Decorator that tags a function with one or more DOI references.

    Attaches metadata only (as a `.dois` attribute) and registers the
    function's name in the module-level CITATIONS registry -- it does
    not wrap or call the function, so there is no runtime cost to citing
    a function.

    Parameters
    ----------
    *dois : str
        One or more DOI strings, e.g. '10.1051/0004-6361/200913396'
    """
    def decorator(func):
        func.dois = dois
        CITATIONS[func.__name__] = dois
        return func
    return decorator


def get_citations(func_name):
    """
    Look up the DOIs registered for a function name (empty tuple if none).
    """
    return CITATIONS.get(func_name, ())


def all_citations():
    """
    Return the full citation registry as {function_name: (doi, ...)}.
    """
    return dict(CITATIONS)


class References(): 
    """
    Class structure to get references from PICASO
    """
    def __init__(self): 
        bibfile = os.path.join(os.environ['picaso_refdata'],'references','references.bib')
        reflist = os.path.join(os.environ['picaso_refdata'],'references','reference_list.json')
        with open(bibfile) as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        self.bib_dict = {i['ID']:i for i in bib_database.entries}
        self.reflist = json.load(open(reflist))         

    def get_opa(self, full_output=None, molecules=[]):
        """
        Get opacities references based on full output or a list of molecules 

        Parameters
        ----------
        full_output : dict 
            Full output dictionary from picaso.spectrum e.g. (out['full_output'])
        molecules : list 
            list of string of molecules 

        Returns
        -------
        latex formatted table, bib database
        """
        opa_tex_start = r"""
        \begin{table*}
        \centering
        \begin{tabular}{c|c}
        """
        opa_tex_mid=r"""molXX &  \citet{ID} \\ 
        """
        opa_tex_end=r"""
            \end{tabular}
            \caption{Line lists used to make PICASO Opacities}
            \label{tab:opas}
        \end{table*}
        """


        opacity_refs = self.reflist['opacities']
        if not isinstance(full_output,type(None)):
            molecules = list(full_output['layer']['mixingratios'].keys())
        elif len(molecules) > 0: 
            molecules = molecules 
        else: 
            raise Exception('Need to either entire in a full_ouput or a list of molecules')

        all_opacity_refs_ids = {}
        for imol in molecules:
            for iref in opacity_refs.keys():
                if imol == iref: 
                    all_opacity_refs_ids[iref] = opacity_refs[iref]

        if "H2" in molecules:
            all_opacity_refs_ids['H2--H2'] = opacity_refs['H2--H2']
        if ("H2" in molecules) and ("He" in molecules):
            all_opacity_refs_ids['H2--He'] = opacity_refs['H2--He']
        if ("H2" in molecules) and ("N2" in molecules):
            all_opacity_refs_ids['H2--N2'] = opacity_refs['H2--N2']  
        if  ("H2" in molecules) and ("H" in molecules):
            all_opacity_refs_ids['H2--H'] = opacity_refs['H2--H']
        if  ("H2" in molecules) and ("CH4" in molecules):
            all_opacity_refs_ids['H2--CH4'] = opacity_refs['H2--CH4']
        if ("H-" in molecules):
            all_opacity_refs_ids['H-bf'] = opacity_refs['H-bf']
        if ("H" in molecules) and ("e-" in molecules):
            all_opacity_refs_ids['H-bf'] = opacity_refs['H-ff']
        if ("H2" in molecules) and ("e-" in molecules):
            all_opacity_refs_ids['H2-'] = opacity_refs['H2-']  

        opa_tex=""""""
        all_ids = []
        for imol in all_opacity_refs_ids.keys():
            if isinstance(all_opacity_refs_ids[imol],str):
                #ged ids in string form for latex
                i_ids=all_opacity_refs_ids[imol]
                #get list of refs
                all_ids += [all_opacity_refs_ids[imol]]
            else: 
                i_ids=','.join(all_opacity_refs_ids[imol])
                all_ids += all_opacity_refs_ids[imol]

            opa_tex += opa_tex_mid.replace('ID',i_ids).replace('molXX',imol)

        opa_tex = opa_tex_start+opa_tex+opa_tex_end

        #bibdb = bibtexparser.Library()
        db = BibDatabase()
        for ID in all_ids:
            db.entries += [self.bib_dict[ID]]
        
        return opa_tex, db

def create_bib(bibdb, filename):
    """
    Creates bib file 
    
    Parameters
    ----------
    bibdb : bibtexparser.bibdatabase.BibDatabase
        bib database 
    file : str 
        filename
    """
    writer = BibTexWriter()
    with open(filename, 'w') as bibfile:
        bibfile.write(writer.write(bibdb))
