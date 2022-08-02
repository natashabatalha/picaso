import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bwriter import BibTexWriter
import os
import json 
import numpy as np

class References(): 
    """
    Class structure to get references from PICASO
    """
    def __init__(self): 
        bibfile = os.path.join(os.environ['picaso_refdata'],'references','references.bib')
        reflist = os.path.join(os.environ['picaso_refdata'],'references','reference_list.json')

        with open(bibfile,'r') as bibtex_file:
            #bib_database = bibtexparser.load(bibtex_file,common_strings=True)
            bib_database=bibtexparser.bparser.BibTexParser(
            common_strings=True).parse_file(bibtex_file)
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

        bibdb = BibDatabase()
        bibdb.entries = [self.bib_dict[ID] for ID in all_ids]

        return opa_tex, bibdb

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
