import numpy as np
from openeye import oechem, oeomega, oeshape
from dataclasses import dataclass
from collections import namedtuple 
class Rocs_scorer():
    def __init__(self,shape_w,color_w,cff_path,sq_path=None,sdf_path=None,sim_measure="RefTversky"):
        self.shape_w=shape_w
        self.color_w=color_w
        self.cff_path=cff_path
        self.prep=self.__set_prep(cff_path)
        if sq_path is not None:
            self.overlay_type='shape_query'
            self.input_path=sq_path
        else:
            self.overlay_type='sdf_query'
            self.input_path=sdf_path
        self.sim_measure=sim_measure
        self.sim_func_name_set = self.__similarity_collection(sim_measure)
        print (self.sim_func_name_set)
        self.overlay=self.__prepare_overlay(self.input_path,self.overlay_type)
        self.omega=self.__setup_omega()
        oechem.OEThrow.SetLevel(10000)

    def _calculate_omega_score(self, smiles) -> np.array:
        scores = []
        predicate = getattr(oeshape, self.sim_func_name_set.predicate)()
        for smile in smiles:
            imol = oechem.OEMol()
            best_score = 0.0
            if oechem.OESmilesToMol(imol, smile):
                if self.omega(imol):
                    self.prep.Prep(imol)
                    score = oeshape.OEBestOverlayScore()
                    self.overlay.BestOverlay(score, imol, predicate)
                    best_score_shape = getattr(score, self.sim_func_name_set.shape)()
                    best_score_color = getattr(score, self.sim_func_name_set.color)()
                    best_score_color = correct_color_score(best_score_color)
                    best_score = ((self.shape_w * best_score_shape) + (
                            self.color_w * best_score_color)) / (self.shape_w + self.color_w)
            scores.append(best_score)
        return np.array(scores)

    def __similarity_collection(self, sim_measure_type):
        _SIM_FUNC = namedtuple('sim_func', ['shape', 'color', 'predicate'])
        _SIM_DEF_DICT = {
            "Tanimoto": _SIM_FUNC('GetTanimoto', 'GetColorTanimoto', 'OEHighestTanimotoCombo'),
            "RefTversky": _SIM_FUNC('GetRefTversky', 'GetRefColorTversky','OEHighestRefTverskyCombo'),
            "FitTversky": _SIM_FUNC('GetFitTversky', 'GetFitColorTversky','OEHighestFitTverskyCombo'),
        }
        return _SIM_DEF_DICT.get(sim_measure_type)

    def __set_prep(self,cff_path):
        prep = oeshape.OEOverlapPrep()
        if cff_path is None:
            cff_path = oeshape.OEColorFFType_ImplicitMillsDean
        cff = oeshape.OEColorForceField()
        if cff.Init(cff_path):
            prep.SetColorForceField(cff)
        else:
            raise ValueError("Custom color force field initialisation failed")
        return prep

    def __setup_reference_molecule(self, file_path):
        input_stream = oechem.oemolistream()
        input_stream.SetFormat(oechem.OEFormat_SDF)
        input_stream.SetConfTest(oechem.OEAbsoluteConfTest(compTitles=False))
        refmol = oechem.OEMol()
        if input_stream.open(file_path):
            oechem.OEReadMolecule(input_stream, refmol)
        cff = oeshape.OEColorForceField()
        if cff.Init(oeshape.OEColorFFType_ImplicitMillsDean):
            self.prep.SetColorForceField(cff)
        self.prep.Prep(refmol)
        overlay = oeshape.OEMultiRefOverlay()
        overlay.SetupRef(refmol)
        return overlay

    def __setup_reference_molecule_with_shape_query(self, shape_query):
        qry = oeshape.OEShapeQuery()
        overlay = oeshape.OEOverlay()
        if oeshape.OEReadShapeQuery(shape_query, qry):
            overlay.SetupRef(qry)
        return overlay

    def __prepare_overlay(self,file_path,overlay_type):
        if self.overlay_type=='sdf_query':
            overlay=self.__setup_reference_molecule(file_path)
        else:
            overlay=self.__setup_reference_molecule_with_shape_query(file_path)
        return overlay

    def __setup_omega(self):
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetStrictStereo(False)
        return oeomega.OEOmega(omegaOpts)
        
def correct_color_score(score):
    if score >= 1.0:
        score = 0.9 # or alternative
    return score
"""
rocser=Rocs_scorer(shape_w=0.5,color_w=0.5,cff_path='./6w8i_ligand.cff',sdf_path='6w8i_ligand.sdf')
with open('ligands.smi','r') as f:
    smiles=[line.strip() for line in f.readlines()]
scores=rocser._calculate_omega_score(smiles)
print (scores)
"""