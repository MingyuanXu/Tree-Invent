
from collections import namedtuple
import torch,pickle 
from rdkit import DataStructs
from rdkit.Chem import QED, AllChem
import numpy as np
from ..comparm import *

from ..graphs.rdkitutils import *
from ..dock.utils_dock import *

def sigmoid_transformation(scores,_low=0,_high=1.0,_k=0.25):
    def _exp(pred_val, low, high, k):
        try:
            return math.pow(10, (10 * k * (pred_val - (low + high) * 0.5) / (low - high)))
        except:
            return 0
    transformed = [1 / (1 + _exp(pred_val, _low, _high, _k)) for pred_val in scores]
    return np.array(transformed, dtype=np.float32)

def reverse_sigmoid_transformation(scores,_low, _high, _k):
    def _reverse_sigmoid_formula(value, low, high, k):
        try:
            return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
        except:
            return 0
    transformed = [_reverse_sigmoid_formula(pred_val, _low, _high, _k) for pred_val in scores]
    return np.array(transformed, dtype=np.float32)
class dockstream_docker():
    def __init__(self):
        self.input_path=GP.docksetting.dock_input_path
        self.center=[None,None,None]
        pass
    def prepare_target(self):
        if GP.docksetting.backend=='AutoDock-Vina':
            try:
                prepare_target_in_vina_format(  input_path=GP.docksetting.dock_input_path,
                                            target_pdb_path=GP.docksetting.target_pdb,
                                            reflig_pdb_path=GP.docksetting.reflig_pdb,
                                            out_path=GP.docksetting.dock_input_path,
                                            log_path='target_prep.log',
                                            dockstream_root_path=GP.docksetting.dockstream_root_path,
                                            bin_path=GP.docksetting.vina_bin_path)
                self.receptor_pdbqt=f"{GP.docksetting.target_pdb.strip('pdb')}fix.pdbqt"
                with open (f'{GP.docksetting.dock_input_path}/target_prep.log','r') as f:
                    for line in f.readlines():
                        if 'X coordinates' in line:
                            self.center[0]=float(line.strip().split()[-1][5:])
                        if 'Y coordinates' in line:
                            self.center[1]=float(line.strip().split()[-1][5:])
                        if 'Z coordinates' in line:
                            self.center[2]=float(line.strip().split()[-1][5:])
            except Exception as e:
                print (f'Dockstream prepare target failed due to {e}, center is {self.center}')
            else:
                print ('Dockstream prepared target into Vina format PDBQT!')
                print (self.center)
        elif GP.docksetting.backend=="Glide":
            pass
        return 

    def dock(self,mols,output_path):
        smiles=[]
        for mol in mols:
            if mol:
                smi=Chem.MolToSmiles(mol)
                smiles.append(smi)
            else:
                smiles.append('None')
        if not os.path.exists(output_path):
            os.system(f'mkdir -p {output_path}')
        with open(f'{output_path}/ligand.smi','w') as f:
            for smi in smiles:
                f.write(smi+'\n') 
        if  GP.docksetting.backend=='AutoDock-Vina': 
            prepare_docking_in_vina_format( input_path=GP.docksetting.dock_input_path,
                                            center=self.center,
                                            box_size=GP.docksetting.box_size,
                                            receptor_pdbqt_path=f'{GP.docksetting.dock_input_path}/{self.receptor_pdbqt}',
                                            lig_smiles_csv_path=f'{output_path}/ligand.smi',
                                            lig_conformers_path=f'{output_path}/ligand.sdf',
                                            vina_binary_location=GP.docksetting.vina_bin_path,
                                            out_path=output_path,
                                            log_path='dock.log',
                                            pose_path='pose.sdf',
                                            score_path='score.log',
                                            ncores=GP.docksetting.ncores,
                                            nposes=GP.docksetting.nposes
                                        )

        elif GP.docksetting.backend=="Glide":
            #print (output_path)
            prepare_docking_in_glide_format(input_path=GP.docksetting.dock_input_path,
                                            grid_path=GP.docksetting.grid_path,
                                            smiles_path=f'{output_path}/ligand.smi',
                                            glide_flags=GP.docksetting.glide_flags,
                                            glide_keywords=GP.docksetting.glide_keywords,
                                            ncores=GP.docksetting.ncores,
                                            out_path=output_path,
                                            log_path='dock.log',
                                            pose_path='pose.sdf',
                                            score_path='score.log',
                                            glide_ver=GP.docksetting.glide_ver
                                            )
            
        docking_scores=dockstream_dock(conf=f'{output_path}/dock.json') 
        return docking_scores

        
    def compute_scores(self,mols,output_path='./'):
        dock_scores=self.dock(mols,output_path)
        scores=reverse_sigmoid_transformation(dock_scores,_low=GP.docksetting.low_threshold,
                                                _high=GP.docksetting.high_threshold,
                                                _k=GP.docksetting.k) 
        #print (scores)
        return scores