"""
This class is used for defining the scoring function(s) which can be used during
fine-tuning.
"""
# load general packages and functions
from collections import namedtuple
import torch,pickle 
from rdkit import DataStructs
from rdkit.Chem import QED, AllChem
import numpy as np
import sklearn,math
from sklearn import svm
from ..comparm import *
from tqdm import tqdm
import random
from ..graphs.rdkitutils import *
from ..dock.utils_dock import *
from .dock_score import * 
from .rocs_similarity import * 

def deal_activity_csv(csvfile):
    smis=[]
    labels=[]
    with open(csvfile,'r') as f:
        lines=[line for line in f.readlines() ][1:-1]
        for line in lines:
            smi=line.strip().split()[-1]
            label=line.strip().split()[3]
            smis.append(smi)
            labels.append(label)
    return smis,labels

def create_activity_model(smis,labels,path='./activity',cut_rate=0.9):
    if not os.path.exists(path):
        os.system(f'mkdir -p {path}')
    model=svm.SVC(probability=True)
    X=[]
    Y=[]
    ids=[i for i in range(len(smis))]
    random.shuffle(ids)
    smis=[smis[i] for i in ids]
    labels=[labels[i] for i in ids]

    for idx, smi in tqdm(enumerate(smis)):
        mol=Chem.MolFromSmiles(smi)
        if mol:
            fingerprint   = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)
            ecfp4         = np.zeros((2048,))
            DataStructs.ConvertToNumpyArray(fingerprint, ecfp4)
            if labels[idx]=='N':
                label=0
            else:
                label=1
            X.append(ecfp4)
            Y.append(label)
    #print (Y)
    with open(f'{path}/X.pickle','wb') as f:
        pickle.dump(X,f)
    with open(f'{path}/Y.pickle','wb') as f:
        pickle.dump(Y,f)
    with open(f'{path}/X.pickle','rb') as f:
        X=pickle.load(f)
    with open(f'{path}/Y.pickle','rb') as f:
        Y=pickle.load(f)
    cutnum=math.ceil(len(X)*cut_rate)
    model.fit(np.array(X)[:cutnum],np.array(Y)[:cutnum])
    scores=model.score(np.array(X)[cutnum:],np.array(Y)[cutnum:])
    Y_pred=model.predict(np.array(X)[cutnum:])

    print (f'Mean accuracy of SVC model is {scores}')
    with open(f'{path}/SVC.pickle','wb') as f:
        pickle.dump(model,f)
    return 

class ScoringFunction:
    """
    A class for defining the scoring function components.
    """
    def __init__(self):
        """
        Args:
        ----
            constants (namedtuple) : Contains job parameters as well as global
                                     constants.
        """
        self.score_components = GP.rlsetting.score_components # list
        self.score_weights = GP.rlsetting.score_weights
        self.score_type       = GP.rlsetting.score_type        # list
        self.target_smiles    = GP.rlsetting.target_smiles
        self.qsar_models_path = GP.rlsetting.qsar_models_path  # dict
        self.device           = GP.modelsetting.device
        self.max_gen_atoms      = GP.rlsetting.max_gen_atoms
        self.score_thresholds = GP.rlsetting.score_thresholds
        self.tanimoto_k    = GP.rlsetting.tanimoto_k
        self.target_mols=[Chem.MolFromSmiles(smi)  for smi in self.target_smiles]
        self.target_fps=[AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True) for mol in self.target_mols]
        #self.dock_output_path=GP.docksetting.output_path

        #self.target_smarts = GP.rlsetting.target_smarts
        #print (len(self.target_fps)) 
        if 'activity' in self.score_components:
            if os.path.exists(self.qsar_models_path):
                with open (self.qsar_models_path,'rb') as f:
                    self.qsar_model=pickle.load(f)
            else:
                print (f'QSAR Model not exist in {self.qsar_models_path}!')
                stop
        if 'dockscore' in self.score_components:
            self.dock_model=dockstream_docker()
            self.dock_model.prepare_target()
        if "shape" in self.score_components:
            self.rocs_model=Rocs_scorer(shape_w=GP.rocssetting.shape_w,
                                        color_w=GP.rocssetting.color_w,
                                        cff_path=GP.rocssetting.cff_path,
                                        sdf_path=GP.rocssetting.reflig_sdf_path,
                                        sim_measure=GP.rocssetting.sim_measure
                                        )
        assert len(self.score_components) == len(self.score_thresholds), \
               "`score_components` and `score_thresholds` do not match."
        assert len(self.score_components) == len(self.score_weights), \
               "`score_components` and `score_weights` do not match."
    def compute_score(self, mols : list,output_path='./') -> torch.Tensor:
        nmols=len(mols)
        validity=torch.zeros(nmols,device=self.device)
        uniqueness=torch.ones(nmols,device=self.device)
        smiles=[]
        for mid,mol in enumerate(mols):
            if mol:
                validity[mid]=1
                smi=Chem.MolToSmiles(mol)
                if smi in smiles:
                    uniqueness[mid]=0
                smiles.append(smi)

        contributions_to_score,unknown = self.get_contributions_to_score(mols,output_path)

        if len(self.score_components) == 1:
            final_score = contributions_to_score[0]

        elif self.score_type == "continuous":
            final_score = contributions_to_score[0]*self.score_weights[0]
            for cid,component in enumerate(contributions_to_score[1:]):
                final_score *= component*self.score_weights[cid]

        elif self.score_type == "binary":
            component_masks = []
            for idx, score_component in enumerate(contributions_to_score):
                component_mask = torch.where(
                    score_component > self.score_thresholds[idx],
                    torch.ones(self.n_graphs, device=self.device, dtype=torch.uint8),
                    torch.zeros(self.n_graphs, device=self.device, dtype=torch.uint8)
                )
                component_masks.append(component_mask)
            final_score = component_masks[0]
            for mask in component_masks[1:]:
                final_score *= mask
                final_score  = final_score.float()
        else:
            raise NotImplementedError

        #print (final_score.shape,uniqueness.shape)
        # remove contribution of duplicate molecules to the score
        #final_score *= uniqueness

        # remove contribution of invalid molecules to the score
        final_score *= validity
        return final_score,validity,uniqueness,unknown,contributions_to_score

    def get_contributions_to_score(self, mols : list,output_path='./') -> list:
        contributions_to_score = []
        nmols=len(mols)
        unknown=torch.zeros(nmols,device=self.device)
        for score_component in self.score_components:
            if "target_size" in score_component:
                target_size  = int(score_component[12:])
                assert target_size <= self.max_atoms, \
                       "Target size > largest possible size (`GP.syssetting.max_atoms`)."
                assert 0 < target_size, "Target size must be greater than 0."

                target_size *= torch.ones(self.n_graphs, device=self.device)

                n_nodes      = torch.tensor([len(mol.GetAtoms()) for mol in mols],
                                            device=self.device)
                max_nodes    = self.max_gen_atoms

                score        = (
                    torch.ones(nmols, device=self.device)
                    - torch.abs(n_nodes - target_size)
                    / (max_nodes - target_size)
                )
                contributions_to_score.append(score)

            elif score_component == "QED":
                # compute the QED score for each molecule (if possible)
                qed = []
                for mol in mols:
                    try:
                        qed.append(QED.qed(mol))
                    except:
                        qed.append(0.0)
                score = torch.tensor(qed, device=self.device)
                contributions_to_score.append(score)

            elif score_component == "similarities":
                tanimoto_similarities=[]
                for mid,mol in enumerate(mols):
                    if mol:
                        fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
                        similarities=[]
                        for target_fp in self.target_fps:
                            score = DataStructs.TanimotoSimilarity(target_fp, fp)
                            #score = min(score, self.tanimoto_k) / self.tanimoto_k
                            similarities.append(score)
                        sscore=max(similarities)
                        if sscore<1:
                            tanimoto_similarities.append(min(sscore, self.tanimoto_k) / self.tanimoto_k)
                            unknown[mid]=1
                        else:
                            tanimoto_similarities.append(0.0)
                    else:
                        tanimoto_similarities.append(0.0)
                fscore=torch.tensor(tanimoto_similarities,device=self.device)
                contributions_to_score.append(fscore)

            elif score_component == "match_substructure":
                matches=self.compute_match_scores(mols)
                #print (matches)
                score =torch.tensor(matches,device=self.device).float()
                contributions_to_score.append(score)

            elif score_component == "mcs_similarity":
                matches=self.compute_mcs_scores(mols)
                #print (matches)
                score =torch.tensor(matches,device=self.device).float()
                contributions_to_score.append(score)
            elif score_component == "activity":
                score      = self.compute_activity(mols, self.qsar_model)
                contributions_to_score.append(score)
            elif score_component == "dockscore":
                score=self.dock_model.compute_scores(mols,output_path)
                scores=torch.tensor(score,device=self.device)
                contributions_to_score.append(scores)
            elif score_component == "shape":
                smiles=[]
                for mol in mols:
                    if mol:
                        smi=Chem.MolToSmiles(mol)
                        smiles.append(smi)
                    else:
                        smiles.append('None')
                score=self.rocs_model._calculate_omega_score(smiles)
                scores=torch.tensor(score,device=self.device)
                contributions_to_score.append(scores) 
            else:
                raise NotImplementedError("The score component is not defined. "
                                          "You can define it in "
                                          "`ScoringFunction.py`.")
        return contributions_to_score,unknown

    def compute_activity(self, mols, activity_model):
        """
        Note: this function may have to be tuned/replicated depending on how
        the activity model is saved.
        Args:
        ----
            mols (list) : Contains `rdkit.Mol` objects corresponding to molecular
                          graphs sampled.
            activity_model (sklearn.svm.classes.SVC) : Pre-trained QSAR model.
        Returns:
        -------
            activity (list) : Contains predicted activities for input molecules.
        """
        nmols   = len(mols)
        activity = torch.zeros(nmols, device=self.device)
        for idx, mol in enumerate(mols):
            #try:
            if mol:
                fingerprint   = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                ecfp4         = np.zeros((2048,))
                DataStructs.ConvertToNumpyArray(fingerprint, ecfp4)
                #print (activity_model.predict_proba([ecfp4]))
                activity[idx] = activity_model.predict_proba([ecfp4])[0][1]
                #print(activity)
            #except:
            #    pass  # activity[idx] will remain 0.0
        return activity
    def compute_match_scores(self,mols):
        matches=match_substructures(mols,submols=self.target_mols)
        return matches 
    def compute_mcs_scores(self,mols):
        matches=mcs_similarity(mols,submols=self.target_mols)
        return matches
    