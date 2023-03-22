from torch.utils.data import Dataset,DataLoader
from ..comparm import * 
import torch ,copy
import numpy as np 
from tqdm import tqdm 
import random ,pickle 
from ..graphs.mftree import *
from rdkit.Chem import Draw
def Dataset_From_MFTsflist(MFTsflist,path='./datasets',cut=0.99):
    if not os.path.exists(path):
        os.system(f'mkdir -p {path}')
    if not os.path.exists(f'{path}/Frag_type_statistics.pickle' ):
    #if True:
        frag_type_statistics={}
        frag_smis_statistics={}
        for fname in tqdm (MFTsflist):
            #try:
            with open(fname,'rb') as f:
                MFTs=pickle.load(f)
            for MFTree in MFTs:
                for fid,frag in enumerate(MFTree.clique_mols):
                    f=MFTree.f_cliques[fid]
                    frag_natoms=len(MFTree.clique_inner_f_atoms[fid])
                    descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}'
                    if frag_natoms>1:
                        if descriptor not in frag_type_statistics.keys():
                            frag_type_statistics[descriptor]=0
                            frag_smis_statistics[descriptor]=[]
                        frag_type_statistics[descriptor]+=1
                        frag_smis_statistics[descriptor].append(MFTree.clique_smis[fid])

        with open(f'{path}/Frag_type_statistics.pickle','wb') as f:
            pickle.dump(frag_type_statistics,f)
        print ('Save Frag type statistics, Done:\n')
        with open(f'{path}/Frag_smis_statistics.pickle','wb') as f:
            pickle.dump(frag_smis_statistics,f)
        print ('Save Frag smis statistics, Done:\n')
    with open(f'{path}/Frag_type_statistics.pickle','rb') as f:
        frag_type_statistics=pickle.load(f)
    with open(f'{path}/Frag_smis_statistics.pickle','rb') as f:
        frag_smis_statistics=pickle.load(f)

    print ('Start analysis dataset:\n')
    frag_type_num=np.array([v for k,v in frag_type_statistics.items()])
    frag_type_key=[k for k,v in frag_type_statistics.items()]
    order=np.argsort(-frag_type_num)
    total_times=np.sum(frag_type_num)
    cum_frag_num=np.cumsum(frag_type_num[order])
    frag_type_label_dict={}
    frag_smis_label_dict={}
    for i in range(len(order)):
        if cum_frag_num[i]<cut*total_times:
            frag_type_label_dict[frag_type_key[order[i]]]=frag_type_statistics[frag_type_key[order[i]]]
            frag_smis_label_dict[frag_type_key[order[i]]]=frag_smis_statistics[frag_type_key[order[i]]]
    for key in frag_smis_label_dict.keys():
        smileslist=frag_smis_label_dict[key]
        mollist=[]
        for smi in smileslist:
            tmpmol=Chem.MolFromSmiles(smi)
            if tmpmol:
                mollist.append(tmpmol)
        inchilist=list(set([Chem.MolToInchi(mol) for mol in mollist]))
        unique_smilist=[Chem.MolToSmiles(Chem.MolFromInchi(inchi)) for inchi in inchilist]
        frag_smis_label_dict[key]=unique_smilist
    with open(f'{path}/Frag_label_num_{cut}.pickle','wb') as f:
        pickle.dump(frag_type_label_dict,f)
    with open(f'{path}/Frag_label_smis_{cut}.pickle','wb') as f:
        pickle.dump(frag_smis_label_dict,f) 
    with open(f'{path}/descriptor_{cut}.csv','w') as f:
        for key in frag_smis_label_dict.keys():
            f.write(f'{key} {frag_type_label_dict[key]}\n')
            try:
                img=Draw.MolsToGridImage([Chem.MolFromSmiles(smi) for smi in frag_smis_label_dict[key]],molsPerRow=5,subImgSize=(250,250),legends=frag_smis_label_dict[key])
                img.save(f'{path}/{key}.png')
            except Exception as e:
                print (f'{key} draw mols failed due to {e}')
    print ('Analyze analysis dataset, Done\n')
    for fname in MFTsflist: 
        with open (fname,'rb') as f:
            MFTs=pickle.load(f)
        saved_MFTs=[]
        for MFTree in tqdm(MFTs):
            flag=True
            for fid,frag in enumerate(MFTree.clique_mols):
                f=MFTree.f_cliques[fid]
                frag_natoms=len(MFTree.clique_inner_f_atoms[fid])
                if frag_natoms>1:
                    descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}'
                    if descriptor not in frag_type_label_dict.keys():
                        flag=False
            if flag:
                saved_MFTs.append(MFTree)
        with open(f'{fname[:-7]}_dealed.pickle','wb') as f:
            pickle.dump(saved_MFTs,f)
        print (f'Datasets have been saved in {fname[:-7]}_dealed.pickle, for {len(saved_MFTs)} molecules, Done\n')
    return

def Dataset_From_Rdkitmols(mols,mfts_per_file=10000,path='./datasets',cut=0.99):
    if not os.path.exists(path):
        os.system(f'mkdir -p {path}')
    if not os.path.exists(f'{path}/Frag_type_statistics.pickle' ):
    #if True:
        frag_type_statistics={}
        frag_smis_statistics={}
        pid=0
        MFTs=[]
        for mol in tqdm (mols):
            #try:
            if True:
                smi=Chem.MolToSmiles(mol)
                MFTree=MolFragTree(mol,smi=smi)
                for fid,frag in enumerate(MFTree.clique_mols):
                    f=MFTree.f_cliques[fid]
                    frag_natoms=len(MFTree.clique_inner_f_atoms[fid])
                    descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}'
                    if frag_natoms>1:
                        if descriptor not in frag_type_statistics.keys():
                            frag_type_statistics[descriptor]=0
                            frag_smis_statistics[descriptor]=[]
                        frag_type_statistics[descriptor]+=1
                        frag_smis_statistics[descriptor].append(MFTree.clique_smis[fid])
                MFTs.append(MFTree)
            #except Exception as e:
            #    print (f'Failed to trans {Chem.MolToSmiles(mol)} into MFTree!')
            if (len(MFTs)+1)%mfts_per_file==0:
                with open(f'{path}/MFTs_{pid}.pickle','wb') as f:
                    pickle.dump(MFTs,f)
                MFTs=[]
                pid+=1
        print ('Save initial results:\n')
    
        with open(f'{path}/MFTs_{pid}.pickle','wb') as f:
            pickle.dump(MFTs,f)
        print ('Save MFTs, Done:\n')

        with open(f'{path}/Frag_type_statistics.pickle','wb') as f:
            pickle.dump(frag_type_statistics,f)
        print ('Save Frag type statistics, Done:\n')
        with open(f'{path}/Frag_smis_statistics.pickle','wb') as f:
            pickle.dump(frag_smis_statistics,f)
        print ('Save Frag smis statistics, Done:\n')
    #pid=24
    with open(f'{path}/Frag_type_statistics.pickle','rb') as f:
        frag_type_statistics=pickle.load(f)
    with open(f'{path}/Frag_smis_statistics.pickle','rb') as f:
        frag_smis_statistics=pickle.load(f)
    pid=214
    print ('Start analysis dataset:\n')
    frag_type_num=np.array([v for k,v in frag_type_statistics.items()])
    frag_type_key=[k for k,v in frag_type_statistics.items()]
    order=np.argsort(-frag_type_num)
    total_times=np.sum(frag_type_num)
    cum_frag_num=np.cumsum(frag_type_num[order])
    frag_type_label_dict={}
    frag_smis_label_dict={}
    for i in range(len(order)):
        if cum_frag_num[i]<cut*total_times:
            frag_type_label_dict[frag_type_key[order[i]]]=frag_type_statistics[frag_type_key[order[i]]]
            frag_smis_label_dict[frag_type_key[order[i]]]=frag_smis_statistics[frag_type_key[order[i]]]
    for key in frag_smis_label_dict.keys():
        smileslist=frag_smis_label_dict[key]
        mollist=[]
        for smi in smileslist:
            tmpmol=Chem.MolFromSmiles(smi)
            if tmpmol:
                mollist.append(tmpmol)
        inchilist=list(set([Chem.MolToInchi(mol) for mol in mollist]))
        unique_smilist=[Chem.MolToSmiles(Chem.MolFromInchi(inchi)) for inchi in inchilist]
        frag_smis_label_dict[key]=unique_smilist
    with open(f'{path}/Frag_label_num_{cut}.pickle','wb') as f:
        pickle.dump(frag_type_label_dict,f)
    with open(f'{path}/Frag_label_smis_{cut}.pickle','wb') as f:
        pickle.dump(frag_smis_label_dict,f) 
    with open(f'{path}/descriptor_{cut}.csv','w') as f:
        for key in frag_smis_label_dict.keys():
            f.write(f'{key} {frag_type_label_dict[key]}\n')
            try:
                img=Draw.MolsToGridImage([Chem.MolFromSmiles(smi) for smi in frag_smis_label_dict[key]],molsPerRow=5,subImgSize=(250,250),legends=frag_smis_label_dict[key])
                img.save(f'{path}/{key}.png')
            except Exception as e:
                print (f'{key} draw mols failed due to {e}')
    print ('Analyze analysis dataset, Done\n')
    MFTs=[]
    for i in tqdm(range(pid)):
        with open(f'{path}/MFTs_{i}.pickle','rb') as f:
            MFTs+=pickle.load(f)
    print ('Dealed MFTs :',len(MFTs))
    saved_MFTs=[]
    print ('Save molecules with common fragments:\n')
    for MFTree in tqdm(MFTs):
        flag=True
        for fid,frag in enumerate(MFTree.clique_mols):
            f=MFTree.f_cliques[fid]
            frag_natoms=len(MFTree.clique_inner_f_atoms[fid])
            if frag_natoms>1:
                descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}'
                if descriptor not in frag_type_label_dict.keys():
                    flag=False
        if flag:
            saved_MFTs.append(MFTree)
    print ('Dealed MFTs :',len(saved_MFTs))
    MFTs=[]
    npart=math.ceil(len(saved_MFTs)/200000)
    for i in range(npart):
        with open(f'{path}/MFTs_saved_{cut}_{i}.pickle','wb') as f:
            pickle.dump(saved_MFTs[i*200000:(i+1)*200000],f)
        print (f'Datasets have been saved in {path}/MFTs_saved_{cut}_{i}.pickle, for {len(saved_MFTs[i*200000:(i+1)*200000])} molecules, Done\n')
    return

def molfielter(mol,allowed_atoms=[6,7,8,9,15,16,17,35,53],max_mw=600,max_atoms=60):
    atoms=mol.GetAtoms()

    flag=True
    for atom in atoms:
        if atom.GetAtomicNum() not in allowed_atoms:
            flag=False
    mw=Chem.Descriptors.ExactMolWt(mol)
    ssrlist=[list(x) for x in Chem.GetSymmSSSR(mol)]
    if len(ssrlist)>1:
        max_ring_size=np.max([len(ssr) for ssr in ssrlist])
    else:
        max_ring_size=0
    if max_ring_size>9:
        flag=False
    if mw>max_mw:
        flag=False
    if len(atoms)>max_atoms:
        flag=False
    smi=Chem.MolToSmiles(mol)
    if '.' in smi:
        flag=False
    
    return flag

def prepare_mols_from_smi(smiles,savepath='./datasets/mols.smi'):
    with open(savepath,'w') as f:
        mols=[]
        for smi in tqdm(smiles):
            if smi:
                try:
                    mol=Chem.MolFromSmiles(smi)
                    mol=Neutralize_atoms(mol)
                    Chem.Kekulize(mol)
                    if mol:
                        flag=molfielter(mol)
                        if flag:
                            mols.append(mol)
                            f.write(Chem.MolToSmiles(mol)+'\n')
                except Exception as e:
                    print (smi,e)
    return mols 

def prepare_mols_from_sdf(sdfname,save_path='./datasets'):
    mols=[]
    supp=Chem.rdmolfiles.SDMolSupplier(sdfname)
    with open(f'{save_path}/prepared.smi','w') as f:
        print (f'Prepare mols from {sdfname}')
        for mol in tqdm(supp):
            try:
                mol=Neutralize_atoms(mol)
                Chem.Kekulize(mol)
                if mol:
                    flag=molfielter(mol)
                    if flag:
                        mols.append(mol)
                        f.write(Chem.MolToSmiles(mol)+'\n')
                    #else:
                        #print (Chem.MolToSmiles(mol)+' is not allowed')
            except Exception as e:
                print (e)
                pass
    return mols 
    
    

def create_finetune_datasets(mols,path='./datasets',cut=0.99):
    with open (f'{path}/Frag_label_num_{cut}.pickle','rb') as f:
        frag_type_label_dict=pickle.load(f)
    finetune_MFTs=[]
    for mol in mols:
        smi=Chem.MolToSmiles(mol)
        try:
            MFTree=MolFragTree(mol,smi=smi) 
            flag=True
            for fid,frag in enumerate(MFTree.clique_mols):
                f=MFTree.f_cliques[fid]
                frag_natoms=len(MFTree.clique_inner_f_atoms[fid])
                if frag_natoms>1:
                    descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}'
                    if descriptor not in frag_type_label_dict.keys():
                        flag=False
            if MFTree.natoms>GP.syssetting.max_atoms:
                flag=False
            if MFTree.ncliques>GP.syssetting.max_cliques:
                flag=False

            if flag:
                finetune_MFTs.append(MFTree)
        except Exception as e:
            print (smi,e)
    return finetune_MFTs

def find_similar_molecules(target_smi,smis):
    target_mol=Chem.MolFromSmiles(target_smi)
    target_mol=Neutralize_atoms(target_mol)
    Chem.Kekulize(target_mol)
    collect_mols=[target_mol]
    for smi in tqdm(smis):
        mol=Chem.MolFromSmiles(smi)
        if mol:
            mol=Neutralize_atoms(mol)
            Chem.Kekulize(mol)
            simi=tanimoto_similarities(target_mol,mol)
            if simi>0.7:
                collect_mols.append(mol)
    return collect_mols

def Statistic_dataset_params(mfts):
    params={
            'max_atoms':0,
            'max_cliques':0,
            'max_rings':0,
            'max_ring_size':0,
            'max_ring_states':0,
            'max_node_add_states':0,
            'max_node_connect_states':0,
            'atom_types':[],
            'ring_atom_types':[]
            }
    for mid,mft in tqdm(enumerate(mfts)):
        natoms=mft.natoms
        ncliques=mft.ncliques
        nrings=np.sum([1 for clique in mft.cliques if len(clique)>1])
        ring_size=np.max([len(clique) for clique in mft.cliques])
        bonds_num=np.array([len(fragmol.GetBonds()) for fragmol in mft.clique_mols if len(fragmol.GetAtoms())>1])
        ring_atom_types=[]
        ring_atoms=[]

        for atoms in mft.cliques:
            if len(atoms)>1:
                ring_atoms+=list(atoms)
        ring_atoms=list(set(ring_atoms))
        ring_natoms=[len(clique) for clique in mft.cliques]
        for atoms in mft.clique_inner_atoms:
            if len(atoms)>1:
                ring_atom_types+=list(atoms)
        ring_atom_types=list(set(ring_atom_types))
        for atomicnum in ring_atom_types:
            if atomicnum not in params['ring_atom_types']:
                params['ring_atom_types'].append(atomicnum)
        for atomicnum in mft.atoms:
            if atomicnum not in params['atom_types']:
                params['atom_types'].append(atomicnum)

        node_add_states=ncliques
        node_connect_states=ncliques
        ring_states=np.sum(bonds_num)+nrings*2

        if natoms > params['max_atoms']:
            params['max_atoms']=natoms

        if ncliques>params['max_cliques']:
            params['max_cliques']=ncliques

        if nrings >params['max_rings']:
            params['max_rings']=nrings

        if ring_size > params['max_ring_size']:
            params['max_ring_size']=ring_size

        if node_add_states>params['max_node_add_states']:
            params['max_node_add_states']=node_add_states

        if node_connect_states>params['max_node_connect_states']:
            params['max_node_connect_states']=node_connect_states

        if ring_states>params['max_ring_states']:
            params['max_ring_states']=ring_states
    print ('*'*80)
    print ('Dataset statistic params:')
    print (params)
    print ('*'*80)

    #GP.syssetting.possible_atom_types=params['atom_types']
    #GP.syssetting.possible_ringatom_types=params['ring_atom_types']
    GP.syssetting.max_ring_size=params['max_ring_size']
    GP.syssetting.max_atoms=params['max_atoms']
    GP.syssetting.max_cliques=params['max_cliques']
    GP.syssetting.max_rings=params['max_rings']
    GP.syssetting.max_node_add_states=params['max_node_add_states']
    GP.syssetting.max_node_connect_states=params['max_node_connect_states']
    GP.syssetting.max_ring_states=params['max_ring_states']
    GP.syssetting.update()
    return

def Statistic_dataset_params_from_MFTsflist(MFTsflist):
    params={
            'max_atoms':0,
            'max_cliques':0,
            'max_rings':0,
            'max_ring_size':0,
            'max_ring_states':0,
            'max_node_add_states':0,
            'max_node_connect_states':0,
            'atom_types':[],
            'ring_atom_types':[]
            }
    for fname in MFTsflist:
        with open(fname,'rb') as f:
            mfts=pickle.load(f)
        for mid,mft in tqdm(enumerate(mfts)):
            natoms=mft.natoms
            ncliques=mft.ncliques
            nrings=np.sum([1 for clique in mft.cliques if len(clique)>1])
            ring_size=np.max([len(clique) for clique in mft.cliques])
            bonds_num=np.array([len(fragmol.GetBonds()) for fragmol in mft.clique_mols if len(fragmol.GetAtoms())>1])
            ring_atom_types=[]
            ring_atoms=[]
            for atoms in mft.cliques:
                if len(atoms)>1:
                    ring_atoms+=list(atoms)
            ring_atoms=list(set(ring_atoms))
            ring_natoms=[len(clique) for clique in mft.cliques]
            for atoms in mft.clique_inner_atoms:
                if len(atoms)>1:
                    ring_atom_types+=list(atoms)
            ring_atom_types=list(set(ring_atom_types))
            for atomicnum in ring_atom_types:
                if atomicnum not in params['ring_atom_types']:
                    params['ring_atom_types'].append(atomicnum)
            for atomicnum in mft.atoms:
                if atomicnum not in params['atom_types']:
                    params['atom_types'].append(atomicnum)
 
            node_add_states=ncliques
            node_connect_states=ncliques
            ring_states=np.sum(bonds_num)+nrings*2
 
            if natoms > params['max_atoms']:
                params['max_atoms']=natoms
 
            if ncliques>params['max_cliques']:
                params['max_cliques']=ncliques
 
            if nrings >params['max_rings']:
                params['max_rings']=nrings
 
            if ring_size > params['max_ring_size']:
                params['max_ring_size']=ring_size
 
            if node_add_states>params['max_node_add_states']:
                params['max_node_add_states']=node_add_states
 
            if node_connect_states>params['max_node_connect_states']:
                params['max_node_connect_states']=node_connect_states
 
            if ring_states>params['max_ring_states']:
                params['max_ring_states']=ring_states
    print ('*'*80)
    print ('Dataset statistic params:')
    print (params)
    print ('*'*80)

    #GP.syssetting.possible_atom_types=params['atom_types']
    #GP.syssetting.possible_ringatom_types=params['ring_atom_types']
    GP.syssetting.max_ring_size=params['max_ring_size']
    GP.syssetting.max_atoms=params['max_atoms']
    GP.syssetting.max_cliques=params['max_cliques']
    GP.syssetting.max_rings=params['max_rings']

    GP.syssetting.max_node_add_states=params['max_node_add_states']
    GP.syssetting.max_node_connect_states=params['max_node_connect_states']
    GP.syssetting.max_ring_states=params['max_ring_states']
    GP.syssetting.update()

    return 

class MFTree_Dataset(Dataset):
    def __init__(self,MFTlist,name):
        super(Dataset,self).__init__()
        self.mftlist=MFTlist
        self.name=name
        self.max_ring_size=GP.syssetting.max_ring_size
        self.max_atoms=GP.syssetting.max_atoms
        self.max_cliques=GP.syssetting.max_cliques
        self.max_rings=GP.syssetting.max_rings
        self.max_ring_states=GP.syssetting.max_ring_states
        self.max_node_add_states=GP.syssetting.max_node_add_states
        self.max_node_connect_states=GP.syssetting.max_node_connect_states
        self.ring_atom_types=GP.syssetting.possible_ringatom_types
        self.mol_atom_types=GP.syssetting.possible_atom_types
        #print (self.max_atoms,self.max_ring_size,)
        self.nmols=len(self.mftlist)
        #self.get_dataset_params()
        return 

    def __len__(self):
        return len(self.mftlist)

    def check(self):
        checked_mfts=[]
        for i in range(len(self.mftlist)):
            #try:
                #print (self.mftlist[i].smi)
                self.getitem__(i)
                checked_mfts.append(self.mftlist[i])
            #except Exception as e :
            #    print (self.mftlist[i].smi+f'due to {e}')
        print (f'{len(checked_mfts)} passed dataset self-check!')
        with open(self.name+'.pickle','wb') as f:
            pickle.dump(checked_mfts,f)
        return 
    def __getitem__(self,idx):
        return self.getitem__(idx)
    def getitem__(self,idx):
            mft=copy.deepcopy(self.mftlist[idx])
            #try:
            mft.rearrange(mode=GP.trainsetting.rearrange_molgraph_mode,mix_rate=GP.trainsetting.rearrange_molgraph_mix_rate,ring_seq=GP.trainsetting.ring_seq)
            #except:
            #    print (mft.smi)
            mol_graph_states,mol_graph_apd,ring_graph_states,ring_graph_apd,mol_ring_connect_states,mol_ring_connect_apd=mft.get_action_states_graph()
            #print (len(ring_graph_states))
            Node_add_nstates=len(mol_graph_states)
            Node_add_mask=torch.zeros(GP.syssetting.max_node_add_states).long()
            Node_add_mask[:Node_add_nstates]=1
            Node_add_mask=Node_add_mask.view(-1,1)
            #print (self.max_node_add_states,Node_add_nstates)
            #print (list(random.sample(list(np.arange(Node_add_nstates,dtype=int)),self.max_node_add_states-Node_add_nstates)))
            times=self.max_node_add_states//Node_add_nstates
            addition=self.max_node_add_states%Node_add_nstates
     
            Node_add_sample_idx=[i for i in range(Node_add_nstates)]*times+\
                list(random.sample(list(np.arange(Node_add_nstates,dtype=int)),addition))
            #try: 
            Node_add_molgraph_nodes = torch.Tensor(
                np.array(
                    [
                        np.pad(
                            mol_graph_states[i][0],
                            ((0,self.max_atoms-len(mol_graph_states[i][0])),(0,0)),
                            'constant',
                            constant_values=0
                        )
                        for i in Node_add_sample_idx
                    ]
                    )
                )
            #except:
            #    print (mft.smi,times,Node_add_nstates,self.max_node_add_states,Node_add_sample_idx,mft.natoms)
            Node_add_molgraph_edges = torch.Tensor(
                np.array(
                    [
                        np.pad(
                                mol_graph_states[i][1],
                                ((0,0),(0,self.max_atoms-len(mol_graph_states[i][0])),(0,self.max_atoms-len(mol_graph_states[i][0]))),
                                'constant',
                                constant_values=0 
                            )
                        for i in Node_add_sample_idx
                        ]
                    )
                )
     
            Node_add_molgraph_add       = torch.Tensor(np.array([mol_graph_apd[i][0] for i in Node_add_sample_idx]))
            Node_add_molgraph_terminate = torch.Tensor(np.array([mol_graph_apd[i][1] for i in Node_add_sample_idx]))
            Node_add_molgraph_apd       = torch.cat((Node_add_molgraph_add,Node_add_molgraph_terminate),dim=1)
     
            Ring_gen_nstates=len(ring_graph_states)
            Ring_step_mask=torch.zeros(GP.syssetting.max_ring_states).long()
            Ring_step_mask[:Ring_gen_nstates]=1
            Ring_step_mask=Ring_step_mask.view(-1,1)

            if Ring_gen_nstates>0:
                times=self.max_ring_states//Ring_gen_nstates
                addition=self.max_ring_states%Ring_gen_nstates
                Ring_gen_sample_idx=[i for i in range(Ring_gen_nstates)]*times+\
                    random.sample(list(np.arange(Ring_gen_nstates,dtype=int)),addition)
                Ring_Gen_molgraph_nodes=torch.Tensor(
                    np.array(
                        [
                            np.pad(
                                    ring_graph_states[i][0],
                                    ((0,self.max_atoms-len(ring_graph_states[i][0])),(0,0)),
                                    'constant',
                                    constant_values=0
                                )
                            for i in Ring_gen_sample_idx
                        ]
                        )
                    )
                #print ([arr.shape for arr in Ring_Gen_molgraph_nodes])
                Ring_Gen_molgraph_edges=torch.Tensor(
                    np.array(
                        [
                            np.pad(
                                    ring_graph_states[i][1],
                                    ((0,0),(0,self.max_atoms-len(ring_graph_states[i][0])),(0,self.max_atoms-len(ring_graph_states[i][0]))),
                                    'constant',
                                    constant_values=0
                                ) 
                            for i in Ring_gen_sample_idx
                        ]
                        )
                    )
                
                Ring_Gen_ringgraph_nodes=torch.Tensor(
                    np.array(
                        [
                            np.pad(
                                    ring_graph_states[i][2],
                                    ((0,self.max_ring_size-len(ring_graph_states[i][2])),(0,0)),
                                    'constant',
                                    constant_values=0
                                ) 
                            for i in Ring_gen_sample_idx
                        ]
                        )
                    )
                #print ([ring_graph_states[i][3].shape for i in Ring_gen_sample_idx])
                Ring_Gen_ringgraph_edges=torch.Tensor(
                        np.array(
                            [
                            np.pad(
                                    ring_graph_states[i][3],
                                    ((0,0),(0,self.max_ring_size-len(ring_graph_states[i][2])),(0,self.max_ring_size-len(ring_graph_states[i][2]))),
                                    'constant',
                                    constant_values=0
                                ) 
                            for i in Ring_gen_sample_idx
                            ]
                        )
                    )
                #Ring_Gen_ringgraph_edges=torch.Tensor(np.array([ring_graph_states[i][3] for i in Ring_gen_sample_idx]))
                Ring_Gen_ringgraph_at=torch.Tensor(np.array([ring_graph_states[i][4] for i in Ring_gen_sample_idx]))
                Ring_graph_add=torch.Tensor(np.array([ring_graph_apd[i][0] for i in Ring_gen_sample_idx]))
                #print (Ring_graph_add.shape)
                Ring_graph_add=Ring_graph_add.view(self.max_ring_states,-1)
                Ring_graph_connect=torch.Tensor(np.array([ring_graph_apd[i][1] for i in Ring_gen_sample_idx])).view(self.max_ring_states,-1)
                Ring_graph_terminate=torch.Tensor(np.array([ring_graph_apd[i][2] for i in Ring_gen_sample_idx])).view(self.max_ring_states,-1)
                #print (Ring_graph_add.shape,Ring_graph_connect.shape,Ring_graph_terminate.shape)
                Ring_Gen_apd=torch.cat((Ring_graph_add,Ring_graph_connect,Ring_graph_terminate),dim=1)
            else:
                Ring_Gen_molgraph_nodes = torch.zeros((self.max_ring_states,GP.syssetting.max_atoms,GP.syssetting.n_node_features)).to(Node_add_molgraph_nodes)
                Ring_Gen_molgraph_edges = torch.zeros((self.max_ring_states,GP.syssetting.n_edge_features,GP.syssetting.max_atoms,GP.syssetting.max_atoms)).to(Node_add_molgraph_edges)
                Ring_Gen_ringgraph_nodes = torch.zeros((self.max_ring_states,GP.syssetting.max_ring_size,GP.syssetting.n_node_features)).to(Node_add_molgraph_nodes)
                Ring_Gen_ringgraph_edges = torch.zeros((self.max_ring_states,GP.syssetting.n_edge_features,GP.syssetting.max_ring_size,GP.syssetting.max_ring_size)).to(Node_add_molgraph_edges)
                Ring_Gen_ringgraph_at = torch.zeros((self.max_ring_states,GP.syssetting.n_clique_features))
                Ring_Gen_apd = torch.zeros((self.max_ring_states,np.prod(GP.syssetting.f_ring_add_dim)+np.prod(GP.syssetting.f_ring_connect_dim)+np.prod(GP.syssetting.f_ring_termination_dim))).to(Node_add_molgraph_apd)

            node_connect_nstates=len(mol_ring_connect_states)
            node_connect_step_mask=torch.zeros(GP.syssetting.max_node_connect_states).long()
            node_connect_step_mask[:node_connect_nstates]=1
            node_connect_step_mask=node_connect_step_mask.view(-1,1)
            #print ('++++',node_connect_nstates,self.max_node_connect_states)
            if node_connect_nstates>0:
                times=self.max_node_connect_states//node_connect_nstates
                addition=self.max_node_connect_states%node_connect_nstates
                node_connect_sample_idx=[i for i in range(node_connect_nstates)]*times+\
                    random.sample(list(np.arange(node_connect_nstates,dtype=int)),addition)
                node_connect_molgraph_nodes=torch.Tensor(
                        np.array(
                            [
                                np.pad(
                                    mol_ring_connect_states[i][0],
                                    ((0,self.max_atoms-len(mol_ring_connect_states[i][0])),(0,0)),
                                    'constant',
                                    constant_values=0 
                                )
                                for i in node_connect_sample_idx
                            ]
                        )
                    )
         
                node_connect_molgraph_edges = torch.Tensor(
                        np.array(
                            [
                                np.pad(
                                    mol_ring_connect_states[i][1],
                                    ((0,0),(0,self.max_atoms-len(mol_ring_connect_states[i][0])),(0,self.max_atoms-len(mol_ring_connect_states[i][0]))),
                                    'constant',
                                    constant_values=0 
                                    )
                                for i in node_connect_sample_idx
                            ]
                        )
                    )
         
                node_connect_ringgraph_nodes = torch.Tensor(
                        np.array(
                            [
                                np.pad(
                                    mol_ring_connect_states[i][2],
                                    ((0,self.max_ring_size-len(mol_ring_connect_states[i][2])),(0,0)),
                                    'constant',
                                    constant_values=0
                                )
                                for i in node_connect_sample_idx
                            ]
                        )
                    )
                
                node_connect_ringgraph_edges = torch.Tensor(
                        np.array(
                            [
                                np.pad(
                                    mol_ring_connect_states[i][3],
                                    ((0,0),(0,self.max_ring_size-len(mol_ring_connect_states[i][2])),(0,self.max_ring_size-len(mol_ring_connect_states[i][2]))),
                                    'constant',
                                    constant_values=0 
                                )
                                for i in node_connect_sample_idx
                            ]
                        )
                    )
         
                node_connect_joint=torch.Tensor(np.array([mol_ring_connect_apd[i][0] for i in node_connect_sample_idx])).view(self.max_node_connect_states,-1)
                node_connect_focused_ids=torch.zeros(self.max_node_connect_states).long().to(node_connect_joint).long()

            else:
                node_connect_molgraph_nodes = torch.zeros((self.max_node_connect_states,GP.syssetting.max_atoms,GP.syssetting.n_node_features)).to(Node_add_molgraph_nodes)
                node_connect_molgraph_edges = torch.zeros((self.max_node_connect_states,GP.syssetting.n_edge_features,GP.syssetting.max_atoms,GP.syssetting.max_atoms)).to(Node_add_molgraph_edges)
                node_connect_ringgraph_nodes = torch.zeros((self.max_node_connect_states,GP.syssetting.max_ring_size,GP.syssetting.n_node_features)).to(Node_add_molgraph_nodes)
                node_connect_ringgraph_edges = torch.zeros((self.max_node_connect_states,GP.syssetting.n_edge_features,GP.syssetting.max_ring_size,GP.syssetting.max_ring_size)).to(Node_add_molgraph_edges)
                node_connect_joint = torch.zeros((self.max_node_connect_states,np.prod(GP.syssetting.f_node_joint_dim))).to(Node_add_molgraph_apd)
                node_connect_focused_ids=torch.zeros(self.max_node_connect_states).long().to(node_connect_joint).long()
                
            return {
                    'Nadd_mg_nodes':Node_add_molgraph_nodes,
                    'Nadd_mg_edges':Node_add_molgraph_edges,
                    'Nadd_apd':Node_add_molgraph_apd,
                    'Nadd_step_masks':Node_add_mask,

                    'Rgen_mg_nodes':Ring_Gen_molgraph_nodes,
                    'Rgen_mg_edges':Ring_Gen_molgraph_edges,
                    'Rgen_rg_nodes':Ring_Gen_ringgraph_nodes,
                    'Rgen_rg_edges':Ring_Gen_ringgraph_edges,
                    'Rgen_rg_ftype':Ring_Gen_ringgraph_at,
                    'Rgen_apd':Ring_Gen_apd,
                    'Rgen_step_masks':Ring_step_mask,

                    'Njoint_mg_nodes':node_connect_molgraph_nodes,
                    'Njoint_mg_edges':node_connect_molgraph_edges,
                    'Njoint_rg_nodes':node_connect_ringgraph_nodes,
                    'Njoint_rg_edges':node_connect_ringgraph_edges,
                    'Njoint_apd':node_connect_joint,
                    'Njoint_focused_ids':node_connect_focused_ids,
                    'Njoint_step_masks':node_connect_step_mask
                    }
    
        
        
            
            


                
            

                

