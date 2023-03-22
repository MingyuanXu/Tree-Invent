from rdkit import Chem 
import numpy as np
import os ,json 
import torch 
import pickle 
def predeal_charged_molgraph(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol
class modelsetting:
    def __init__(self):
        self.graphnet_type = 'GGNN'
        # common used in all type of models
        self.mlp1_depth = 4 
        self.mlp2_depth = 4
        self.mlp1_hidden_dim = 150
        self.mlp2_hidden_dim = 150
        self.mlp1_dropout_p = 0.0
        self.mlp2_dropout_p = 0.0
        self.message_passes = 3
        self.message_size = 100
        self.hidden_node_features = 100
        #used in S2V, 
        self.enn_depth = 4
        self.enn_dropout_p = 0.0
        self.enn_hidden_dim = 250
        #used in AttS2V,AttGGNN,EMN
        self.att_depth = 4
        self.att_dropout_p = 0.0
        self.att_hidden_dim = 250
        #used in S2V,AttS2V
        self.s2v_lstm_computations = 3
        self.s2v_memory_size = 100
        #used in GGNN,AttGGNN,EMN
        self.gather_att_depth = 4
        self.gather_att_hidden_dim = 250
        self.gather_att_dropout_p = 0.0
        self.gather_emb_depth = 4
        self.gather_emb_hidden_dim = 250
        self.gather_emb_dropout_p = 0.0
        self.gather_width = 100
        self.msg_depth = 4
        self.msg_dropout_p = 0.0
        self.msg_hidden_dim = 250
        #used in EMN
        self.edge_emb_depth = 4 
        self.edge_emb_dropout_p = 0.0
        self.edge_emb_hidden_dim = 250
        self.edge_emb_size = 100
        self.device = 'cuda'
        self.big_negative = -1e6
        self.big_positive = 1e6
        
class trainsetting:
    def __init__(self):
        self.n_workers=2
        self.dataset_path='./datasets'
        self.rearrange_molgraph_mode='random'
        self.rearrange_molgraph_mix_rate=[1.0,0]
        self.batchsize=1000
        self.epochs=100
        self.initlr=1e-4
        self.shuffle=False
        self.cut=0.95
        self.optimizer='Adam'
        self.lr_patience=10
        self.lr_cooldown=10
        self.max_grad_norm=1.5
        self.max_rel_lr=1
        self.min_rel_lr=0.01
        self.nmols_per_pickle=25000
        self.ring_seq="dfs"

class syssetting:
    def __init__(self):
        self.max_atoms = 38
        self.ring_type_cover = 0.95
        self.implicit_Hs_types = [0, 1, 2, 3]
        self.possible_atom_types = [6,7,8,9,15,16,17,35,53]
        self.possible_ringatom_types = [6,7,8,16]
        self.chiral_tag_types = [0, 1, 2, 3]
        self.use_chiral_tag=False
        self.use_Hs=False
        self.formal_charge_types = [-2,-1,0,1,2]
        #self.bond_types = [Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC]
        self.bond_types = [Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE]
        self.ring_cover_rate = 0.99
        self.max_ring_size = 20
        self.max_cliques=0
        self.max_rings=0
        self.max_node_add_states=0
        self.max_node_connect_states=0
        self.max_ring_states=0
        self.n_edge_features = len(self.bond_types)
        self.n_node_features = len(self.possible_atom_types)+len(self.formal_charge_types)
        self.similarity_type='Morgan' #'rdkit'
        self.similarity_radius=2
        
    def update(self):
        self.ring_types_save_path = f'./datasets/descriptor_{self.ring_cover_rate}.csv'
        if os.path.exists(self.ring_types_save_path):
            with open(self.ring_types_save_path,'r') as f:
                self.ring_types=[line.strip().split()[0] for line in f.readlines()]
        else:
            self.ring_types=[]

        self.num_node_types=len(self.ring_types)+len(self.possible_atom_types)
        self.f_ring_add_dim=(self.max_ring_size,len(self.possible_ringatom_types),len(self.formal_charge_types),len(self.bond_types))
        self.f_ring_connect_dim=(self.max_ring_size,len(self.bond_types))
        self.f_ring_add_per_node=np.prod(self.f_ring_add_dim[1:])
        self.f_ring_connect_per_node=np.prod(self.f_ring_connect_dim[1:])
        self.f_ring_termination_dim=(1)
        self.f_graph_add_dim=(self.num_node_types)
        self.f_graph_termination_dim=(1)
        self.f_node_joint_dim=(self.max_atoms,len(self.bond_types))
        self.node_type_dict={}
        self.ringatom_type_dict={}
        for aid,i in enumerate(self.possible_atom_types):
            self.node_type_dict[i]=aid
        for aid,i in enumerate(self.possible_ringatom_types):
            self.ringatom_type_dict[i]=aid
        for rid,ringtype in enumerate(self.ring_types):
            self.node_type_dict[ringtype]=rid+len(self.possible_atom_types)
        self.n_edge_features = len(self.bond_types)
        self.n_node_features = len(self.possible_atom_types)+len(self.formal_charge_types)
        self.node_type_reverse_dict={v:k for k,v in self.node_type_dict.items()}
        #print (self.node_type_reverse_dict)
        self.f_node_dict={}
        for key,value in self.node_type_reverse_dict.items():
            if str(value)[0]=='R':
                var=value.split('-')
                rnum=int(var[0][1:])
                Cnum=int(var[1][1:])
                Nnum=int(var[2][1:])
                Onum=int(var[3][1:])
                Fnum=int(var[4][1:])
                Pnum=int(var[5][1:])
                Snum=int(var[6][1:])
                Clnum=int(var[7][2:])
                Brnum=int(var[8][2:])
                Inum=int(var[9][1:])
                Besnum=int(var[10][3:])
                ARnum=int(var[11][2:]) 
                f=[rnum,Cnum,Nnum,Onum,Fnum,Pnum,Snum,Clnum,Brnum,Inum,Besnum,ARnum]
            else:
                f=[0]*12
                aid=int(key)+1
                f[aid]=1
            self.f_node_dict[key]=f
            
        self.n_clique_features = 12
        self.ringat_to_molat=np.array([self.possible_atom_types.index(i) for i in self.possible_ringatom_types])

class rlsetting:
    def __init__(self):
        self.score_components=['similarities']
        self.score_weights=[1]
        #self.target_smiles=['Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)C(F)(F)F']
        self.target_smiles=['CO[C@H](C)[C@H](O)CC1CCOCC1']
        self.score_type='continuous'
        self.qsar_models_path='./qsar.pkl'
        self.max_gen_atoms=38
        self.score_thresholds=[1.0]
        self.tanimoto_k=0.7
        self.sigma=50
        self.vsigma=1.5
        self.ksigma=1.5
        self.acc_steps=1
        self.target_molfile=''
        self.temp_range=(1.0,1.2)
        self.temp_scheduler="same"
        self.unknown_fielter=False
        self.save_pic=True
    def update(self):
        if self.target_molfile!='':
            with open(self.target_molfile,'r') as f:
                self.target_smiles=[line.strip() for line in f.readlines()]
        if len(self.score_components)>1:
            n=len(self.score_components)-len(self.score_weights)
            for i in range(n):
                self.score_weights.append(1)
        if len(self.score_components)>1:
            n=len(self.score_components)-len(self.score_thresholds)
            for i in range(n):
                self.score_thresholds.append(1)
        return 
class ftsetting:
    def __init__(self):
        self.smi_path=''
        self.sdf_path=''
        self.converge_similarity=0.7

class leadoptsetting:
    def __init__(self):
        self.smi_path=''
        self.sdf_path=''

class clsetting:
    def __init__(self):
        self.smi_path=''
        self.sdf_path=''
        self.max_clsteps_per_iter=1000

class nadd_type_constrain:
    def __init__(self):
        self.max_ring_num_per_node=10
        self.min_ring_num_per_node=0
        self.max_aromatic_rings=10
        self.min_aromatic_rings=0
        self.min_branches=0
        self.max_branches=10
        self.max_anum_per_atomtype={'C':100,'N':100,'O':100,'F':100,'P':100,'S':100,'Cl':100,'Br':100,'I':100}
        self.min_anum_per_atomtype={'C':0,'N':0,'O':0,'F':0,'P':0,'S':0,'Cl':0,'Br':0,'I':0}
        self.max_heavy_atoms=100
        self.force_step=False
        self.specific_nodefile=None
        self.update()
        return

    def update(self):
        if self.specific_nodefile:
            with open(self.specific_nodefile,'rb') as f:
                self.specific_nodegraph=pickle.load(f)
                self.specific_nodegraph=predeal_charged_molgraph(self.specific_nodegraph)
                Chem.Kekulize(self.specific_nodegraph)
        else:
            self.specific_nodegraph=None
        return 

class node_conn_constrain:
    def __init__(self):
        self.saturation_atomid_list=[] # defined the non-anchor atomids
        self.constrain_connect_node_id=[] #defined the node to be connected
        self.constrain_connect_atom_id=[[]]
        self.constrain_connect_bond_type=[0,1,2] 
        self.constrain_connect_atomic_type=[6,7,8,9,15,16,17,35,53]
        self.anchor_before=-1 #define the anchor to connect the existed nodes, only used for specific rdkit mol nodes
        return 
    
class sample_constrain_setting:
    def __init__(self):
        self.max_node_steps=100
        self.max_ring_nodes=100
        self.temp=1.0
        self.node_constrain_basic=nadd_type_constrain()
        self.node_conn_constrain_basic=node_conn_constrain()
        self.constrain_step_dict={}
        self.ring_check_mode="only ring"
        return 
    
    def update(self):
        for i in range(self.max_node_steps+1):
            if str(i) not in self.constrain_step_dict.keys():
                self.constrain_step_dict[str(i)]={"node add":self.node_constrain_basic,"node conn":self.node_conn_constrain_basic}
            else:
                if "node add" not in self.constrain_step_dict[str(i)].keys():
                    self.constrain_step_dict[str(i)]["node add"]=self.node_constrain_basic
                if "node conn" not in self.constrain_step_dict[str(i)].keys():
                    self.constrain_step_dict[str(i)]["node conn"]=self.node_conn_constrain_basic 
        for i in range(self.max_node_steps):
            if self.constrain_step_dict[str(i)]["node add"].specific_nodegraph:
                for j in range(i+1):
                    self.constrain_step_dict[str(i)]["node add"].force_step=True
        return

class docksetting:
    def __init__(self):
        self.dock_input_path='.'
        self.target_pdb=''
        self.reflig_pdb=''
        self.backend='AutoDock-Vina' #"Glide"
        self.box_size=20
        self.low_threshold=-13.0
        self.high_threshold=-2.0
        self.k=0.25
        self.dockstream_root_path='./'
        self.vina_bin_path='./'
        self.glide_keywords={}
        self.ncores=10
        self.nposes=2
        self.grid_path=''
        self.glide_flags={}
        self.glide_ver='2017'
        self.glide_keywords={}
    def update(self):
        pass

class rocssetting:
    def __init__(self):
        self.cff_path=''
        self.reflig_sdf_path=''
        self.shape_w=0.5
        self.color_w=0.5
        self.sim_measure="Tanimoto"
    def update(self):
        pass

class samplesetting:
    def __init__(self):
        self.max_node_steps=100
        self.max_ring_num_per_node=3
        self.max_ring_nodes=3
        self.max_ring_heavy_atoms=100
        self.max_branches=3
        self.max_aromatic_rings=2
        self.min_aromatic_rings=0
        self.max_anum_per_atomtype={'C':100,'N':100,'O':100,'F':100,'P':100,'S':100,'Cl':100,'Br':100,'I':100}
        self.sample_control=False
        self.temp=1.0
    def update(self,node_type_dict):
        n_types=len(node_type_dict.keys())
        self.avaliable_node_mask=np.ones(n_types+1)
        if self.sample_control==True:
            for k,v in node_type_dict.items():
                if str(k)[0]=='R':
                    var=k.split('-')
                    rnum=int(var[0][1:])
                    Cnum=int(var[1][1:])
                    Nnum=int(var[2][1:])
                    Onum=int(var[3][1:])
                    Fnum=int(var[4][1:])
                    Pnum=int(var[5][1:])
                    Snum=int(var[6][1:])
                    Clnum=int(var[7][2:])
                    Brnum=int(var[8][2:])
                    Inum=int(var[9][1:])
                    Besnum=int(var[10][3:])
                    ARnum=int(var[11][2:])     
                    print (rnum,Cnum,Nnum,Onum,Fnum,Pnum,Snum,Clnum,Brnum,Inum,Besnum,ARnum)           
                    if rnum>self.max_ring_num_per_node:
                        self.avaliable_node_mask[v]=0                
                    if Cnum>self.max_anum_per_atomtype['C']:
                        self.avaliable_node_mask[v]=0
                    if Nnum>self.max_anum_per_atomtype['N']:
                        self.avaliable_node_mask[v]=0
                    if Onum>self.max_anum_per_atomtype['O']:
                        self.avaliable_node_mask[v]=0
                    if Fnum>self.max_anum_per_atomtype['F']:
                        self.avaliable_node_mask[v]=0
                    if Pnum>self.max_anum_per_atomtype['P']:
                        self.avaliable_node_mask[v]=0
                    if Snum>self.max_anum_per_atomtype['S']:
                        self.avaliable_node_mask[v]=0
                    if Clnum>self.max_anum_per_atomtype['Cl']:
                        self.avaliable_node_mask[v]=0
                    if Brnum>self.max_anum_per_atomtype['Br']:
                        self.avaliable_node_mask[v]=0
                    if Inum>self.max_anum_per_atomtype['I']:
                        self.avaliable_node_mask[v]=0
                    if Besnum>self.max_branches:
                        self.avaliable_node_mask[v]=0
                    if ARnum>self.max_aromatic_rings or ARnum<self.min_aromatic_rings:
                        self.avaliable_node_mask[v]=0
        self.avaliable_node_mask=torch.Tensor(self.avaliable_node_mask)
        return 
        

class GPARAMS:
    def __init__(self):
        self.modelsetting=modelsetting()
        self.syssetting=syssetting()
        self.trainsetting=trainsetting()
        self.rlsetting=rlsetting()
        self.sample_constrain_setting=sample_constrain_setting()
        self.ftsetting=ftsetting()
        self.leadoptsetting=leadoptsetting()
        self.clsetting=clsetting()
        self.docksetting=docksetting()
        self.rocssetting=rocssetting()
        
    def update(self):
        self.modelsetting.message_size=self.syssetting.n_node_features
        self.modelsetting.hidden_node_features=self.syssetting.n_node_features
        self.sample_constrain_setting.update()
        self.rlsetting.update()
        #print (self.sample_constrain_setting.__dict__)
        
GP=GPARAMS()

def Loaddict2obj(dict,obj):
    objdict=obj.__dict__
    for i in dict.keys():
        if i not in objdict.keys():
            print ("%s not is not a standard setting option!"%i)
        objdict[i]=dict[i]
    obj.__dict__==objdict

def UpdateGPARAMS(jsonfile):
    with open(jsonfile,'r') as f:
        jsondict=json.load(f)
        if 'model' in jsondict.keys():
            Loaddict2obj(jsondict['model'],GP.modelsetting)
            #print (GP.modelsetting.__dict__)
        if 'system' in jsondict.keys():
            Loaddict2obj(jsondict['system'],GP.syssetting)
            GP.syssetting.update()
            #print (GP.syssetting.__dict__["max_atoms"])
        if 'train' in jsondict.keys():
            Loaddict2obj(jsondict['train'],GP.trainsetting)
            #print (GP.trainsetting.__dict__)
        if 'rl' in jsondict.keys():
            Loaddict2obj(jsondict['rl'],GP.rlsetting)
        #if 'sample' in jsondict.keys():
        #    Loaddict2obj(jsondict['sample'],GP.samplesetting)
            #GP.samplesetting.update()
        if 'ft' in jsondict.keys():
            Loaddict2obj(jsondict['ft'],GP.ftsetting)
        if 'leadopt' in jsondict.keys():
            Loaddict2obj(jsondict['leadopt'],GP.leadoptsetting)
        if 'cl' in jsondict.keys():
            Loaddict2obj(jsondict['cl'],GP.clsetting)
        if 'sample_constrain' in jsondict.keys():
            Loaddict2obj(jsondict['sample_constrain'],GP.sample_constrain_setting)
            if "node_constrain_basic" in jsondict["sample_constrain"].keys():
                tmp_node_constrain=nadd_type_constrain()
                Loaddict2obj(jsondict["sample_constrain"]["node_constrain_basic"],tmp_node_constrain)
                GP.sample_constrain_setting.node_constrain_basic=tmp_node_constrain
            if "node_conn_constrain_basic" in jsondict["sample_constrain"].keys():
                tmp_node_conn_constrain=node_conn_constrain()
                Loaddict2obj(jsondict["sample_constrain"]["node_conn_constrain_basic"],tmp_node_constrain)
                GP.sample_constrain_setting.node_conn_constrain_basic=tmp_node_conn_constrain
                
            if 'constrain_step_dict' in jsondict['sample_constrain']:
                for key in jsondict['sample_constrain']["constrain_step_dict"].keys():
                    if "node add" in jsondict["sample_constrain"]["constrain_step_dict"][key].keys():
                        tmp_node_constrain=nadd_type_constrain()
                        Loaddict2obj(jsondict["sample_constrain"]["constrain_step_dict"][key]["node add"],tmp_node_constrain)
                        GP.sample_constrain_setting.constrain_step_dict[key]["node add"]=tmp_node_constrain
                        GP.sample_constrain_setting.constrain_step_dict[key]["node add"].update() 
                    if "node conn" in jsondict["sample_constrain"]["constrain_step_dict"][key].keys(): 
                        tmp_node_conn_constrain=node_conn_constrain()
                        Loaddict2obj(jsondict["sample_constrain"]["constrain_step_dict"][key]["node conn"],tmp_node_conn_constrain)
                        #print (key,jsondict["sample_constrain"]["constrain_step_dict"][key]["node conn"],tmp_node_conn_constrain.__dict__)
                        GP.sample_constrain_setting.constrain_step_dict[key]["node conn"]=tmp_node_conn_constrain
            GP.sample_constrain_setting.update()
            #print (GP.sample_constrain_setting)
        if 'docking' in jsondict.keys():
            Loaddict2obj(jsondict['docking'],GP.docksetting)
            #print (GP.docksetting)
        if 'rocs' in jsondict.keys():
            Loaddict2obj(jsondict['rocs'],GP.rocssetting)
            #print (GP.rocssetting)
        GP.update()
    return 
    
      