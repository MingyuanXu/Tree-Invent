from rdkit import Chem,RDConfig
from rdkit.Chem import rdmolfiles,FragmentCatalog,AllChem ,Draw

from tqdm import tqdm
import numpy as np
import networkx as nx 
import random,pickle,os 

from .features import *
from .graphroute import * 
from .rdkitutils import *
from ..comparm import * 

from .Utils3d_np import * 
def analysis_molecules_ringsystems(smis):
    frag_type_statistics={}
    frag_smis_statistics={}
    max_ring_sizes=[]
    max_single_ring_sizes=[]
    pid=0
    MFTs=[]
    max_ringnums=[]
    for smi in tqdm (smis):
        if smi:
            try:
                mol=Chem.MolFromSmiles(smi)
                mol=Neutralize_atoms(mol)
                Chem.Kekulize(mol)
                #smi=Chem.MolToSmiles(mol)
                MFTree=MolFragTree(mol,smi=smi)
                ssrlist=[list(x) for x in Chem.GetSymmSSSR(mol)]
                max_single_ring_size=np.max([len(x) for x in ssrlist])
                max_ring_size=0
                max_ringnum=0
                for fid,frag in enumerate(MFTree.clique_mols):
                    f=MFTree.f_cliques[fid]
                    frag_natoms=len(MFTree.clique_inner_f_atoms[fid])
                    if frag_natoms>max_ring_size:
                        max_ring_size=frag_natoms
                    descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}'
                    ringnum=f[0]
                    if ringnum>max_ringnum:
                        max_ringnum=ringnum
                    if frag_natoms>1:
                        if descriptor not in frag_type_statistics.keys():
                            frag_type_statistics[descriptor]=0
                            frag_smis_statistics[descriptor]=[]
                        frag_type_statistics[descriptor]+=1
                        frag_smis_statistics[descriptor].append(MFTree.clique_smis[fid])
                max_ring_sizes.append(max_ring_size)
                max_single_ring_sizes.append(max_single_ring_size)
                max_ringnums.append(max_ringnum)
            except Exception as e :
                print (e,smi)
    return max_ring_sizes,max_single_ring_sizes,frag_type_statistics,frag_smis_statistics,max_ringnums

def node_type_to_feature(typeid):
    descriptor=GP.syssetting.node_type_reverse_dict[typeid]
    val=descriptor.split('-')
    ring_num=int(val[0][1:])
    Cnum=int(val[1][1:])
    Nnum=int(val[2][1:])
    Onum=int(val[3][1:])
    Fnum=int(val[4][1:])
    Pnum=int(val[5][1:])
    Snum=int(val[6][1:])
    Clnum=int(val[7][2:])
    Brnum=int(val[8][2:])
    Inum=int(val[9][1:])
    Besnum=int(val[10][3:])
    ARnum=int(val[11][2:])
    return np.array([ring_num,Cnum,Nnum,Onum,Fnum,Pnum,Snum,Clnum,Brnum,Inum,Besnum,ARnum])

def Find_ring_system(mol,mode='single-rings'):
    smileslist=[]
    small_rings=[list(x) for x in Chem.GetSymmSSSR(mol)]
    if mode=='single-rings':
        return (small_rings)
    nrings=len(small_rings)
    natoms=mol.GetNumAtoms()
    atoms=[atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_adjacency=np.zeros((GP.syssetting.n_edge_features,natoms,natoms))

    for bond in mol.GetBonds():
        a1=bond.GetBeginAtom().GetIdx()
        a2=bond.GetEndAtom().GetIdx()
        bt=bond.GetBondType() 
        ch=GP.syssetting.bond_types.index(bt)
        atom_adjacency[ch,a1,a2]=1
        atom_adjacency[ch,a2,a1]=1

    atomflags=[False for i in range(natoms)]
    for i in range(nrings):
        for atomi in small_rings[i]:
            atomflags[atomi]=True
    cliques=small_rings+[[i] for i in range(natoms) if not atomflags[i]]
    ncliques=len(cliques)
    cliques_adjacency=np.zeros((ncliques,ncliques))
    for i,cliquei in enumerate(small_rings):
        for j,cliquej in enumerate(small_rings):
            for atomi in cliquei:
                for atomj in cliquej:
                    if atomi==atomj:
                        cliques_adjacency[i,j]=1
                        cliques_adjacency[j,i]=1
        for j,cliquej in enumerate(cliques[nrings:]):
            for atomj in cliquej:
                if np.sum(atom_adjacency[1:,atomi,atomj])==1:
                    cliques_adjacency[i,j+nrings]=1
                    cliques_adjacency[j+nrings,i]=1
    cliques_edges=[]
    for i in range(ncliques):
        for j in range(i+1,ncliques):
            if cliques_adjacency[i,j]==1:
                cliques_edges.append([i,j])
    cliques_graph=nx.Graph(cliques_edges)
    new_cliques=[]
    connected_cliques=[]
    for c in nx.connected_components(cliques_graph):
        nodeSet=cliques_graph.subgraph(c).nodes()
        connected_cliques+=nodeSet
        #print (nodeSet)
        new_cliques.append([])
        for i in nodeSet:
            new_cliques[-1]+=cliques[i]
    for i in range(nrings):
        if i not in connected_cliques:
            new_cliques.append(cliques[i])
    new_cliques=[np.sort(list(set(clique))) for clique in new_cliques]
    return new_cliques

def Generate_fragmols_descriptors(frag):
    natoms=len(frag.GetAtoms())
    atomicnum=[atom.GetAtomicNum() for atom in frag.GetAtoms()]
    #print (natoms,atomicnum)
    f_frag=frag_features(frag)
    adjs=np.zeros((GP.syssetting.n_edge_features,natoms,natoms))
    for bond in frag.GetBonds():
        a1=bond.GetBeginAtom().GetIdx()
        a2=bond.GetEndAtom().GetIdx()
        bt=bond.GetBondType() 
        ch=GP.syssetting.bond_types.index(bt)
        adjs[ch,a1,a2]=1
        adjs[ch,a2,a1]=1
    #f_atoms=[]
    #for atom in frag.GetAtoms():
    #    f_atoms.append(atom_features(atom))
    #f_atoms=np.array(f_atoms)
    atomicnum=np.array(atomicnum)
    return  f_frag,adjs,atomicnum

def Generate_fragmols_descriptors_with_ids(molobj,aids):
    mol_natoms=len(molobj.GetAtoms())
    natoms=len(aids)
    atomicnum=[atom.GetAtomicNum() for aid,atom in enumerate(molobj.GetAtoms()) if aid in aids]
    ringfrag=Chem.RWMol(molobj)
    ringfrag.BeginBatchEdit()
    for i in range(mol_natoms):
        if i not in aids:
            ringfrag.RemoveAtom(i)
    ringfrag.CommitBatchEdit()
    frag=ringfrag.GetMol()
    ringnum= len(Chem.GetSymmSSSR(frag))
    ringatoms=[]
    for ring in Chem.GetSymmSSSR(frag):
        for i in ring:
            if i not in ringatoms:
                ringatoms.append(i)
    CNOFPSClBrI=[atomicnum.count(i) for i in [6,7,8,9,15,16,17,35,53]]
    if natoms>1:
        branches=natoms-len(ringatoms)
    else:
        branches=0
    num_aromatic_rings=Chem.rdMolDescriptors.CalcNumAromaticRings(frag)
    f_frag=[ringnum]+CNOFPSClBrI+[branches]+[num_aromatic_rings]
    adjs=np.zeros((GP.syssetting.n_edge_features,natoms,natoms))
    for bond in molobj.GetBonds():
        a1=bond.GetBeginAtom().GetIdx()
        a2=bond.GetEndAtom().GetIdx()
        if (a1 in aids) and (a2 in aids):
            aid1=list(aids).index(a1)
            aid2=list(aids).index(a2)
            bt=bond.GetBondType() 
            ch=GP.syssetting.bond_types.index(bt)
            adjs[ch,aid1,aid2]=1
            adjs[ch,aid2,aid1]=1
    return np.array(f_frag),adjs,np.array(atomicnum),frag


def Decompose_molecule_to_fragtree(mol,ifdraw=False,pngpath='./mol'):
    flag=True
    natoms=mol.GetNumAtoms()
    atoms=[atom.GetAtomicNum() for atom in mol.GetAtoms()]
    cliques=[]
    # cut frags
    rings=Find_ring_system(mol,mode='multi-rings')
    frags=[]
    if len(rings)>0:
        cliques+=rings
    flags=[0 for i in range(natoms)]
    for clique in cliques:
        for i in clique:
            flags[i]=1
    for i in range(natoms):
        if not flags[i]:
            cliques.append([i])
    atom_fs=[]
    for i,atom in enumerate(mol.GetAtoms()):
        atom_fs.append(atom_features(atom))
    #fragmols=Get_fragmols(mol,cliques)
    fragmols=[]
    frag_fs=[]
    frag_adjs=[]
    frag_f_atoms=[]
    for clique in cliques:
        tmp_f=[]
        for i in clique:
            tmp_f.append(atom_fs[i])
        frag_f_atoms.append(np.array(tmp_f))
    frag_smis=[]
    frag_atoms=[]

    for i in range(len(cliques)):
        frag_f,frag_adj,frag_atom,frag=Generate_fragmols_descriptors_with_ids(mol,cliques[i])
        fragmols.append(frag)
        frag_smi=Chem.MolToSmiles(fragmols[i])
        frag_fs.append(frag_f)# (ncliques,fatom_dim)
        frag_adjs.append(frag_adj) #(ncliques,4,natoms,natoms) need padding
        #frag_f_atoms.append(frag_f_atom) # (ncliques,natoms,fatom_dim)
        frag_atoms.append(frag_atom) #(ncliques,natoms) need padding
        frag_smis.append(frag_smi)
    
    if ifdraw:
        try:
            os.system('mkdir -p png')
            img=Draw.MolsToGridImage([Chem.MolFromSmiles(smi) for smi in [Chem.MolToSmiles(mol)]+frag_smis],molsPerRow=5,subImgSize=(250,250),legends=[f'{i},{fragsmi}' for i,fragsmi in enumerate([Chem.MolToSmiles(mol)]+frag_smis)])
            img.save(f'./png/{Chem.MolToSmiles(mol)}.png')
        except:
            pass
    
    #Get adjs
    atom_adjs=np.zeros((GP.syssetting.n_edge_features,natoms,natoms))
    ncliques=len(cliques) 
    clique_adjs=np.zeros((ncliques,ncliques))
    #for i in range(natoms):
    #    atom_adjs[:,i,i]=1
    
    for bond in mol.GetBonds():
        a1=bond.GetBeginAtom().GetIdx()
        a2=bond.GetEndAtom().GetIdx()
        bt=bond.GetBondType() 
        ch=GP.syssetting.bond_types.index(bt)
        atom_adjs[ch,a1,a2]=1
        atom_adjs[ch,a2,a1]=1
    
    #for i in range(ncliques):
    #    clique_adjs[i,i]=1
    clique_adjs_info=[]
    ncliques=len(cliques)
    clique_adjs_info=np.zeros((ncliques,ncliques,2))
    for i,cliquei in enumerate(cliques):
        for j,cliquej in enumerate(cliques):
            for aiid,atomi in enumerate(cliquei):
                for ajid,atomj in enumerate(cliquej):
                    if np.sum(atom_adjs[:,atomi,atomj])==1 or atomi==atomj:
                        clique_adjs[i,j]=1
                        clique_adjs[j,i]=1
                        clique_adjs_info[i][j]=np.array([ajid,atomj])
                        clique_adjs_info[j][i]=np.array([aiid,atomi])

    cliques=[np.array(clique) for clique in cliques]

    return atoms,cliques,atom_adjs,clique_adjs,clique_adjs_info,atom_fs,frag_fs,frag_adjs,frag_f_atoms,frag_atoms,frag_smis,fragmols
        
class MolFragTree():
    def __init__(self,rdkitmol,smi=''):
        self.mol=rdkitmol
        if smi=='':
            self.smi=Chem.MolToSmiles(rdkitmol)
        else:
            self.smi=smi
        #print (smi)
        
        self.atoms,self.cliques,self.atom_adjs,self.clique_adjs,self.clique_adjs_info,\
            self.f_atoms,self.f_cliques,self.clique_inner_adjs,self.clique_inner_f_atoms,self.clique_inner_atoms,self.clique_smis,\
                self.clique_mols=Decompose_molecule_to_fragtree(rdkitmol,ifdraw=False)
        self.cliques=[np.array(clique) for clique in self.cliques] 
        self.natoms=len(self.atoms)
        try:
            self.coords=np.array(rdkitmol.GetConformer(0).GetPositions())
        except:
            self.coords=np.zeros((self.natoms,3))
        #print (self.sub_clique_adjs)
        #print (self.cliques)
        self.atoms=np.array(self.atoms)
        self.f_atoms=np.array(self.f_atoms)
        self.f_cliques=np.array(self.f_cliques)
        self.ncliques=len(self.cliques)
 
    def permindex(self,mode='random',mix_rate=[0.5,0.5],ring_seq="dfs"):
        #print ('tree composition',self.cliques)
        if len(self.cliques)>1:
            if mode=='random':
                clique_startid=random.choice([i for i in range(self.ncliques)])
            elif mode=='mix':
                choice=np.random.choice(['a','b'],p=np.array(mix_rate))
                if choice=='a':
                    clique_startid=0
                else:
                    clique_startid=random.choice([i for i in range(self.ncliques)])
            else:
                clique_startid=0
            #print (clique_startid)
            clique_graph=nx.Graph()
            clique_bonds=[]
            for i in range(self.ncliques):
                for j in range(i+1,self.ncliques):
                    if self.clique_adjs[i,j]==1:
                        clique_bonds.append((i,j))
            clique_graph.add_edges_from(clique_bonds)
            clique_bfs_order=bfs_seq(clique_graph,clique_startid)
        else:
            clique_bfs_order=np.array([0])
        
        atom_order=[]
        clique_inner_order=[]
        for i in clique_bfs_order:
            sub_order=[]
            sub_bonds=[]
            if len(self.cliques[i])>1:
                sub_graph=nx.Graph()
                for si in range(len(self.cliques[i])):
                    for sj in range(si+1,len(self.cliques[i])):
                        #print (clique_bfs_order,i,si,sj,self.clique_inner_adjs[i].shape,len(self.cliques[i]))
                        if np.sum(self.clique_inner_adjs[i][:,si,sj])==1:
                            sub_bonds.append((si,sj))
                candidate_start_ids=[]
                if len(atom_order)==0:
                    candidate_start_ids=[si for si in range(len(self.cliques[i]))]
                else:
                    for si in range(len(self.cliques[i])):
                        if self.cliques[i][si] in atom_order:
                            candidate_start_ids.append(si)
                    if len(candidate_start_ids)==0:
                        for si in range(len(self.cliques[i])):
                            for j in range(len(atom_order)):
                                if np.sum(self.atom_adjs[:,self.cliques[i][si],atom_order[j]])==1:
                                    candidate_start_ids.append(si)
                        set(candidate_start_ids)

                sub_start_id=random.choice(candidate_start_ids)
                sub_graph.add_edges_from(sub_bonds)
                if self.f_cliques[i][0]>0:
                    if ring_seq=="dfs":
                        sub_order=dfs_seq(sub_graph,sub_start_id)
                    else:
                        sub_order=bfs_seq(sub_graph,sub_start_id)
                else:
                    sub_order=bfs_seq(sub_graph,sub_start_id)
                #sub_clique_atom_order=np.argsort(sub_clique_atom_order)    
                clique_inner_order.append(sub_order)
                for si in sub_order:
                    if self.cliques[i][si] not in atom_order:
                        atom_order.append(self.cliques[i][si])
            else:
                clique_inner_order.append([0])
                atom_order.append(self.cliques[i][0])
        return atom_order,np.array(clique_bfs_order),clique_inner_order

    def rearrange(self,mode='random',mix_rate=[0.5,0.5],ring_seq="dfs"):
        #Drawmols(self.mol,permindex=[],filename=f'{self.smi}_1.png',cliques=self.cliques)
        self.cliques=[np.array(clique) for clique in self.cliques]
        atom_order,clique_bfs_order,clique_inner_order=self.permindex(mode,mix_rate,ring_seq)
        #Drawmols(self.mol,permindex=atom_order,filename=f'{self.smi}_2.png',cliques=self.cliques)
        b2n_index={}
        n2b_index={}
        for aid,a in enumerate(atom_order):
            b2n_index[a]=aid
            n2b_index[aid]=a
        #atom_reorder=[int(i) for i in np.argsort(atom_order)]
        self.atoms=self.atoms[np.ix_(atom_order)]
        self.coords=self.coords[np.ix_(atom_order)]
        atom_perm_index=np.ix_(atom_order,atom_order)
        for i in range(GP.syssetting.n_edge_features):
            self.atom_adjs[i]=self.atom_adjs[i][atom_perm_index]
        self.f_atoms=self.f_atoms[np.ix_(atom_order)]
        clique_perm_index=np.ix_(clique_bfs_order,clique_bfs_order)

        self.clique_adjs=self.clique_adjs[clique_perm_index]

        self.f_cliques=self.f_cliques[np.ix_(clique_bfs_order)]
        self.clique_smis=[self.clique_smis[i] for i in clique_bfs_order]
        self.clique_mols=[self.clique_mols[i] for i in clique_bfs_order]
        self.cliques=[self.cliques[i] for i in clique_bfs_order]
        self.clique_inner_f_atoms=[np.array(self.clique_inner_f_atoms[i]) for i in clique_bfs_order]
        self.clique_inner_atoms=[self.clique_inner_atoms[i] for i in clique_bfs_order]
        self.clique_inner_adjs=[self.clique_inner_adjs[i] for i in clique_bfs_order]
        for cid,clique_order in enumerate(clique_inner_order):

            clique_inner_permindex=np.ix_(clique_order,clique_order)
            #print (clique_order,self.clique_inner_adjs[cid].shape)
            for i in range(GP.syssetting.n_edge_features):
                self.clique_inner_adjs[cid][i]=self.clique_inner_adjs[cid][i][clique_inner_permindex]
            self.clique_inner_f_atoms[cid]=self.clique_inner_f_atoms[cid][np.ix_(clique_order)]
            self.clique_inner_atoms[cid]=np.array(self.clique_inner_atoms[cid])[np.ix_(clique_order)]
            #print (self.cliques,self.cliques[cid],type(self.cliques[cid]),clique_order,np.ix_(clique_order))
            #print (self.cliques[cid][np.ix_(clique_order)])
            self.cliques[cid]=[b2n_index[c] for c in self.cliques[cid][np.ix_(clique_order)]]
        #print (self.mol,atom_order,type(atom_order),type(atom_order[0]))
        self.mol=Chem.rdmolops.RenumberAtoms(self.mol,[int(a) for a in atom_order])
        self.clique_adjs_info=np.zeros((self.ncliques,self.ncliques,2))
        connects=[]
        for i in range(self.ncliques):
            cliquei=self.cliques[i]
            for j in range(i):
                cliquej=self.cliques[j]
                for aiid,atomi in enumerate(cliquei):
                    for ajid,atomj in enumerate(cliquej):
                        if np.sum(self.atom_adjs[:,atomi,atomj])==1:
                            self.clique_adjs_info[i][j]=np.array([ajid,atomj])
                            self.clique_adjs_info[j][i]=np.array([aiid,atomi])
                            connects.append([i,j,ajid,atomj])
                            connects.append([j,i,aiid,atomi])

        return

    def cal_ics(self):
        print (self.atom_adjs)
        self.zmat=np.zeros((self.natoms,5))
        self.zmat=np_adjs_to_zmat2(self.atom_adjs)
        x1=self.coords[zmat[:,0]] 
        x2=self.coords[zmat[:,1]]
        x3=self.coords[zmat[:,2]]
        x4=self.coords[zmat[:,3]]
        self.ics=np.zeros(s(self.natoms,3))
        bonds-calc_bond(x1[1:],x2[1:])
        angles=calc_angle(x1[2:],x2[2:],x3[2:])
        dihedrals=calc_angle(x1[3:],x2[3:],x3[3:],x4[3:])
        self.ics[1:,0]=bonds
        self.ics[2:,1]=angles
        self.ics[3:,2]=dihedrals
        return 
    def perb_coordinates(self,ratio=0.05):
        pass
        return 
    
    def get_action_states_3D(self):
        mol_graph_states=[]
        focus_atom_ids=[]
        focus_atom_ics=[]
        for i in range(self.ncliques):
            graph_atoms_mask_node=np.zeros(self.natoms)
            graph_atoms=[]
            for j in range(i+1):
                graph_atoms+=self.cliques[j]
            for j in graph_atoms:
                graph_atoms_mask_node[j]=1
            graph_atoms_mask_node_2D=graph_atoms_mask_node.reshape(-1,1)*graph_atoms_mask_node.reshape(-1,1).T
            mol_graph_adjs=self.atom_adjs*graph_atoms_mask_node_2D
            mol_graph_nodes=self.f_atoms*graph_atoms_mask_node
            for j in self.cliques[i]:
                d3_atoms_mask_node=np.zeros(self.natoms)
                d3_atoms=[k for k in range(j)]
                d3_atoms_mask_node[:j]=1
                d3_atoms_mask_node_2D=d3_atoms_mask_node.reshape(-1,1)*d3_atoms_mask_node.reshape(-1,1).T
                mol_graph_nodes_3D=self.coords*d3_atoms_mask_node
                mol_graph_states.append([mol_graph_adjs,mol_graph_nodes,mol_graph_nodes_3D,d3_atom_mask_node,d3_atom_mask_node_2D])
                focus_atom_ids.append(self.zmat[j])
                focus_atoms_ics.append(self.ics[j])
    
    def get_action_states_graph_3D(self):
        #print ([type(f_atoms) for f_atoms in self.clique_inner_f_atoms])
        mol_graph_states=[]
        mol_graph_apd=[]
        ring_graph_states=[]
        ring_graph_apd=[]
        node_connect_states=[]
        node_connect_apd=[]

        for i in range(self.ncliques):
            graph_atoms_mask_node=np.zeros(self.natoms)
            graph_atoms=[]
            if i>0:
                for j in range(i):
                    graph_atoms+=list(self.cliques[j])
                for j in graph_atoms:
                    graph_atoms_mask_node[j]=1
            graph_atoms_mask_node_2D=graph_atoms_mask_node.reshape(-1,1)*graph_atoms_mask_node.reshape(-1,1).T
            mol_graph_adjs=self.atom_adjs*graph_atoms_mask_node_2D
            mol_graph_nodes=self.f_atoms*graph_atoms_mask_node.reshape(-1,1)
            
            f=self.f_cliques[i]
            if len(self.cliques[i])>1:
                descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}'
                #print (descriptor,GP.syssetting.node_type_dict.keys())
                if descriptor not in GP.syssetting.node_type_dict.keys():
                    print (f'Unsupport Node type for MFTree named {self.smi} with descriptor {descriptor}')
                    return None 
                else:
                    fid=GP.syssetting.node_type_dict[descriptor]
            else:
                fid=GP.syssetting.node_type_dict[self.clique_inner_atoms[i][0]]

            f_T=np.zeros(GP.syssetting.num_node_types)
            f_terminate=np.zeros(1)

            if f[0]==0 and len(self.cliques[i])==1:
                f_T[GP.syssetting.node_type_dict[self.clique_inner_atoms[i][0]]]=1
            else:
                f_T[GP.syssetting.node_type_dict[descriptor]]=1

            mol_graph_states.append([mol_graph_nodes,mol_graph_adjs])
            mol_graph_apd.append([f_T,f_terminate])
            
            if f[0]>0:
                for ii in range(len(self.cliques[i])):
                    atomi=self.cliques[i][ii]
                    atom=self.mol.GetAtomWithIdx(int(atomi))
                    ring_atoms_mask_node=np.zeros(len(self.cliques[i]))
                    ring_atoms_mask_node[:ii]=1
                    ring_atoms_mask_node_2D=ring_atoms_mask_node.reshape(-1,1)*ring_atoms_mask_node.reshape(-1,1).T
                    ring_graph_adjs=self.clique_inner_adjs[i]*ring_atoms_mask_node_2D
                    ring_graph_nodes=self.clique_inner_f_atoms[i]*ring_atoms_mask_node.reshape(-1,1)
                    f_ring_add=np.zeros(GP.syssetting.f_ring_add_dim)
                    f_ring_connect=np.zeros(GP.syssetting.f_ring_connect_dim)
                    f_ring_terminate=np.zeros(1)
                    atypeid=GP.syssetting.ringatom_type_dict[atom.GetAtomicNum()]
                    if ii!=0:
                        #print (np.sum(self.clique_inner_adjs[i],axis=0)[ii][:ii],self.cliques[i])
                        connect_atom_id=np.where(np.sum(self.clique_inner_adjs[i],axis=0)[ii][:ii])[0][-1]
                    else:
                        connect_atom_id=0

                    atom_formal_charge=atom.GetFormalCharge()
                    formal_charge_id=GP.syssetting.formal_charge_types.index(atom_formal_charge)
                    implicit_Hs_num=atom.GetImplicitValence()
                    implicit_Hs_id=GP.syssetting.implicit_Hs_types.index(implicit_Hs_num)
                    chiral_tag=int(atom.GetChiralTag())
                    chiral_id=GP.syssetting.chiral_tag_types.index(chiral_tag)
                    if ii!=0:
                        bond_type=self.mol.GetBondBetweenAtoms(int(atomi),int(self.cliques[i][connect_atom_id])).GetBondType()
                        bond_type_id=GP.syssetting.bond_types.index(bond_type)
                    else:
                        bond_type_id=0

                    #print (atypeid,connect_atom_id,formal_charge_id,implicit_Hs_id,chiral_id,bond_type_id,GP.syssetting.f_ring_add_dim)
                    if GP.syssetting.use_chiral_tag:
                        if GP.syssetting.use_Hs:
                            f_ring_add[connect_atom_id,atypeid,formal_charge_id,implicit_Hs_id,chiral_id,bond_type_id]=1
                        else:
                            f_ring_add[connect_atom_id,atypeid,formal_charge_id,chiral_id,bond_type_id]=1
                    else:
                        if GP.syssetting.use_Hs:
                            f_ring_add[connect_atom_id,atypeid,formal_charge_id,implicit_Hs_id,bond_type_id]=1
                        else:
                            f_ring_add[connect_atom_id,atypeid,formal_charge_id,bond_type_id]=1
                    #print (ii,type(ring_graph_nodes))
                    ring_graph_states.append([mol_graph_nodes,mol_graph_adjs,ring_graph_nodes,ring_graph_adjs,f])
                    ring_graph_apd.append([f_ring_add,f_ring_connect,f_ring_terminate])
                    if len(np.where(np.sum(self.clique_inner_adjs[i],axis=0)[ii][:ii])[0])>1: 
                        connect_atoms=np.where(np.sum(self.clique_inner_adjs[i],axis=0)[ii][:ii])[0][:-1]
                        ring_atoms_mask_node[ii]=1
                        ring_atoms_mask_node_2D[ii,connect_atom_id]=1
                        ring_atoms_mask_node_2D[connect_atom_id,ii]=1
                        for jj in range(len(connect_atoms)):
                            if jj!=0:
                                ring_atoms_mask_node_2D[ii,connect_atoms[jj-1]]=1
                                ring_atoms_mask_node_2D[connect_atoms[jj-1],ii]=1
                            ring_graph_adjs=self.clique_inner_adjs[i]*ring_atoms_mask_node_2D
                            ring_graph_nodes=self.clique_inner_f_atoms[i]*ring_atoms_mask_node.reshape(-1,1)
                            #print (ii,jj,type(ring_graph_nodes))
                            f_ring_add=np.zeros(GP.syssetting.f_ring_add_dim)
                            f_ring_connect=np.zeros(GP.syssetting.f_ring_connect_dim)
                            f_ring_terminate=np.zeros(1)
                            atomj=self.cliques[i][connect_atoms[jj]]
                            bond_type=self.mol.GetBondBetweenAtoms(int(atomi),int(atomj)).GetBondType()
                            bond_type_id=GP.syssetting.bond_types.index(bond_type)
                            f_ring_connect[connect_atoms[jj],bond_type_id]=1
                            ring_graph_states.append([mol_graph_nodes,mol_graph_adjs,ring_graph_nodes,ring_graph_adjs,f])
                            ring_graph_apd.append([f_ring_add,f_ring_connect,f_ring_terminate])

                f_ring_add=np.zeros(GP.syssetting.f_ring_add_dim)
                f_ring_connect=np.zeros(GP.syssetting.f_ring_connect_dim)
                f_ring_terminate=np.ones(1)   
                ring_graph_states.append([mol_graph_nodes,mol_graph_adjs,np.array(self.clique_inner_f_atoms[i]),np.array(self.clique_inner_adjs[i]),f])
                ring_graph_apd.append([f_ring_add,f_ring_connect,f_ring_terminate])
            if i>0:
                f_joint=np.zeros(GP.syssetting.f_node_joint_dim)
                focused_clique_id=np.where(self.clique_adjs[i][:i])[0][-1]
                joint_id=int(self.clique_adjs_info[i,focused_clique_id][1])
                bond_type=self.mol.GetBondBetweenAtoms(int(self.clique_adjs_info[i,focused_clique_id][1]),int(self.clique_adjs_info[focused_clique_id,i][1])).GetBondType()
                bond_type_id=GP.syssetting.bond_types.index(bond_type)
                f_joint[joint_id,bond_type_id]=1
                Added_nodes=np.array(self.clique_inner_f_atoms[i])
                Added_adjs=np.array(self.clique_inner_adjs[i])
                node_connect_states.append([mol_graph_nodes,mol_graph_adjs,Added_nodes,Added_adjs])
                node_connect_apd.append([f_joint])

        f_T=np.zeros(GP.syssetting.num_node_types)
        f_terminate=np.ones(1)
        mol_graph_states.append([self.f_atoms,self.atom_adjs])
        mol_graph_apd.append([f_T,f_terminate])        
        
        return mol_graph_states,mol_graph_apd,ring_graph_states,ring_graph_apd,node_connect_states,node_connect_apd

    def get_action_states_graph(self):
        #print ([type(f_atoms) for f_atoms in self.clique_inner_f_atoms])
        mol_graph_states=[]
        mol_graph_apd=[]
        ring_graph_states=[]
        ring_graph_apd=[]
        node_connect_states=[]
        node_connect_apd=[]

        for i in range(self.ncliques):
            graph_atoms_mask_node=np.zeros(self.natoms)
            graph_atoms=[]
            if i>0:
                for j in range(i):
                    graph_atoms+=list(self.cliques[j])
                for j in graph_atoms:
                    graph_atoms_mask_node[j]=1
            graph_atoms_mask_node_2D=graph_atoms_mask_node.reshape(-1,1)*graph_atoms_mask_node.reshape(-1,1).T
            mol_graph_adjs=self.atom_adjs*graph_atoms_mask_node_2D
            mol_graph_nodes=self.f_atoms*graph_atoms_mask_node.reshape(-1,1)
            #print (graph_atoms)
            #print (i,mol_graph_nodes)
            #print (mol_graph_adjs)
            f=self.f_cliques[i]
            #print (f)
            if len(self.cliques[i])>1:
                descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}'
                #print (descriptor,GP.syssetting.node_type_dict.keys())
                if descriptor not in GP.syssetting.node_type_dict.keys():
                    print (f'Unsupport Node type for MFTree named {self.smi} with descriptor {descriptor}')
                    return None 
                else:
                    fid=GP.syssetting.node_type_dict[descriptor]
            else:
                fid=GP.syssetting.node_type_dict[self.clique_inner_atoms[i][0]]

            f_T=np.zeros(GP.syssetting.num_node_types)
            f_terminate=np.zeros(1)

            if f[0]==0 and len(self.cliques[i])==1:
                f_T[GP.syssetting.node_type_dict[self.clique_inner_atoms[i][0]]]=1
            else:
                f_T[GP.syssetting.node_type_dict[descriptor]]=1

            #mol_graph_nodes=self.f_atoms*graph_atoms_mask_node.reshape(-1,1)
            #mol_graph_adjs=self.atom_adjs*graph_atoms_mask_node_2D
            mol_graph_states.append([mol_graph_nodes,mol_graph_adjs])
            mol_graph_apd.append([f_T,f_terminate])
            
            if f[0]>0:
                for ii in range(len(self.cliques[i])):
                    atomi=self.cliques[i][ii]
                    atom=self.mol.GetAtomWithIdx(int(atomi))
                    ring_atoms_mask_node=np.zeros(len(self.cliques[i]))
                    ring_atoms_mask_node[:ii]=1
                    ring_atoms_mask_node_2D=ring_atoms_mask_node.reshape(-1,1)*ring_atoms_mask_node.reshape(-1,1).T
                    ring_graph_adjs=self.clique_inner_adjs[i]*ring_atoms_mask_node_2D
                    ring_graph_nodes=self.clique_inner_f_atoms[i]*ring_atoms_mask_node.reshape(-1,1)
                    f_ring_add=np.zeros(GP.syssetting.f_ring_add_dim)
                    f_ring_connect=np.zeros(GP.syssetting.f_ring_connect_dim)
                    f_ring_terminate=np.zeros(1)
                    atypeid=GP.syssetting.ringatom_type_dict[atom.GetAtomicNum()]
                    if ii!=0:
                        #print (np.sum(self.clique_inner_adjs[i],axis=0)[ii][:ii],self.cliques[i])
                        connect_atom_id=np.where(np.sum(self.clique_inner_adjs[i],axis=0)[ii][:ii])[0][-1]
                    else:
                        connect_atom_id=0

                    atom_formal_charge=atom.GetFormalCharge()
                    formal_charge_id=GP.syssetting.formal_charge_types.index(atom_formal_charge)
                    implicit_Hs_num=atom.GetImplicitValence()
                    implicit_Hs_id=GP.syssetting.implicit_Hs_types.index(implicit_Hs_num)
                    chiral_tag=int(atom.GetChiralTag())
                    chiral_id=GP.syssetting.chiral_tag_types.index(chiral_tag)
                    if ii!=0:
                        bond_type=self.mol.GetBondBetweenAtoms(int(atomi),int(self.cliques[i][connect_atom_id])).GetBondType()
                        bond_type_id=GP.syssetting.bond_types.index(bond_type)
                    else:
                        bond_type_id=0

                    #print (atypeid,connect_atom_id,formal_charge_id,implicit_Hs_id,chiral_id,bond_type_id,GP.syssetting.f_ring_add_dim)
                    if GP.syssetting.use_chiral_tag:
                        if GP.syssetting.use_Hs:
                            f_ring_add[connect_atom_id,atypeid,formal_charge_id,implicit_Hs_id,chiral_id,bond_type_id]=1
                        else:
                            f_ring_add[connect_atom_id,atypeid,formal_charge_id,chiral_id,bond_type_id]=1
                    else:
                        if GP.syssetting.use_Hs:
                            f_ring_add[connect_atom_id,atypeid,formal_charge_id,implicit_Hs_id,bond_type_id]=1
                        else:
                            f_ring_add[connect_atom_id,atypeid,formal_charge_id,bond_type_id]=1
                    #print (ii,type(ring_graph_nodes))
                    ring_graph_states.append([mol_graph_nodes,mol_graph_adjs,ring_graph_nodes,ring_graph_adjs,f])
                    ring_graph_apd.append([f_ring_add,f_ring_connect,f_ring_terminate])
                    if len(np.where(np.sum(self.clique_inner_adjs[i],axis=0)[ii][:ii])[0])>1: 
                        connect_atoms=np.where(np.sum(self.clique_inner_adjs[i],axis=0)[ii][:ii])[0][:-1]
                        ring_atoms_mask_node[ii]=1
                        ring_atoms_mask_node_2D[ii,connect_atom_id]=1
                        ring_atoms_mask_node_2D[connect_atom_id,ii]=1
                        for jj in range(len(connect_atoms)):
                            if jj!=0:
                                ring_atoms_mask_node_2D[ii,connect_atoms[jj-1]]=1
                                ring_atoms_mask_node_2D[connect_atoms[jj-1],ii]=1
                            ring_graph_adjs=self.clique_inner_adjs[i]*ring_atoms_mask_node_2D
                            ring_graph_nodes=self.clique_inner_f_atoms[i]*ring_atoms_mask_node.reshape(-1,1)
                            #print (ii,jj,type(ring_graph_nodes))
                            f_ring_add=np.zeros(GP.syssetting.f_ring_add_dim)
                            f_ring_connect=np.zeros(GP.syssetting.f_ring_connect_dim)
                            f_ring_terminate=np.zeros(1)
                            atomj=self.cliques[i][connect_atoms[jj]]
                            bond_type=self.mol.GetBondBetweenAtoms(int(atomi),int(atomj)).GetBondType()
                            bond_type_id=GP.syssetting.bond_types.index(bond_type)
                            f_ring_connect[connect_atoms[jj],bond_type_id]=1
                            ring_graph_states.append([mol_graph_nodes,mol_graph_adjs,ring_graph_nodes,ring_graph_adjs,f])
                            ring_graph_apd.append([f_ring_add,f_ring_connect,f_ring_terminate])

                f_ring_add=np.zeros(GP.syssetting.f_ring_add_dim)
                f_ring_connect=np.zeros(GP.syssetting.f_ring_connect_dim)
                f_ring_terminate=np.ones(1)   
                ring_graph_states.append([mol_graph_nodes,mol_graph_adjs,np.array(self.clique_inner_f_atoms[i]),np.array(self.clique_inner_adjs[i]),f])
                ring_graph_apd.append([f_ring_add,f_ring_connect,f_ring_terminate])
            if i>0:
                f_joint=np.zeros(GP.syssetting.f_node_joint_dim)
                focused_clique_id=np.where(self.clique_adjs[i][:i])[0][-1]
                joint_id=int(self.clique_adjs_info[i,focused_clique_id][1])
                #print ('++++',joint_id)
                bond_type=self.mol.GetBondBetweenAtoms(int(self.clique_adjs_info[i,focused_clique_id][1]),int(self.clique_adjs_info[focused_clique_id,i][1])).GetBondType()
                bond_type_id=GP.syssetting.bond_types.index(bond_type)
                f_joint[joint_id,bond_type_id]=1
                Added_nodes=np.array(self.clique_inner_f_atoms[i])
                Added_adjs=np.array(self.clique_inner_adjs[i])
                node_connect_states.append([mol_graph_nodes,mol_graph_adjs,Added_nodes,Added_adjs])
                node_connect_apd.append([f_joint])

        f_T=np.zeros(GP.syssetting.num_node_types)
        f_terminate=np.ones(1)
        mol_graph_states.append([self.f_atoms,self.atom_adjs])
        mol_graph_apd.append([f_T,f_terminate])
        
        return mol_graph_states,mol_graph_apd,ring_graph_states,ring_graph_apd,node_connect_states,node_connect_apd
    
    def get_CL_states(self):
        molsteps=[]
        clique_pts=[]
        clique_tree_steps=[]

        for i in range(self.ncliques-1):
            if len(self.cliques[i])>1 or len(self.cliques[i+1])>1:
                clique_pts.append(i)
                clique_tree_steps.append((self.f_cliques[:i+1],self.clique_adjs[:i+1,:i+1]))
                
        for i in range(self.ncliques-1):
            graph_atoms_mask_node=np.zeros(self.natoms)
            graph_atoms=[]
            if i in clique_pts:
                for j in range(i+1):
                    graph_atoms+=list(self.cliques[j])
                molstep=Chem.RWMol(self.mol)
                molstep.BeginBatchEdit()
                for j in range(self.natoms):
                    if j not in graph_atoms:
                        molstep.RemoveAtom(j)
                molstep.CommitBatchEdit()
                tmpmol=molstep.GetMol()
                tmpmol=Neutralize_atoms(tmpmol)
                #hem.Kekulize(tmpmol)
                Chem.SanitizeMol(tmpmol)
                molsteps.append(tmpmol)
        molsteps.append(self.mol)
        clique_tree_steps.append((self.f_cliques,self.clique_adjs))
        return molsteps,clique_tree_steps
    
    def get_avaliable_node_mask(self):
        n_types=len(GP.syssetting.node_type_dict.keys())
        avaliable_node_mask=np.zeros(n_types+1)
        avaliable_node_mask[-1]=1
        with open('./datasets/Frag_label_smis_0.99.pickle','rb') as f:
            frag_label_smis_dict=pickle.load(f)
        frag_smis=[]
        frag_labels=[]
        for key in frag_label_smis_dict.keys():
            frag_smis+=frag_label_smis_dict[key]
            frag_labels+=[key]*len(frag_label_smis_dict[key])
        #print (frag_label_smis_dict)
        for i in range(self.ncliques):
            f=self.f_cliques[i]
            #print (f)
            if len(self.cliques[i])>1:
                descriptor=f'R{f[0]}-C{f[1]}-N{f[2]}-O{f[3]}-F{f[4]}-P{f[5]}-S{f[6]}-Cl{f[7]}-Br{f[8]}-I{f[9]}-Bes{f[10]}-AR{f[11]}'
                #print (descriptor,GP.syssetting.node_type_dict.keys())
                if descriptor in GP.syssetting.node_type_dict.keys():
                    fid=GP.syssetting.node_type_dict[descriptor]
                    avaliable_node_mask[fid]=1
                else:
                    clique_mol=self.clique_mols[i]
                    print (f'Unsupport ring type {descriptor} corresponding to {Chem.MolToSmiles(clique_mol)}')
                    similarity_array=np.zeros(len(frag_smis))
                    for sid,smi in enumerate(frag_smis):
                        mol=Chem.MolFromSmiles(smi)
                        if mol:
                            similarity=tanimoto_similarities(clique_mol,mol)
                            similarity_array[sid]=similarity
                    max_ids=np.argsort(-similarity_array)[:5]
                    temp_labels=[frag_labels[a] for a in max_ids]
                    for label in temp_labels:
                        fid=GP.syssetting.node_type_dict[descriptor]
                        avaliable_node_mask[fid]=1
            else:
                fid=GP.syssetting.node_type_dict[self.clique_inner_atoms[i][0]]
                #avaliable_node_mask[fid]=1
            avaliable_node_mask=torch.Tensor(avaliable_node_mask)
        return avaliable_node_mask            

    def get_3Daction_states(self):
        #print ([type(f_atoms) for f_atoms in self.clique_inner_f_atoms])
        clique_states=[]
        clique_focus=[]
        return 

def Gclique_To_Constrain(fclique,clique_adjs,ring_replace_only=False):
    ncliques=len(fclique)
    constrains={}
    for i in range(ncliques):
        constrains[str(i)]={}
        print (fclique[i])
        rnum,Cnum,Nnum,Onum,Fnum,Pnum,Snum,Clnum,Brnum,Inum,Branches,Aromatics=list(fclique[i])
        
        nadd_constrain=nadd_type_constrain() 
        nadd_constrain.max_ring_num_per_node=rnum
        nadd_constrain.min_ring_num_per_node=rnum
        nadd_constrain.max_aromatic_rings=Aromatics
        nadd_constrain.min_aromatic_rings=Aromatics
        nadd_constrain.max_branches=Branches
        nadd_constrain.min_branches=Branches
        if ring_replace_only:
            if np.sum(fclique[i][1:-2])>1:
                nadd_constrain.max_anum_per_atomtype={'C':100,'N':100,'O':100,'F':100,'P':100,'S':100,'Cl':100,'Br':100,'I':100}
                nadd_constrain.min_anum_per_atomtype={'C':0,'N':0,'O':0,'F':0,'P':0,'S':0,'Cl':0,'Br':0,'I':0}
            else:
                nadd_constrain.max_anum_per_atomtype={'C':Cnum,'N':Nnum,'O':Onum,'F':Fnum,'P':Pnum,'S':Snum,'Cl':Clnum,'Br':Brnum,'I':Inum}
                nadd_constrain.min_anum_per_atomtype={'C':Cnum,'N':Nnum,'O':Onum,'F':Fnum,'P':Pnum,'S':Snum,'Cl':Clnum,'Br':Brnum,'I':Inum}
        else:
            if np.sum(fclique[i][1:-2])>1:
                nadd_constrain.max_anum_per_atomtype={'C':Cnum,'N':Nnum,'O':Onum,'F':Fnum,'P':Pnum,'S':Snum,'Cl':Clnum,'Br':Brnum,'I':Inum}
                nadd_constrain.min_anum_per_atomtype={'C':Cnum,'N':Nnum,'O':Onum,'F':Fnum,'P':Pnum,'S':Snum,'Cl':Clnum,'Br':Brnum,'I':Inum}
            else:
                nadd_constrain.max_anum_per_atomtype={'C':1,'N':1,'O':1,'F':1,'P':1,'S':1,'Cl':1,'Br':1,'I':1}
                nadd_constrain.min_anum_per_atomtype={'C':0,'N':0,'O':0,'F':0,'P':0,'S':0,'Cl':0,'Br':0,'I':0}
        nadd_constrain.force_step=True
        print (nadd_constrain.__dict__)
        nconn_constrain=node_conn_constrain()
        if i>0:
            #print (clique_adjs[i,:i],np.nonzero(clique_adjs[i,:i]))
            nconn_constrain.constrain_connect_node_id=[np.nonzero(clique_adjs[i,:i])[0][0]]
            print (nconn_constrain.__dict__)
        constrains[str(i)]["node add"]=nadd_constrain
        constrains[str(i)]["node conn"]=nconn_constrain
    return constrains



                  


                            


                    

                    
            
            
