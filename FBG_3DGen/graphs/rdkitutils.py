from rdkit.Chem.Draw import rdMolDraw2D
#from IPython.display import Image
import copy
import numpy as np
from rdkit import Chem 
from rdkit.Chem import AllChem,rdmolfiles,rdFMCS,Draw
import copy
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from tqdm import tqdm 
from ..comparm import *
from .features import *

def tanimoto_similarities(mol1,mol2):
    if GP.syssetting.similarity_type=='Morgan':
        fp1 = AllChem.GetMorganFingerprint(mol1, GP.syssetting.similarity_radius, useCounts=True, useFeatures=True)
        fp2 = AllChem.GetMorganFingerprint(mol2, GP.syssetting.similarity_radius, useCounts=True, useFeatures=True)
    similarity= DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity

def match_substructures(mols,smarts=[],submols=[]):
    submols+=[Chem.MolFromSmarts(submol) for subst in smarts if Chem.MolFromSmarts(subst)]
    print (submols)
    if len(submols)>0:
        matches=[]
        for mol in mols:
            if mol:
                match = any([mol.HasSubstructMatch(submol) for submol in submols])
                if match:
                    matches.append(1)
                else:
                    matches.append(0)
            else:
                matches.append(0)
        return np.array(matches)
    else:
        return np.zeros(len(mols))

def mcs_similarity(mols,smarts=[],submols=[]):
    similarity_scores=[]
    submols+=[Chem.MolFromSmarts(submol) for subst in smarts if Chem.MolFromSmarts(subst)]
    for mol in mols:
        max_max_similarity=0
        try:
            for submol in submols:
                #print ([mol,submol],[Chem.MolToSmiles(mol),Chem.MolToSmiles(submol)])
                #mcs=rdFMCS.FindMCS([mol,submol],ringMatchesRingOnly=True,comleteRingsOnly=True)
                mcs=rdFMCS.FindMCS([mol,submol])
                #print (mcs.smartsString)
                patt=Chem.MolFromSmarts(mcs.smartsString)
                matched_substructures=mol.GetSubstructMatches(patt)
                #print (matched_substructures)
                max_similarity=0
                msubsets=Get_fragmols(mol,matched_substructures)
                for msub in msubsets:
                #    print (matched_substruct)
                #    msubst=Get_fragmols(mol,matched_substruct)
                    similarity=tanimoto_similarities(msub,submol)
                    if similarity>max_similarity:
                        max_similarity=similarity
                if max_similarity>max_max_similarity:
                    max_max_similarity=max_similarity
        except:
            max_max_similarity=0
        similarity_scores.append(max_max_similarity)
    return np.array(similarity_scores)

def Change_mol_xyz(rdkitmol,coords):
    molobj=copy.deepcopy(rdkitmol)
    conformer=molobj.GetConformer()
    id=conformer.GetId()
    for cid,xyz in enumerate(coords):
        conformer.SetAtomPosition(cid,Point3D(float(xyz[0]),float(xyz[1]),float(xyz[2])))
    conf_id=molobj.AddConformer(conformer)
    molobj.RemoveConformer(id)
    return molobj

def Gen_ETKDG_structures(rdkitmol,nums=1,basenum=50,mode='opt+lowest',withh=False,ifwrite=False,path='./mol'):
    mol=copy.deepcopy(rdkitmol)
    mol_h=Chem.AddHs(mol)
    confids=AllChem.EmbedMultipleConfs(mol_h,basenum)
    confs=[]
    energies=[]
    mollist=[]
    for cid,c in enumerate(confids):
        conformer=mol_h.GetConformer(c)
        tmpmol=Chem.Mol(mol_h)
        ff=AllChem.UFFGetMoleculeForceField(tmpmol)
        if 'opt' in mode:
            ff.Minimize()
        uffenergy=ff.CalcEnergy()
        energies.append(uffenergy)
        if not withh:
            tmpmol=Chem.RemoveHs(tmpmol) 
        optconf=tmpmol.GetConformer(0).GetPositions()
        confs.append(optconf)
        mollist.append(tmpmol)
    lowest_ids=np.argsort(energies)
    lowest_confs=[confs[i] for i in lowest_ids[:nums]]
    if ifwrite:
        for i in lowest_ids[:nums]:
            rdmolfiles.MolToMolFile(mollist[i],f'{path}.mol2')
    return lowest_confs
        
def Drawmols(rdkitmol,filename='Mol.png',permindex=[],cliques=[]):
    reindex=np.zeros(len(permindex))
    for pid,p in enumerate(permindex):
        reindex[p]=pid 
    mol=copy.deepcopy(rdkitmol)
    Chem.rdDepictor.Compute2DCoords(mol)
    hatomlist=[]
    hbondlist=[]
    colors=[(1,1,0),(1,0,1),(1,0,0),(0,1,1),(0,1,0),(0,0,1)]
    if len(cliques)>0:
        for clique in cliques:
            if len(clique)>1:
                clique_bonds=[]
                hatomlist.append([int(a) for a in clique])
                for bond in mol.GetBonds():
                    a1=bond.GetBeginAtom().GetIdx()
                    a2=bond.GetEndAtom().GetIdx()
                    if a1 in clique and a2 in clique:
                        clique_bonds.append(bond.GetIdx())
                hbondlist.append(clique_bonds)
        atom_colors={}
        bond_colors={}
        atomlist=[]
        bondlist=[]
        for i,(hl_atom,hl_bond) in enumerate(zip(hatomlist,hbondlist)):
            #print (hl_atom,hl_bond)
            hl_atom=list(hl_atom)
            for at in hl_atom:
                atom_colors[at]=colors[i%6]
                atomlist.append(at)
            for bt in hl_bond:
                bond_colors[bt]=colors[i%6]
                bondlist.append(bt)
    """
    ri = mol.GetRingInfo()
    ringatomlist=[]
    ringbondlist=[]
    for ring in ri.AtomRings():
        ringatomlist.append(ring)
    for bond in ri.BondRings():
        ringbondlist.append(bond)
    print (zip(ringatomlist,ringbondlist))
    colors=[(1,1,0),(1,0,1),(1,0,0),(0,1,1),(0,1,0),(0,0,1)]
    atom_colors={}
    bond_colors={}
    atomlist=[]
    bondlist=[]
    
    for i,(hl_atom,hl_bond) in enumerate(zip(ringatomlist,ringbondlist)):
        print (hl_atom,hl_bond)
        hl_atom=list(hl_atom)
        for at in hl_atom:
            atom_colors[at]=colors[i%6]
            atomlist.append(at)
        for bt in hl_bond:
            bond_colors[bt]=colors[i%6]
            bondlist.append(bt)
    """
    #for clique in cliques:
    #    Chem.Mol

    options=rdMolDraw2D.MolDrawOptions()
    options.addAtomIndices=True
    draw=rdMolDraw2D.MolDraw2DCairo(500,500)
    for i in range(len(reindex)):
        mol.GetAtomWithIdx(i).SetProp("atomNote",':'+str(int(reindex[i])))
    draw.SetDrawOptions(options)
    #print (type(atomlist[0]),type(atom_colors),type(bondlist[0]),type(bond_colors))
    rdMolDraw2D.PrepareAndDrawMolecule(draw,mol,highlightAtoms=atomlist,
                                                highlightAtomColors=atom_colors,
                                                highlightBonds=bondlist,
                                                highlightBondColors=bond_colors)
    draw.FinishDrawing()
    draw.WriteDrawingText(filename)

def SmilesToSVG(smiles,legends=None,fname='mol.svg'):
    mols=[]
    vlegends=[]
    for sid,smi in enumerate(smiles):
        mol=Chem.MolFromSmiles(smi)
        if mol:
            Chem.AllChem.Compute2DCoords(mol)
            mols.append(mol)
            if legends:
                vlegends.append(legends[sid])
    img=Draw.MolsToGridImage(mols,legends=vlegends,molsPerRow=5,subImgSize=(250,250),useSVG=True)
    with open (fname,'w') as f:
        f.write(img)
    return 

def Neutralize_atoms(mol):
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

def Get_fragmols(mol,cliques):
    natoms=mol.GetNumAtoms()
    frags=[]
    for clique in cliques:
        ringfrag=Chem.RWMol(mol)
        ringfrag.BeginBatchEdit()
        for i in range(natoms):
            if i not in clique:
                ringfrag.RemoveAtom(i)
        ringfrag.CommitBatchEdit()
        frag=ringfrag.GetMol()
        #Chem.Kekulize(frag)
        frag=Neutralize_atoms(frag)
        frags.append(frag)
    return frags

def Save_smiles_list(smileslist,filename):
    with open(filename,'w')  as f:
        for smiles in smileslist:
            if smiles:
                f.write(smiles+'\n')
            else:
                f.write('None\n')

def Load_smiles_list(filename):
    with open(filename,'r')  as f:
        smileslist=[line.strip() for line in f.readlines() if '.' not in line]
    return smileslist

def Save_smarts_list(smartslist,filename):
    with open(filename,'w')  as f:
        for smiles in smartslist:
            f.write(smiles+'\n')

def Load_smarts_list(filename):
    with open(filename,'r')  as f:
        smartslist=[line.strip() for line in f.readlines()]
    return smartslist

def analysis_molecules_properties(smis):
    molwts=[]
    qeds=[]
    tpsas=[]
    logps=[]
    hbas=[]
    hbds=[]
    for smi in smis:
        if smi:
            mol=Chem.MolFromSmiles(smi)
            qed=QED.qed(mol)
            logp  = Descriptors.MolLogP(mol)
            tpsa  = Descriptors.TPSA(mol)
            molwt = Descriptors.ExactMolWt(mol)
            hba   = rdMolDescriptors.CalcNumHBA(mol)
            hbd   = rdMolDescriptors.CalcNumHBD(mol)
            molwts.append(molwts)
            qeds.append(qed)
            logps.append(logp)
            tpsas.append(tpsa)
            hbas.append(hba)
            hbds.append(hbd)
    return molwts,qeds,tpsas,logps,hbas,hbds



def Define_box_grid_figure(datadict,figtext={},grids=(2,1),size=(12,12),filename='plot.png',wspace=0.3,hspace=0.2):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import numpy as np

    def set_ax_frame(ax):
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        return
    print (datadict)
    clist='abcdefghijklmnopqrstuvwxyz'
    colors=['red','green','orange','blue','pink','purple','yellow']
    plt.rc('font',size=12)
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    figure=plt.figure(figsize=size)
    xgrids=grids[0]
    ygrids=grids[1]

    gs=gridspec.GridSpec(xgrids,ygrids)
    ndatas=len(datadict.keys())
    for pid,key in enumerate(datadict.keys()):
        xid=pid//ygrids
        yid=pid%ygrids
        ax=plt.subplot(gs[xid,yid])
        subdatadict=datadict[key]
        if 'xlabel' in datadict[key].keys():
            xlabel=datadict[key]['xlabel']
        else:
            xlabel=''
        if 'ylabel' in datadict[key].keys():
            ylabel=datadict[key]['ylabel']
        else:
            ylabel=''
        style=datadict[key]['style']

        if 'xlim' in datadict[key].keys():
            xlim=datadict[key]['xlim']
        else:
            xlim=None
        if 'ylim' in datadict[key].keys():
            ylim=datadict[key]['ylim']
        else:
            ylim=None
        if 'xticks' in datadict[key].keys():
            xticks=np.arrange(*datadict[key]['xticks'])
        else:
            xticks=None
        if 'yticks' in datadict[key].keys():
            yticks=np.arrange(*datadict[key]['yticks'])
        else:
            yticks=None 
        if 'ylabel' in datadict[key].keys():
            ylabel=datadict[key]['ylabel']
        else:
            ylabel=''
        if 'xlabel' in datadict[key].keys():
            xlabel=datadict[key]['xlabel']
        else:
            xlabel=''
        if 'legend' in datadict[key].keys():
            iflegend=datadict[key]['legend']
        else:
            iflegend=False
        if 'xticks_rotation' in datadict[key].keys():
            xticks_rotation=datadict[key]['xticks_rotation']
        else:
            xticks_rotation=0
        if 'yticks_rotation' in datadict[key].keys():
            yticks_rotation=datadict[key]['yticks_rotation']
        else:
            yticks_rotation=0
        if "markersize" in datadict[key].keys():
            markersize=datadict[key]["markersize"]
        else:
            markersize=8
        char=clist[pid]
        for did,subkey in enumerate(subdatadict['data'].keys()):
            if 'pointline' in style:
                plt.plot(subdatadict['data'][subkey][0],subdatadict['data'][subkey][1],label=subkey,color=colors[did],marker='o',linewidth=2,markersize=markersize)
            if 'scatter' in style:
                plt.plot(subdatadict['data'][subkey][0],subdatadict['data'][subkey][1],label=subkey,color=colors[did],marker='o',markersize=markersize)
            if 'regplot' in style:
                sns.regplot(ax=ax,x=subdatadict['data'][subkey][0],y=subdatadict['data'][subkey][1],label=subkey,color=colors[did],scatter=True)
            if 'distplot' in style:
                sns.distplot(subdatadict['data'][subkey],ax=ax,bins=50,label=subkey,hist_kws={},color=colors[did],kde=True,kde_kws={"shade":True})
            if 'bar' in style:
                plt.bar(subdatadict['data'][subkey][0],height=subdatadict['data'][subkey][1],align="edge")
            if iflegend:
                leg=plt.legend(fancybox=True,framealpha=0,fontsize=12,markerscale=0.5)
        if xlim:
            plt.xlim(*xlim)
        if ylim:
            plt.ylim(*ylim)

        if xticks:
            plt.xticks(xticks,rotation=xticks_rotation)
        else:
            plt.xticks(rotation=xticks_rotation,horizontalalignment='right',verticalalignment='top')
        if yticks:
            plt.yticks(yticks,rotation=yticks_rotation)
        else:
            plt.yticks(rotation=yticks_rotation)

        plt.tick_params(length=5,top=True,bottom=True,left=True,right=True)
        plt.xlabel(xlabel,fontsize=16)
        plt.ylabel(ylabel,fontsize=16)
        if 'text' in datadict[key].keys():
            for textkey in datadict[key]['text'].keys():
                plt.text(datadict[key]['text'][textkey][0],datadict[key]['text'][textkey][1],s=textkey,transform=ax.transAxes, fontsize=12)
        #plt.text(0.1,0.85,s=f'({clist[pid]})',transform=ax.transAxes,fontsize=12)
        set_ax_frame(ax)
    for key in figtext.keys():
        pos=figtext[key]
        figure.text(pos[0],pos[1],key,fontsize=16,rotation=pos[2])
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    #plt.savefig(filename)
    #plt.show()
    figure.savefig(filename,format='svg',dpi=300)
        
def MolToMolgraph(mol,ifpadding=False,mode='mol'):
    natoms=mol.GetNumAtoms()
    atom_fs=np.zeros((natoms,GP.syssetting.n_node_features))

    for i,atom in enumerate(mol.GetAtoms()):
        atom_fs[i]=np.array(atom_features(atom))

    atom_adjs=np.zeros((GP.syssetting.n_edge_features,natoms,natoms))

    for bond in mol.GetBonds():
        a1=bond.GetBeginAtom().GetIdx()
        a2=bond.GetEndAtom().GetIdx()
        bt=bond.GetBondType() 
        ch=GP.syssetting.bond_types.index(bt)
        atom_adjs[ch,a1,a2]=1
        atom_adjs[ch,a2,a1]=1
    if ifpadding:
        if mode=="mol":
            atom_fs=np.pad(atom_fs,((0,GP.syssetting.max_atoms-natoms),(0,0)),'constant',constant_values=0)
            atom_adjs=np.pad(
                            atom_adjs,((0,0),(0,GP.syssetting.max_atoms-natoms),(0,GP.syssetting.max_atoms-natoms)),
                            'constant',
                            constant_values=0 
                        )
        else:
            atom_fs=np.pad(atom_fs,((0,GP.syssetting.max_ring_size-natoms),(0,0)),'constant',constant_values=0)
            atom_adjs=np.pad(
                            atom_adjs,((0,0),(0,GP.syssetting.max_ring_size-natoms),(0,GP.syssetting.max_ring_size-natoms)),
                            'constant',
                            constant_values=0 
                        )

    return atom_fs,atom_adjs


    

