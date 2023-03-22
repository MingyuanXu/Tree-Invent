from rdkit import Chem
from rdkit.Chem import rdmolfiles
import torch 
from .util_trans import * 


def adjs2ctable(adjs):
    xline=torch.eye(adjs.shape[2]).to(adjs)
    xline=torch.tile(torch.tile(xline.unsqueeze(0),(adjs.shape[1],1,1)).unsqueeze(0),(adjs.shape[0],1,1,1))
    adjs_=adjs-xline
    ctable=torch.sum(adjs_,dim=1,dtype=torch.double)
    ctable[:,0,0]=1
    tmp=ctable[:,0,2]
    tmp=torch.where(tmp==0.0,0.5,tmp)
    ctable[:,0,2]=tmp 
    ctable[:,2,0]=tmp 
    tmp=ctable[:,1,2]
    tmp=torch.where(tmp==0.0,0.5,tmp)
    ctable[:,1,2]=tmp
    ctable[:,2,1]=tmp    
    return ctable

def Adjs2zmat(adjs,mask=None): 
    ctable=adjs2ctable(adjs[:,:3])
    max_size=ctable.shape[1]
    Bctable=ctable
    bt,it,jt=torch.where(Bctable>0)
    rbij=Bctable[bt,it,jt]
    id=torch.where((it>jt)|(it<3))
    bt=bt[id];it=it[id];jt=jt[id];rbij=rbij[id]
    zb=torch.stack((bt,it,jt),dim=1)
    newindex=bt*max_size+jt
    Actable=torch.index_select(ctable.view(-1,max_size),dim=0,index=newindex)
    indext,kt=torch.where(Actable>0)
    rbjk=Actable[indext,kt]
    bt=torch.index_select(bt,dim=0,index=indext)
    it=torch.index_select(it,dim=0,index=indext)
    jt=torch.index_select(jt,dim=0,index=indext)
    rbij=torch.index_select(rbij,dim=0,index=indext)
    ids=torch.where((it>kt)|(it<3))
    bt=bt[ids];it=it[ids];jt=jt[ids];kt=kt[ids];rbij=rbij[ids];rbjk=rbjk[ids]
    za=torch.stack((bt,it,jt,kt),dim=1)
    newindex=bt*max_size+kt
    Dctable=torch.index_select(ctable.view(-1,max_size),dim=0,index=newindex)
    indext,lt=torch.where(Dctable>0)
    rbkl=Dctable[indext,lt]
    bt=torch.index_select(bt,dim=0,index=indext)
    it=torch.index_select(it,dim=0,index=indext)
    jt=torch.index_select(jt,dim=0,index=indext)
    kt=torch.index_select(kt,dim=0,index=indext)
    rbij=torch.index_select(rbij,dim=0,index=indext)
    rbjk=torch.index_select(rbjk,dim=0,index=indext)
    ids=torch.where((it>lt)|(it<3))
    bt=bt[ids];it=it[ids];jt=jt[ids];kt=kt[ids];lt=lt[ids];rbij=rbij[ids];rbjk=rbjk[ids];rbkl=rbkl[ids]
    ids_all=torch.where(((it!=jt)&(it!=kt)&(it!=lt)&(jt!=kt)&(kt!=lt)&(jt!=lt)&(it>4)&(rbij+rbjk+rbkl==3.0)))[0]
    ids_3=torch.where((it==3)&(rbij+rbjk+rbkl==3.0)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0]
    if ids_3.shape[0]==0:
        ids_3=torch.where((it==3)&(rbij+rbjk+rbkl==2.5)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0]
    ids_4=torch.where((it==4)&(rbij+rbjk+rbkl==3.0)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0]
    if ids_4.shape[0]==0:
        ids_4=torch.where((it==4)&(rbij+rbjk+rbkl==2.5)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0] 
    ids_2=torch.where((it==2)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=kt))&(rbij+rbjk+rbkl==3))[0]
    if ids_2.shape[0]==0:
        ids_2=torch.where((it==2)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=kt))&(rbij+rbjk+rbkl==2.5))[0]
    ids_1=torch.where((it<=1))[0]
    ids,_=torch.sort(torch.cat([ids_1,ids_2,ids_3,ids_4,ids_all]))
    bt=bt[ids];it=it[ids];jt=jt[ids];kt=kt[ids];lt=lt[ids];rbij=rbij[ids];rbjk=rbjk[ids];rbkl=rbkl[ids]
    Zmat=torch.stack((bt,it,jt,kt,lt,rbij+rbjk+rbkl),dim=1)
    uni=bt*max_size+it
    uni,ids=np.unique(uni.detach().cpu().numpy(),return_index=True)
    bt=bt[ids];it=it[ids];jt=jt[ids];kt=kt[ids];lt=lt[ids];rbij=rbij[ids];rbjk=rbjk[ids];rbkl=rbkl[ids]
    uni,counts=torch.unique(bt,return_counts=True)
    Zmat=torch.stack((bt,it,jt,kt,lt,rbij+rbjk+rbkl),dim=1)
    pt=0
    Zlist=[]
    for cid,count in enumerate(counts):
        mat=torch.zeros(max_size,6)
        mat[:,0]=cid 
        mat[:count]=Zmat.narrow(0,pt,count) 
        Zlist.append(mat)    
        pt+=count
    Zmat=torch.stack(Zlist,dim=0).to(torch.int32)
    return Zmat

def Adj2selectic(adjs,id,batch_size,repeat_num):
    ctable=adjs2ctable(adjs[:,:3])
    max_size=ctable.shape[1]
    steps_need_ic=id[0]*repeat_num +id[1]
    it=id[2].clone()
    Actable=torch.index_select(ctable.reshape(-1,max_size),dim=0,index=steps_need_ic*max_size+it)
    indext,jt=torch.where(Actable>0)
    steps_need_ic=steps_need_ic[indext]
    it=it[indext]
    rbij=ctable[steps_need_ic,it,jt]
    ids=torch.where((it>jt)|(it<3))
    steps_need_ic=steps_need_ic[ids]
    rbij=rbij[ids]
    it=it[ids]
    jt=jt[ids]
    Actable=torch.index_select(ctable.reshape(-1,max_size),dim=0,index=steps_need_ic*max_size+jt)
    indext,kt=torch.where(Actable>0)
    steps_need_ic=steps_need_ic[indext]
    it=it[indext]
    jt=jt[indext]
    rbij=rbij[indext]
    rbjk=ctable[steps_need_ic,jt,kt]
    ids=torch.where((it>kt)|(it<3))
    steps_need_ic=steps_need_ic[ids]
    rbij=rbij[ids]
    rbjk=rbjk[ids]
    it=it[ids]
    jt=jt[ids]
    kt=kt[ids]
    Actable=torch.index_select(ctable.reshape(-1,max_size),dim=0,index=steps_need_ic*max_size+kt)
    indext,lt=torch.where(Actable>0)
    steps_need_ic=steps_need_ic[indext]
    it=it[indext]
    jt=jt[indext]
    kt=kt[indext]
    rbij=rbij[indext]
    rbjk=rbjk[indext]
    rbkl=ctable[steps_need_ic,kt,lt]
    ids=torch.where((it>lt)|(it<3))
    steps_need_ic=steps_need_ic[ids]
    rbij=rbij[ids]
    rbjk=rbjk[ids]
    rbkl=rbkl[ids]
    it=it[ids]
    jt=jt[ids]
    kt=kt[ids]
    lt=lt[ids]
    ids_all=torch.where(((it!=jt)&(it!=kt)&(it!=lt)&(jt!=kt)&(kt!=lt)&(jt!=lt)&(it>4)&(rbij+rbjk+rbkl==3.0)))[0]
    ids_3=torch.where(((it>jt)&(it>kt)&(it>lt)&(jt!=kt)&(kt!=lt)&(jt!=lt)&(it==3)&(rbij+rbjk+rbkl>=2.5)))[0] 
    ids_4=torch.where(((it!=jt)&(it!=kt)&(it!=lt)&(jt!=kt)&(kt!=lt)&(jt!=lt)&(it==4)&(rbij+rbjk+rbkl>=2.5)))[0] 
    ids_2=torch.where((it==2)&(it>jt)&(it>kt)&(it>lt)&(rbij+rbjk+rbkl==3)&(((jt==0)&(lt==0)&(kt!=0))|((kt==0)&(lt==0)&(jt!=0))))[0]
    ids_1=torch.where((it<=1)&(jt==0)&(kt==0)&(lt==0))[0]
    #ids,_=torch.sort(torch.cat([ids_1,ids_2,ids_all]+ids_4list+ids_3list))
    ids,_=torch.sort(torch.cat([ids_1,ids_2,ids_3,ids_4,ids_all]))
    steps_need_ic=steps_need_ic[ids]
    it=it[ids]
    jt=jt[ids]
    kt=kt[ids]
    lt=lt[ids]
    rbij=rbij[ids]
    rbjk=rbjk[ids]
    rbkl=rbkl[ids]

    unique,inverse=torch.unique(steps_need_ic*GP.mol_setting.max_atom_num+it,sorted=True,return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    steps_need_ic=steps_need_ic[perm]
    it=it[perm]
    jt=jt[perm]
    kt=kt[perm]
    lt=lt[perm]
    rbij=rbij[perm]
    rbjk=rbjk[perm]
    rbkl=rbkl[perm]
    bid=steps_need_ic//repeat_num
    uni_bid,inv_bid,counts=torch.unique(bid,sorted=True,return_inverse=True,return_counts=True)
    #print (uni_bid,inv_bid,counts)
    Zmat=torch.stack((steps_need_ic,it,jt,kt,lt,rbij+rbjk+rbkl),dim=1).int()
    pt=0
    ptlist=[]
    for cid,c in enumerate(counts):
        ptlist.append((pt,int(c)))
        pt+=int(c)
    #print (Zmat.shape,Zmat[:100])
    return Zmat,ptlist 

def xyz2selectic(x,zmat,mask=None,eps=1e-7):
    batch_num=x.shape[0]
    max_atom_num=x.shape[1]
    zmat=zmat.view(-1,6)
    it=zmat[:,0]*max_atom_num+zmat[:,1]
    jt=zmat[:,0]*max_atom_num+zmat[:,2]
    kt=zmat[:,0]*max_atom_num+zmat[:,3]
    lt=zmat[:,0]*max_atom_num+zmat[:,4]
    x=x.view(-1,3)
    x0=torch.index_select(x,dim=0,index=it).view(-1,3)
    x1=torch.index_select(x,dim=0,index=jt).view(-1,3)
    x2=torch.index_select(x,dim=0,index=kt).view(-1,3)
    x3=torch.index_select(x,dim=0,index=lt).view(-1,3)
    dist,J_dist=dist_deriv(x0,x1)
    angle,J_angle=angle_deriv(x0,x1,x2)
    dihedral,J_dihedral=torsion_deriv(x0,x1,x2,x3)
    dist=dist.view(-1,1)
    J_dist=dist.view(-1,1)
    angle=angle.view(-1,1)
    J_angle=angle.view(-1,1)
    dihedral=dihedral.view(-1,1)
    return dist,angle,dihedral,J_dist,J_angle,J_dihedral

def xyz2ic(x,zmat,mask=None,eps=1e-7):
    batch_num=x.shape[0]
    max_atom_num=x.shape[1]
    zmat=zmat.view(-1,6)
    it=zmat[:,0]*max_atom_num+zmat[:,1]
    jt=zmat[:,0]*max_atom_num+zmat[:,2]
    kt=zmat[:,0]*max_atom_num+zmat[:,3]
    lt=zmat[:,0]*max_atom_num+zmat[:,4]
    x=x.view(-1,3)
    x0=torch.index_select(x,dim=0,index=it).view(-1,max_atom_num,3)
    x1=torch.index_select(x,dim=0,index=jt).view(-1,max_atom_num,3)
    x2=torch.index_select(x,dim=0,index=kt).view(-1,max_atom_num,3)
    x3=torch.index_select(x,dim=0,index=lt).view(-1,max_atom_num,3)
    dist,J_dist=dist_deriv(x0,x1)
    angle,J_angle=angle_deriv(x0,x1,x2)
    dihedral,J_dihedral=torsion_deriv(x0,x1,x2,x3)
    return dist,angle,dihedral,J_dist,J_angle,J_dihedral

def ic2xyz(dist,angle,dihedral,zmat,cond_xyz=None,mask=None,eps=1e-7):
    max_atom_num=zmat.shape[1]
    x=torch.zeros((zmat.shape[0],zmat.shape[1],3)).cuda()
    x=x.view(-1,3)
    for i in torch.arange(1,max_atom_num,dtype=torch.int32).cuda():
        zindex=torch.index_select(zmat,dim=1,index=i).view(-1,6).long()
        disti=torch.index_select(dist,dim=1,index=i).view(-1,1)
        anglei=torch.index_select(angle,dim=1,index=i).view(-1,1)
        dihedrali=torch.index_select(dihedral,dim=1,index=i).view(-1,1)
        it=zindex[:,0]*max_atom_num+zindex[:,1]
        jt=zindex[:,0]*max_atom_num+zindex[:,2]
        kt=zindex[:,0]*max_atom_num+zindex[:,3]
        lt=zindex[:,0]*max_atom_num+zindex[:,4]
        x0=torch.index_select(x,dim=0,index=it)
        x1=torch.index_select(x,dim=0,index=jt)
        x2=torch.index_select(x,dim=0,index=kt)
        x3=torch.index_select(x,dim=0,index=lt)
        index=zindex[:,0]*max_atom_num+i 
        index=index.to(torch.int64)
        if i==1:
            x[index,2]=disti.view(-1)
        elif i==2:
            xyz,J_2=ic2xy0_deriv(x1,x2,disti,anglei)
            x[index]=xyz
        else:
            xyz,J_i=ic2xyz_deriv(x1,x2,x3,disti,anglei,dihedrali)
            x[index]=xyz
    x=x.view(-1,max_atom_num,3)
    return x

import networkx as nx 
def Adjs_Distance_nx(adjs):
    adjs=adjs.numpy()
    max_graph_size=adjs.shape[2]
    xline=np.eye(adjs.shape[2])
    xline=np.tile(np.tile(xline.reshape(-1,adjs.shape[2],adjs.shape[2]),(adjs.shape[1],1,1)).reshape(-1,adjs.shape[1],adjs.shape[2],adjs.shape[2]),(adjs.shape[0],1,1,1))
    adjs=adjs-xline
    adjs_=np.sum(adjs[:,:3],axis=1)
    for id in range(adjs_.shape[0]):
        G=nx.from_numpy_matrix(adjs_[id])
        connection_length=dict(nx.all_pairs_dijkstra_path_length(G))
        Dadjs=np.zeros((max_graph_size,max_graph_size))
        for key1 in connection_length.keys():
            for key2 in connection_length[key1].keys():
                Dadjs[key1][key2]=connection_length[key1][key2]
    return Dadjs

def Rmatrix(coords):
    R=torch.cdist(coords,coords)
    return R

def D3_embedding(Atoms,Coords,mask=None,Rs=4.5,Rc=6.0,max_neighbor=6):
    R=torch.cdist(Coords,Coords,compute_mode='donot_use_mm_for_euclid_dist')
    xline=torch.eye(R.shape[1]).to(R)
    xline=torch.tile(xline.unsqueeze(0),(R.shape[0],1,1))*1e9
    batch_num=Coords.shape[0]
    max_atoms=Coords.shape[1]
    if mask is not None:
        R=R*mask+xline 
    R=torch.where(R==0,torch.Tensor([1e8]).to(R),R)

    R_inverse,R_rank=torch.sort(R,dim=2)
    R_inverse=R_inverse[:,:,:max_neighbor]
    R_rank=R_rank[:,:,:max_neighbor]
    id0=torch.arange(batch_num).reshape(-1,1).to(Coords)
    
    environ_index=R_rank.reshape(batch_num,-1)+id0*max_atoms
    environ_index=environ_index.int()
    center_index=torch.tile(torch.arange(max_atoms).unsqueeze(0).T.unsqueeze(0).to(Coords),(batch_num,1,max_neighbor)).reshape(batch_num,-1)+id0*max_atoms
    center_index=center_index.int()

    environ_coords=torch.index_select(Coords.reshape(-1,3),dim=0,index=environ_index.reshape(-1)).reshape(-1,max_atoms,max_neighbor,3)[:,:,:max_neighbor,:]
    center_coords =torch.index_select(Coords.reshape(-1,3),dim=0,index=center_index.reshape(-1)).reshape(-1,max_atoms,max_neighbor,3)[:,:,:max_neighbor,:]
    environ_atoms =torch.index_select(Atoms.reshape(-1,1),dim=0,index=environ_index.reshape(-1)).reshape(-1,max_atoms,max_neighbor,1)[:,:,:max_neighbor,:]
    center_atoms  =torch.index_select(Atoms.reshape(-1,1),dim=0,index=center_index.reshape(-1)).reshape(-1,max_atoms,max_neighbor,1)[:,:,:max_neighbor,:]
    
    p1=center_coords[:,:,0,:]
    p2=environ_coords[:,:,0,:]

    p3=environ_coords[:,:,1,:]
    direction=torch.stack(tripod(p1,p2,p3),dim=-1)
    environ_vec=environ_coords-center_coords
    GIE_RC=torch.einsum('bnij,bnjk->bnik',environ_vec,direction)# (b,n,n,3)*(b,n,3,3)
    S_factor=torch.where(R_inverse<=Rs,1/R_inverse,torch.Tensor([0.0]).to(Coords))+torch.where((R_inverse>Rs)&(R_inverse<Rc),1/R_inverse*(0.5*torch.cos((R_inverse-Rs)/(Rc-Rs)*math.pi)+0.5),torch.Tensor([0.0]).to(Coords))
    S_factor=S_factor.unsqueeze(-1)
    GIE_RC=GIE_RC*S_factor/R_inverse.unsqueeze(-1)
    D3_embed=torch.cat((environ_atoms/max(GP.mol_setting.ElementTypeList),GIE_RC),dim=-1).reshape((batch_num,max_atoms,4*max_neighbor))
    if mask is not None:
        clean_mask=mask[:,:,:1]#.byte()
        D3_embed=D3_embed*clean_mask        
    return D3_embed

def IC_embedding(Atoms,Coords,Zmat,mask=None,Rs=4.5,Rc=6.0,max_neighbor=6):
    #1. for graph generation, Zmat is different with geo generation
    #2. for geo generation, Zmat is only for real atoms 
    print (Zmat.shape,Atoms.shape,Coords.shape)
    R=torch.cdist(Coords,Coords,compute_mode='donot_use_mm_for_euclid_dist')
    
    xline=torch.eye(R.shape[1]).to(R)
    xline=torch.tile(xline.unsqueeze(0),(R.shape[0],1,1))*1e9
    batch_num=Coords.shape[0]
    max_atoms=Coords.shape[1]

    if mask is not None:
        R=R*mask+xline
    R                = torch.where(R==0,torch.Tensor([1e8]).to(R),R)
    R_inverse,R_rank = torch.sort(R,dim=2)
    R_inverse        = R_inverse[:,:,:max_neighbor]
    R_rank           = R_rank[:,:,:max_neighbor]
    id0              = torch.arange(batch_num).reshape(-1,1).to(Coords)
    
    environ_index = R_rank.reshape(batch_num,-1)+id0*max_atoms
    environ_index = environ_index.int()
    center_index  = torch.tile(torch.arange(max_atoms).unsqueeze(0).T.unsqueeze(0).to(Coords),(batch_num,1,max_neighbor)).reshape(batch_num,-1)+id0*max_atoms
    center_index  = center_index.int()

    environ_coords = torch.index_select(Coords.reshape(-1,3),dim=0,index=environ_index.reshape(-1)).reshape(-1,max_atoms,max_neighbor,3)[:,:,:max_neighbor,:]
    center_coords  = torch.index_select(Coords.reshape(-1,3),dim=0,index=center_index.reshape(-1)).reshape(-1,max_atoms,max_neighbor,3)[:,:,:max_neighbor,:]
    environ_atoms  = torch.index_select(Atoms.reshape(-1,1),dim=0,index=environ_index.reshape(-1)).reshape(-1,max_atoms,max_neighbor,1)[:,:,:max_neighbor,:]
    center_atoms   = torch.index_select(Atoms.reshape(-1,1),dim=0,index=center_index.reshape(-1)).reshape(-1,max_atoms,max_neighbor,1)[:,:,:max_neighbor,:]
    
    p1=center_coords[:,:,0,:]
    p2=environ_coords[:,:,0,:]
    p3=environ_coords[:,:,1,:]

    direction=torch.stack(tripod(p1,p2,p3),dim=-1)
    environ_vec=environ_coords-center_coords
    GIE_RC=torch.einsum('bnij,bnjk->bnik',environ_vec,direction)# (b,n,n,3)*(b,n,3,3)

    S_factor=torch.where(R_inverse<=Rs,1/R_inverse,torch.Tensor([0.0]).to(Coords))+torch.where((R_inverse>Rs)&(R_inverse<Rc),1/R_inverse*(0.5*torch.cos((R_inverse-Rs)/(Rc-Rs)*math.pi)+0.5),torch.Tensor([0.0]).to(Coords))
    S_factor=S_factor.unsqueeze(-1)
    GIE_RC=GIE_RC*S_factor/R_inverse.unsqueeze(-1)
    D3_embed=torch.cat((environ_atoms/max(GP.mol_setting.ElementTypeList),GIE_RC),dim=-1).reshape((batch_num,max_atoms,4*max_neighbor))
    if mask is not None:
        clean_mask=mask[:,:,:1]#.byte()
        D3_embed=D3_embed*clean_mask        
    return D3_embed