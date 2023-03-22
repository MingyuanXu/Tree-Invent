import copy
import numpy as np
from rdkit import Chem 
from rdkit.Chem import AllChem,rdmolfiles
import copy
import networkx as nx 

def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output

def dfs_seq(G,start_id):
    #dictionary=dict(nx.dfs_successors(G,start_id))
    dictionary={i:[n for n in G.neighbors(i)] for i in range(G.number_of_nodes())}
    #print (dictionary)
    start=[start_id]
    output=[]
    while len(start)>0:
        v=start.pop()
        output.append(v)
        for w in dictionary[v]:
            if w not in output and w not in start:
            #if w not in output:
                start.append(w)
                #output.append(w)
    return output

def rdkit_bfs_seq_mol(molobj,start_id=0):
    natoms=len(molobj.GetAtoms())
    bonds=[]
    for i in range(natoms):
        for j in range(natoms):
            bond=molobj.GetBondBetweenAtoms(i,j)
            if bond:
                bonds.append((i,j))
    G=nx.Graph()
    G.add_edges_from(bonds)
    seq=bfs_seq(G,start_id=start_id)
    print (seq)
    #print (seq)
    reseq=np.argsort(seq)
    molobj=Chem.rdmolops.RenumberAtoms(molobj,seq)
    return molobj

def rdkit_dfs_seq_mol(molobj,start_id=0):
    natoms=len(molobj.GetAtoms())
    bonds=[]
    for i in range(natoms):
        for j in range(natoms):
            bond=molobj.GetBondBetweenAtoms(i,j)
            if bond:
                bonds.append((i,j))
    G=nx.Graph()
    G.add_edges_from(bonds)
    seq=dfs_seq(G,start_id=start_id)
    #print (seq)
    reseq=np.argsort(seq)
    molobj=Chem.rdmolops.RenumberAtoms(molobj,seq)
    return molobj
