from rdkit.Chem.Draw import rdMolDraw2D
#from IPython.display import Image
import copy
import numpy as np
from rdkit import Chem 
from rdkit.Chem import AllChem,rdmolfiles
import copy
import networkx as nx 
from ..comparm import *

def onek_encoding_unk(value, choices):
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices))
    index = choices.index(value)
    encoding[index] = 1
    return encoding

def atom_features(atom: Chem.rdchem.Atom, functional_groups=None) :
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum(), GP.syssetting.possible_atom_types) + \
           onek_encoding_unk(atom.GetFormalCharge(), GP.syssetting.formal_charge_types)
    if GP.syssetting.use_chiral_tag:
        features+=onek_encoding_unk(int(atom.GetChiralTag()), GP.syssetting.chiral_tag_types)
    if GP.syssetting.use_Hs:
        features+=onek_encoding_unk(atom.GetImplicitValence(), GP.syssetting.implicit_hs_types)
    return features

def bond_features(bond: Chem.rdchem.Bond):
    """
    Builds a feature vector for a bond.
    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    bt = bond.GetBondType()
    fbond = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            ]
    return fbond

def frag_features(frag):
    ringnum= len(Chem.GetSymmSSSR(frag))
    ringatoms=[]
    for ring in Chem.GetSymmSSSR(frag):
        for i in ring:
            if i not in ringatoms:
                ringatoms.append(i)
    atomicnum=[atom.GetAtomicNum() for atom in frag.GetAtoms()]
    CNOFPSClBrI=[atomicnum.count(i) for i in [6,7,8,9,15,16,17,35,53]]
    natoms=len(frag.GetAtoms())
    branches=natoms-len(ringatoms)
    num_aromatic_rings=Chem.rdMolDescriptors.CalcNumAromaticRings(frag)
    f_frag=[ringnum]+CNOFPSClBrI+[branches]+[num_aromatic_rings]
    return f_frag