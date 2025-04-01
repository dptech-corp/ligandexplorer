import numpy as np
import os
import pickle
import pkg_resources
from Bio.PDB import PDBParser, Selection
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import warnings

warnings.filterwarnings("ignore")

def read_file(input_pdb):
    parser = PDBParser(QUIET= True)
    structure = parser.get_structure('str', input_pdb)

    for model in structure:
        for chain in model:
            for residue in chain:
                atoms_to_remove = [atom.id for atom in residue if atom.element == 'H' ]
            for atom_id in atoms_to_remove:
                residue.detach_child(atom_id)
    return structure

def get_connected_atoms(atom, atom_list, bond_distance= 1.85):
    atom_coords = atom.coord
    other_atoms = [ a for a in atom_list if a != atom ]
    other_coords = np.array([ a.coord for a in other_atoms])
    deltas = other_coords - atom_coords
    distances = np.linalg.norm(deltas, axis= 1)

    connected_atoms = [other_atoms[i] for i in np.where(distances < bond_distance)[0]]
    return connected_atoms

def count_special_atoms(atoms, element, connections_num):
    atom_elements = np.array([atom.element for atom in atoms])
    count = np.sum(atom_elements == element)
    return count

def get_graph(atoms):
    graphs = nx.Graph()
    for atom in atoms:
        node_id = atom.get_serial_number()
        graphs.add_node(node_id, element= atom.element)
    
    for i, atom1 in enumerate(atoms):
        for j, atom2 in enumerate(atoms):
            if i >= j:
                continue
            if atom1 - atom2 < 1.8:
                node1_id = atom1.get_serial_number()
                node2_id = atom2.get_serial_number()
                graphs.add_edge(node1_id, node2_id)
    for node in graphs.nodes():
        connectivity = graphs.degree(node)
        graphs.nodes[node]['connect'] = connectivity
    return graphs

def build_substructure_graph(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def count_substructures(main_graph, pattern_graph, connect=False):
    def node_match(n1, n2):   
        if n2.get('element') == 'X':
            element_match = True
        else:
            element_match = n1.get('element') == n2.get('element')
        if not element_match:  
            return False  
        if not connect:  
            return True  
        connect_match = n1.get('connect') == n2.get('connect')              
        return connect_match  
    
    def edge_match(e1, e2):  
        return True  

    matcher = nx.algorithms.isomorphism.GraphMatcher(  
        main_graph,   
        pattern_graph,  
        node_match=node_match,  
        edge_match=edge_match  
    )  
 
    found_matches = []   
    for mapping in matcher.subgraph_isomorphisms_iter():  
        matched_nodes = frozenset(mapping.keys())  
        is_unique = True  
        for previous_match in found_matches:  
            if not matched_nodes.isdisjoint(previous_match):  
                is_unique = False  
                break  
        if is_unique:  
            found_matches.append(matched_nodes)    
    return len(found_matches)

def get_features(bio_str):
    element_to_atomic_number = {
    'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10,
    'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18,
    'K': 19, 'CA': 20, 'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 'MN': 25, 'FE': 26, 'CO': 27,
    'NI': 28, 'CU': 29, 'ZN': 30, 'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36,
    'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 'NB': 41, 'MO': 42, 'TC': 43, 'RU': 44, 'RH': 45,
    'PD': 46, 'AG': 47, 'CD': 48, 'IN': 49, 'SN': 50, 'SB': 51, 'TE': 52, 'I': 53, 'XE': 54,
    'CS': 55, 'BA': 56, 'LA': 57, 'CE': 58, 'PR': 59, 'ND': 60, 'PM': 61, 'SM': 62, 'EU': 63,
    'GD': 64, 'TB': 65, 'DY': 66, 'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71, 'HF': 72,
    'TA': 73, 'W': 74, 'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78, 'AU': 79, 'HG': 80, 'TL': 81,
    'PB': 82, 'BI': 83, 'PO': 84, 'AT': 85, 'RN': 86, 'FR': 87, 'RA': 88, 'AC': 89, 'TH': 90,
    'PA': 91, 'U': 92, 'NP': 93, 'PU': 94, 'AM': 95, 'CM': 96, 'BK': 97, 'CF': 98, 'ES': 99,
    'FM': 100, 'MD': 101, 'NO': 102, 'LR': 103, 'RF': 104, 'DB': 105, 'SG': 106, 'BH': 107,
    'HS': 108, 'MT': 109, 'DS': 110, 'RG': 111, 'CN': 112, 'NH': 113, 'FL': 114, 'MC': 115,
    'LV': 116, 'TS': 117, 'OG': 118
    }
    features = []
    count_c = 0
    count_o = 0
    count_n = 0
    count_p = 0
    count_s = 0
    count_halogen = 0
    count_alkali = 0
    # count_meta = 0
    count_less_36 = 0
    count_class_2 = 0
    count_class_3 = 0
    count_class_4 = 0
    count_class_5 = 0
    count_class_6 = 0
    count_class_7 = 0

    atoms = []
    for model in bio_str:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element != 'H':
                        atoms.append(atom)    
    count_all = len(atoms)
    str_graph = get_graph(atoms)
    num_edges = str_graph.number_of_edges()

    for atom in atoms:
        element = atom.element
        atomic_number = element_to_atomic_number.get(element, 1)
        if atomic_number == 6:
            count_c += 1
        elif atomic_number == 7:
            count_n += 1
        elif atomic_number == 8:
            count_o += 1
        elif atomic_number == 15:
            count_p += 1
        elif atomic_number == 16:
            count_s += 1
        elif atomic_number in [9, 17, 35, 53]:
            count_halogen += 1
        elif atomic_number in [3, 11, 19]:
            count_alkali += 1
        if 1 < atomic_number <= 36:
            count_less_36 += 1
        if 3 <= atomic_number <= 10:
            count_class_2 += 1
        elif 11 <= atomic_number <= 18:
            count_class_3 += 1
        elif 19 <= atomic_number <= 36:
            count_class_4 += 1
        elif 37 <= atomic_number <= 54:
            count_class_5 += 1
        elif 55 <= atomic_number <= 86:
            count_class_6 += 1
        elif 87 <= atomic_number <= 118:
            count_class_7 += 1
    O_1 = count_special_atoms(atoms, 'O', 1)
    O_2 = count_special_atoms(atoms, 'O', 2)
    C_1 = count_special_atoms(atoms, 'C', 1)
    C_2 = count_special_atoms(atoms, 'C', 2)
    C_3 = count_special_atoms(atoms, 'C', 3)
    C_4 = count_special_atoms(atoms, 'C', 4)
    N_1 = count_special_atoms(atoms, 'N', 1)
    N_2 = count_special_atoms(atoms, 'N', 2)
    N_3 = count_special_atoms(atoms, 'N', 3)
    S_1 = count_special_atoms(atoms, 'S', 1)
    S_2 = count_special_atoms(atoms, 'S', 2)
    P_3 = count_special_atoms(atoms, 'P', 3)
    P_5 = count_special_atoms(atoms, 'P' ,4)
    # amino acid
    # gly amino acid
    gly_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 2}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1})]
    gly_edge = [(1,2), (2,3), (3,4)]
    gly_graph = build_substructure_graph(gly_node, gly_edge)
    gly_count = count_substructures(str_graph, gly_graph, connect=True)
    # gly amino acid end
    gly_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 2}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1})]
    gly_end_edge = [(1,2), (2,3), (3,4)]
    gly_end_graph = build_substructure_graph(gly_end_node, gly_end_edge)
    gly_end_count = count_substructures(str_graph, gly_end_graph, connect=True)
    # ala amino acid
    ala_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 1})]
    ala_edge = [(1,2), (2,3), (3,4), (2,5)]
    ala_graph = build_substructure_graph(ala_node, ala_edge)
    ala_count = count_substructures(str_graph, ala_graph, connect=True)
    # ala amino acid end
    ala_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 1})]
    ala_end_edge = [(1,2), (2,3), (3,4), (2,5)]
    ala_end_graph = build_substructure_graph(ala_end_node, ala_end_edge)
    ala_end_count = count_substructures(str_graph, ala_end_graph, connect=True)
    # pro amino acid
    pro_node = [(1, {'element': 'N', 'connect': 3}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 2})]
    pro_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,1)]
    pro_graph = build_substructure_graph(pro_node, pro_edge)
    pro_count = count_substructures(str_graph, pro_graph, connect=True)
    # pro amino acid end
    pro_end_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 2})]
    pro_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,1)]
    pro_end_graph = build_substructure_graph(pro_end_node, pro_end_edge)
    pro_end_count = count_substructures(str_graph, pro_end_graph, connect=True)
    # val amino acid
    val_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 3}),
                (6, {'element': 'C', 'connect': 1}),
                (7, {'element': 'C', 'connect': 1})]
    val_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (5,7)]
    val_graph = build_substructure_graph(val_node, val_edge)
    val_count = count_substructures(str_graph, val_graph, connect=True)
    # val amino acid end
    val_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 3}),
                (6, {'element': 'C', 'connect': 1}),
                (7, {'element': 'C', 'connect': 1})]
    val_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (5,7)]
    val_end_graph = build_substructure_graph(val_end_node, val_end_edge)
    val_end_count = count_substructures(str_graph, val_end_graph, connect=True)
    # leu amino acid
    leu_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'C', 'connect': 1}),
                (8, {'element': 'C', 'connect': 1})]
    leu_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (6,8)]
    leu_graph = build_substructure_graph(leu_node, leu_edge)
    leu_count = count_substructures(str_graph, leu_graph, connect=True)
    # leu amino acid end
    leu_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'C', 'connect': 1}),
                (8, {'element': 'C', 'connect': 1})]
    leu_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (6,8)]
    leu_end_graph = build_substructure_graph(leu_end_node, leu_end_edge)
    leu_end_count = count_substructures(str_graph, leu_end_graph, connect=True)
    # ile amino acid
    ile_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 3}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 1}),
                (8, {'element': 'C', 'connect': 1})]
    ile_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (5,8)]
    ile_graph = build_substructure_graph(ile_node, ile_edge)
    ile_count = count_substructures(str_graph, ile_graph, connect=True)
    # ile amino acid end
    ile_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 3}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 1}),
                (8, {'element': 'C', 'connect': 1})]
    ile_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (5,8)]
    ile_end_graph = build_substructure_graph(ile_end_node, ile_end_edge)
    ile_end_count = count_substructures(str_graph, ile_end_graph, connect=True)
    # ser amino acid
    ser_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'O', 'connect': 1})]
    ser_edge = [(1,2), (2,3), (3,4), (2,5), (5,6)]
    ser_graph = build_substructure_graph(ser_node, ser_edge)
    ser_count = count_substructures(str_graph, ser_graph, connect=True)
    # ser amino acid end
    ser_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'O', 'connect': 1})]
    ser_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6)]
    ser_end_graph = build_substructure_graph(ser_end_node, ser_end_edge)
    ser_end_count = count_substructures(str_graph, ser_end_graph, connect=True)
    # thr amino acid
    thr_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 3}),
                (6, {'element': 'C', 'connect': 1}),
                (7, {'element': 'O', 'connect': 1})]
    thr_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (5,7)]
    thr_graph = build_substructure_graph(thr_node, thr_edge)
    thr_count = count_substructures(str_graph, thr_graph, connect=True)
    # thr amino acid end
    thr_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 3}),
                (6, {'element': 'C', 'connect': 1}),
                (7, {'element': 'O', 'connect': 1})]
    thr_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (5,7)]
    thr_end_graph = build_substructure_graph(thr_end_node, thr_end_edge)
    thr_end_count = count_substructures(str_graph, thr_end_graph, connect=True)
    # gln amino acid
    gln_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 3}),
                (8, {'element': 'O', 'connect': 1}),
                (9, {'element': 'N', 'connect': 1})]
    gln_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (7,9)]
    gln_graph = build_substructure_graph(gln_node, gln_edge)
    gln_count = count_substructures(str_graph, gln_graph, connect=True)
    # gln amino acid end
    gln_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 3}),
                (8, {'element': 'O', 'connect': 1}),
                (9, {'element': 'N', 'connect': 1})]
    gln_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (7,9)]
    gln_end_graph = build_substructure_graph(gln_end_node, gln_end_edge)
    gln_end_count = count_substructures(str_graph, gln_end_graph, connect=True)
    # asn amino acid
    asn_node =  [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'O', 'connect': 1}),
                (8, {'element': 'N', 'connect': 1})]
    asn_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (6,8)]
    asn_graph = build_substructure_graph(asn_node, asn_edge)
    asn_count = count_substructures(str_graph, asn_graph, connect=True)
    # asn amino acid end
    asn_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'O', 'connect': 1}),
                (8, {'element': 'N', 'connect': 1})]
    asn_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (6,8)]
    asn_end_graph = build_substructure_graph(asn_end_node, asn_end_edge)
    asn_end_count = count_substructures(str_graph, asn_end_graph, connect=True)
    # met amino acid
    met_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'S', 'connect': 2}),
                (8, {'element': 'C', 'connect': 1})]
    met_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8)]
    met_graph = build_substructure_graph(met_node, met_edge)
    met_count = count_substructures(str_graph, met_graph, connect=True)
    # met amino acid end
    met_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'S', 'connect': 2}),
                (8, {'element': 'C', 'connect': 1})]
    met_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8)]
    met_end_graph = build_substructure_graph(met_end_node, met_end_edge)
    met_end_count = count_substructures(str_graph, met_end_graph, connect=True)
    # cys amino acid
    cys_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'S', 'connect': 1})]
    cys_edge = [(1,2), (2,3), (3,4), (2,5), (5,6)]
    cys_graph = build_substructure_graph(cys_node, cys_edge)
    cys_count = count_substructures(str_graph, cys_graph, connect=True)
    # cys amino acid end
    cys_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'S', 'connect': 1})]
    cys_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6)]
    cys_end_graph = build_substructure_graph(cys_end_node, cys_end_edge)
    cys_end_count = count_substructures(str_graph, cys_end_graph, connect=True)
    # phe amino acid
    phe_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'C', 'connect': 2}),
                (8, {'element': 'C', 'connect': 2}),
                (9, {'element': 'C', 'connect': 2}),
                (10, {'element': 'C', 'connect': 2}),
                (11, {'element': 'C', 'connect': 2})]
    phe_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (6,11)]
    phe_graph = build_substructure_graph(phe_node, phe_edge)
    phe_count = count_substructures(str_graph, phe_graph, connect=True)
    # phe amino acid end
    phe_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'C', 'connect': 2}),
                (8, {'element': 'C', 'connect': 2}),
                (9, {'element': 'C', 'connect': 2}),
                (10, {'element': 'C', 'connect': 2}),
                (11, {'element': 'C', 'connect': 2})]
    phe_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (6,11)]
    phe_end_graph = build_substructure_graph(phe_end_node, phe_end_edge)
    phe_end_count = count_substructures(str_graph, phe_end_graph, connect=True)
    # tyr amino acid
    tyr_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'C', 'connect': 2}),
                (8, {'element': 'C', 'connect': 2}),
                (9, {'element': 'C', 'connect': 3}),
                (10, {'element': 'C', 'connect': 2}),
                (11, {'element': 'C', 'connect': 2}),
                (12, {'element': 'O', 'connect': 1})]
    tyr_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (6,11), (9,12)]
    tyr_graph = build_substructure_graph(tyr_node, tyr_edge)
    tyr_count = count_substructures(str_graph, tyr_graph, connect=True)
    # tyr amino acid end
    tyr_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'C', 'connect': 2}),
                (8, {'element': 'C', 'connect': 2}),
                (9, {'element': 'C', 'connect': 3}),
                (10, {'element': 'C', 'connect': 2}),
                (11, {'element': 'C', 'connect': 2}),
                (12, {'element': 'O', 'connect': 1})]
    tyr_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (6,11), (9,12)]
    tyr_end_graph = build_substructure_graph(tyr_end_node, tyr_end_edge)
    tyr_end_count = count_substructures(str_graph, tyr_end_graph, connect=True)
    # trp amino acid
    trp_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'C', 'connect': 2}),
                (8, {'element': 'N', 'connect': 2}),
                (9, {'element': 'C', 'connect': 3}),
                (10, {'element': 'C', 'connect': 2}),
                (11, {'element': 'C', 'connect': 2}),
                (12, {'element': 'C', 'connect': 2}),
                (13, {'element': 'C', 'connect': 2}),
                (14, {'element': 'C', 'connect': 3})]
    trp_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (9,14), (6,14)]
    trp_graph = build_substructure_graph(trp_node, trp_edge)
    trp_count = count_substructures(str_graph, trp_graph, connect=True)
    # trp amino acid end
    trp_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'C', 'connect': 2}),
                (8, {'element': 'N', 'connect': 2}),
                (9, {'element': 'C', 'connect': 3}),
                (10, {'element': 'C', 'connect': 2}),
                (11, {'element': 'C', 'connect': 2}),
                (12, {'element': 'C', 'connect': 2}),
                (13, {'element': 'C', 'connect': 2}),
                (14, {'element': 'C', 'connect': 3})]
    trp_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (9,14), (6,14)]
    trp_end_graph = build_substructure_graph(trp_end_node, trp_end_edge)
    trp_end_count = count_substructures(str_graph, trp_end_graph, connect=True)
    # asp amino acid
    asp_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'O', 'connect': 1}),
                (8, {'element': 'O', 'connect': 1})]
    asp_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (6,8)]
    asp_graph = build_substructure_graph(asp_node, asp_edge)
    asp_count = count_substructures(str_graph, asp_graph, connect=True)
    # asp amino acid end
    asp_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'O', 'connect': 1}),
                (8, {'element': 'O', 'connect': 1})]
    asp_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (6,8)]
    asp_end_graph = build_substructure_graph(asp_end_node, asp_end_edge)
    asp_end_count = count_substructures(str_graph, asp_end_graph, connect=True)
    # glu amino acid
    glu_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 3}),
                (8, {'element': 'O', 'connect': 1}),
                (9, {'element': 'O', 'connect': 1})]
    glu_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (7,9)]
    glu_graph = build_substructure_graph(glu_node, glu_edge)
    glu_count = count_substructures(str_graph, glu_graph, connect=True)
    # glu amino acid end
    glu_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 3}),
                (8, {'element': 'O', 'connect': 1}),
                (9, {'element': 'O', 'connect': 1})]
    glu_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (7,9)]
    glu_end_graph = build_substructure_graph(glu_end_node, glu_end_edge)
    glu_end_count = count_substructures(str_graph, glu_end_graph, connect=True)
    # lys amino acid
    lys_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 2}),
                (8, {'element': 'C', 'connect': 2}),
                (9, {'element': 'N', 'connect': 1})]
    lys_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9)]
    lys_graph = build_substructure_graph(lys_node, lys_edge)
    lys_count = count_substructures(str_graph, lys_graph, connect=True)
    # lys amino acid end
    lys_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 2}),
                (8, {'element': 'C', 'connect': 2}),
                (9, {'element': 'N', 'connect': 1})]
    lys_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9)]
    lys_end_graph = build_substructure_graph(lys_end_node, lys_end_edge)
    lys_end_count = count_substructures(str_graph, lys_end_graph, connect=True)
    # arg amino acid
    arg_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 2}),
                (8, {'element': 'N', 'connect': 2}),
                (9, {'element': 'C', 'connect': 3}),
                (10, {'element': 'N', 'connect': 1}),
                (11, {'element': 'N', 'connect': 1})]
    arg_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9), (9,10), (9,11)]
    arg_graph = build_substructure_graph(arg_node, arg_edge)
    arg_count = count_substructures(str_graph, arg_graph, connect=True)
    # arg amino acid end
    arg_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2}),
                (7, {'element': 'C', 'connect': 2}),
                (8, {'element': 'N', 'connect': 2}),
                (9, {'element': 'C', 'connect': 3}),
                (10, {'element': 'N', 'connect': 1}),
                (11, {'element': 'N', 'connect': 1})]
    arg_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9), (9,10), (9,11)]
    arg_end_graph = build_substructure_graph(arg_end_node, arg_end_edge)
    arg_end_count = count_substructures(str_graph, arg_end_graph, connect=True)
    # his amino acid
    his_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'N', 'connect': 2}),
                (8, {'element': 'C', 'connect': 2}),
                (9, {'element': 'N', 'connect': 2}),
                (10, {'element': 'C', 'connect': 2})]
    his_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9), (9,10), (6,10)]
    his_graph = build_substructure_graph(his_node, his_edge)
    his_count = count_substructures(str_graph, his_graph, connect=True)
    # his amino acid
    his_end_node = [(1, {'element': 'N', 'connect': 1}),
                (2, {'element': 'C', 'connect': 3}),
                (3, {'element': 'C', 'connect': 3}),
                (4, {'element': 'O', 'connect': 1}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 3}),
                (7, {'element': 'N', 'connect': 2}),
                (8, {'element': 'C', 'connect': 2}),
                (9, {'element': 'N', 'connect': 2}),
                (10, {'element': 'C', 'connect': 2})]
    his_end_edge = [(1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (7,8), (8,9), (9,10), (6,10)]
    his_end_graph = build_substructure_graph(his_end_node, his_end_edge)
    his_end_count = count_substructures(str_graph, his_end_graph, connect=True)
    # phosphate
    phosphoric_node = [(1, {'element': 'P', 'connect': 5}),
                       (2, {'element': 'O', 'connect': 1}),
                       (3, {'element': 'O', 'connect': 1}),
                       (4, {'element': 'O', 'connect': 1}),
                       (5, {'element': 'O', 'connect': 1})]
    phosphoric_edge = [(1,2), (1,3), (1,4), (1,5)]
    phosphoric_graph = build_substructure_graph(phosphoric_node, phosphoric_edge)
    phosphoric_count = count_substructures(str_graph, phosphoric_graph, connect=True)
    # phosphate ester
    phosphate_node = [(1, {'element': 'P', 'connect': 4}),
                       (2, {'element': 'O', 'connect': 1}),
                       (3, {'element': 'O', 'connect': 1}),
                       (4, {'element': 'O', 'connect': 1}),
                       (5, {'element': 'O', 'connect': 2})]
    phosphate_edge = [(1,2), (1,3), (1,4), (1,5)]
    phosphate_graph = build_substructure_graph(phosphate_node, phosphate_edge)
    phosphate_count = count_substructures(str_graph, phosphate_graph, connect=True)
    # phosphoryl
    phosphoryl_node = [(1, {'element': 'P', 'connect': 2}),
                       (2, {'element': 'O', 'connect': 2})]
    phosphoryl_edge = [(1,2)]
    phosphoryl_graph = build_substructure_graph(phosphoryl_node, phosphoryl_edge)
    phosphoryl_count = count_substructures(str_graph, phosphoryl_graph, connect=True)
    #phosphite
    phosphite_node = [(1, {'element': 'P', 'connect': 4}),
                       (2, {'element': 'O', 'connect': 1}),
                       (3, {'element': 'O', 'connect': 1}),
                       (4, {'element': 'O', 'connect': 1})]
    phosphite_edge = [(1,2), (1,3), (1,4)]
    phosphite_graph = build_substructure_graph(phosphite_node, phosphite_edge)
    phosphite_count = count_substructures(str_graph, phosphite_graph, connect=True)
    # iso-Butyl
    iso_bytyl_node = [(1, {'element': 'C', 'connect': 2}),
                       (2, {'element': 'C', 'connect': 3}),
                       (3, {'element': 'C', 'connect': 1}),
                       (4, {'element': 'C', 'connect': 1})]
    iso_bytyl_edge = [(1,2), (2,3), (2,4)]
    iso_bytyl_graph = build_substructure_graph(iso_bytyl_node, iso_bytyl_edge)
    iso_bytyl_count = count_substructures(str_graph, iso_bytyl_graph, connect=True)
    # tert-Butyl
    tert_bytyl_node = [(1, {'element': 'C', 'connect': 4}),
                       (2, {'element': 'C', 'connect': 1}),
                       (3, {'element': 'C', 'connect': 1}),
                       (4, {'element': 'C', 'connect': 1})]
    tert_bytyl_edge = [(1,2), (1,3), (1,4)]
    tert_bytyl_graph = build_substructure_graph(tert_bytyl_node, tert_bytyl_edge)
    tert_bytyl_count = count_substructures(str_graph, tert_bytyl_graph, connect=True)
    # nropentyl
    nropentyl_node = [(1, {'element': 'C', 'connect': 2}),
                       (2, {'element': 'C', 'connect': 4}),
                       (3, {'element': 'C', 'connect': 1}),
                       (4, {'element': 'C', 'connect': 1}),
                       (5, {'element': 'C', 'connect': 1})]
    nropentyl_edge = [(1,2), (2,3), (2,4), (2,5)]
    nropentyl_graph = build_substructure_graph(nropentyl_node, nropentyl_edge)
    nropentyl_count = count_substructures(str_graph, nropentyl_graph, connect=True)
    # Carboxyl
    carboxyl_node = [(1, {'element': 'C', 'connect': 3}),
                     (2, {'element': 'O', 'connect': 1}),
                     (3, {'element': 'O', 'connect': 1})]
    carboxyl_edge = [(1,2), (1,3)]
    carboxyl_graph = build_substructure_graph(carboxyl_node, carboxyl_edge)
    carboxyl_count = count_substructures(str_graph, carboxyl_graph, connect=True)
    # carbonyl
    carbonyl_node = [(1, {'element': 'C', 'connect': 3}),
                     (2, {'element': 'O', 'connect': 1})]
    carbonyl_edge = [(1,2)]
    carbonyl_graph = build_substructure_graph(carbonyl_node, carbonyl_edge)
    carbonyl_count = count_substructures(str_graph, carbonyl_graph, connect=True)
    # anhydride
    anhydride_node = [(1, {'element': 'C', 'connect': 3}),
                     (2, {'element': 'O', 'connect': 1}),
                     (3, {'element': 'O', 'connect': 1}),
                     (4, {'element': 'C', 'connect': 3}),
                     (5, {'element': 'O', 'connect': 1})]
    anhydride_edge = [(1,2), (2,3), (3,4), (4,5)]
    anhydride_graph = build_substructure_graph(anhydride_node, anhydride_edge)
    anhydride_count = count_substructures(str_graph, anhydride_graph, connect=True)
    # peroxy
    peroxy_node = [(1, {'element': 'O', 'connect': 2}),
                     (2, {'element': 'O', 'connect': 2})]
    peroxy_edge = [(1,2)]
    peroxy_graph = build_substructure_graph(peroxy_node, peroxy_edge)
    peroxy_count = count_substructures(str_graph, peroxy_graph, connect=True)
    # Hydroxymethyl
    Hydroxymethyl_node = [(1, {'element': 'C', 'connect': 2}),
                     (2, {'element': 'O', 'connect': 1})]
    Hydroxymethyl_edge = [(1,2)]
    Hydroxymethyl_graph = build_substructure_graph(Hydroxymethyl_node, Hydroxymethyl_edge)
    Hydroxymethyl_count = count_substructures(str_graph, Hydroxymethyl_graph, connect=True)
    # carbonate
    carbonate_node = [(1, {'element': 'O', 'connect': 2}),
                     (2, {'element': 'C', 'connect': 3}),
                     (3, {'element': 'O', 'connect': 1}),
                     (4, {'element': 'O', 'connect': 2})]
    carbonate_edge = [(1,2), (2,3), (2,4)]
    carbonate_graph = build_substructure_graph(carbonate_node, carbonate_edge)
    carbonate_count = count_substructures(str_graph, carbonate_graph, connect=True)
    # secondary_amino 
    secondary_amino_node = [(1, {'element': 'N', 'connect': 2})]
    secondary_amino_edge = []
    secondary_amino_graph = build_substructure_graph(secondary_amino_node, secondary_amino_edge)
    secondary_amino_count = count_substructures(str_graph, secondary_amino_graph, connect=True)
    # tertiary amino 
    tertiary_amino_node = [(1, {'element': 'N', 'connect': 3})]
    tertiary_amino_edge = []
    tertiary_amino_graph = build_substructure_graph(tertiary_amino_node, tertiary_amino_edge)
    tertiary_amino_count = count_substructures(str_graph, tertiary_amino_graph, connect=True)
    # nitro
    nitro_node = [(1, {'element': 'N', 'connect': 3}),
                  (2, {'element': 'O', 'connect': 1}),
                  (3, {'element': 'O', 'connect': 1})]
    nitro_edge = [(1,2), (1,3)]
    nitro_graph = build_substructure_graph(nitro_node, nitro_edge)
    nitro_count = count_substructures(str_graph, nitro_graph, connect=True)
    # cyano
    cyano_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'N', 'connect': 1})]
    cyano_edge = [(1,2)]
    cyano_graph = build_substructure_graph(cyano_node, cyano_edge)
    cyano_count = count_substructures(str_graph, cyano_graph, connect=True)
    # isocyano
    isocyano_node = [(1, {'element': 'C', 'connect': 1}),
                  (2, {'element': 'N', 'connect': 2})]
    isocyano_edge = [(1,2)]
    isocyano_graph = build_substructure_graph(isocyano_node, isocyano_edge)
    isocyano_count = count_substructures(str_graph, isocyano_graph, connect=True)
    # azido
    azido_node = [(1, {'element': 'N', 'connect': 2}),
                  (2, {'element': 'N', 'connect': 2}),
                  (3, {'element': 'N', 'connect': 1})]
    azido_edge = [(1,2), (2,3)]
    azido_graph = build_substructure_graph(azido_node, azido_edge)
    azido_count = count_substructures(str_graph, azido_graph, connect=True)
    # nitroso
    nitroso_node = [(1, {'element': 'N', 'connect': 2}),
                  (2, {'element': 'O', 'connect': 1})]
    nitroso_edge = [(1,2)]
    nitroso_graph = build_substructure_graph(nitroso_node, nitroso_edge)
    nitroso_count = count_substructures(str_graph, nitroso_graph, connect=True)
    # sulfonic
    sulfonic_node = [(1, {'element': 'S', 'connect': 4}),
                  (2, {'element': 'O', 'connect': 1}),
                  (3, {'element': 'O', 'connect': 1}),
                  (4, {'element': 'O', 'connect': 1})]
    sulfonic_edge = [(1,2), (1,3), (1,4)]
    sulfonic_graph = build_substructure_graph(sulfonic_node, sulfonic_edge)
    sulfonic_count = count_substructures(str_graph, sulfonic_graph, connect=True)
    # sulfinic
    sulfinic_node = [(1, {'element': 'S', 'connect': 3}),
                  (2, {'element': 'O', 'connect': 1}),
                  (3, {'element': 'O', 'connect': 1})]
    sulfinic_edge = [(1,2), (1,3)]
    sulfinic_graph = build_substructure_graph(sulfinic_node, sulfinic_edge)
    sulfinic_count = count_substructures(str_graph, sulfinic_graph, connect=True)
    # sulfonyl
    sulfonyl_node = [(1, {'element': 'S', 'connect': 4}),
                  (2, {'element': 'O', 'connect': 1}),
                  (3, {'element': 'O', 'connect': 1})]
    sulfonyl_edge = [(1,2), (1,3)]
    sulfonyl_graph = build_substructure_graph(sulfonyl_node, sulfonyl_edge)
    sulfonyl_count = count_substructures(str_graph, sulfonyl_graph, connect=True)
    # thioester
    thioester_node = [(1, {'element': 'C', 'connect': 3}),
                  (2, {'element': 'O', 'connect': 1}),
                  (3, {'element': 'S', 'connect': 2})]
    thioester_edge = [(1,2), (1,3)]
    thioester_graph = build_substructure_graph(thioester_node, thioester_edge)
    thioester_count = count_substructures(str_graph, thioester_graph, connect=True)
    # thiocyanate
    thiocyanate_node = [(1, {'element': 'S', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'N', 'connect': 1})]
    thiocyanate_edge = [(1,2), (2,3)]
    thiocyanate_graph = build_substructure_graph(thiocyanate_node, thiocyanate_edge)
    thiocyanate_count = count_substructures(str_graph, thiocyanate_graph, connect=True)
    # isothiocyanate
    isothiocyanate_node = [(1, {'element': 'N', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'S', 'connect': 1})]
    isothiocyanate_edge = [(1,2), (2,3)]
    isothiocyanate_graph = build_substructure_graph(isothiocyanate_node, isothiocyanate_edge)
    isothiocyanate_count = count_substructures(str_graph, isothiocyanate_graph, connect=True)



    # Hydroxyamino 
    hydroxyamino_node = [(1, {'element': 'N', 'connect': 2}),
                     (2, {'element': 'O', 'connect': 1})]
    hydroxyamino_edge = [(1,2)]
    hydroxyamino_graph = build_substructure_graph(hydroxyamino_node, hydroxyamino_edge)
    hydroxyamino_count = count_substructures(str_graph, hydroxyamino_graph, connect=True)
    # ester
    ester_node = [(1, {'element': 'C', 'connect': 3}),
                  (2, {'element': 'O', 'connect': 1}),
                  (3, {'element': 'O', 'connect': 2})]
    ester_edge = [(1,2), (1,3)]
    ester_graph = build_substructure_graph(ester_node, ester_edge)
    ester_count = count_substructures(str_graph, ester_graph, connect=True)
    
    # guanidino group (ARG)
    guanidino_node = [(1, {'element': 'N', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 3}),
                  (3, {'element': 'N', 'connect': 1}),
                  (4, {'element': 'N', 'connect': 1})]
    guanidino_edge = [(1,2), (2,3), (2,4)]
    guanidino_graph = build_substructure_graph(guanidino_node, guanidino_edge)
    guanidino_count = count_substructures(str_graph, guanidino_graph, connect=True)
    # amide group
    amide_node = [(1, {'element': 'C', 'connect': 3}),
                  (2, {'element': 'O', 'connect': 1}),
                  (3, {'element': 'N', 'connect': 2})]
    amide_edge = [(1,2), (1,3)]
    amide_graph = build_substructure_graph(amide_node, amide_edge)
    amide_count = count_substructures(str_graph, amide_graph, connect=True)
    # amide group [End]
    amide_ac_node = [(1, {'element': 'C', 'connect': 3}),
                  (2, {'element': 'O', 'connect': 1}),
                  (3, {'element': 'N', 'connect': 1})]
    amide_ac_edge = [(1,2), (1,3)]
    amide_ac_graph = build_substructure_graph(amide_ac_node, amide_ac_edge)
    amide_ac_count = count_substructures(str_graph, amide_ac_graph, connect=True)
    # amide group [Pro]
    amide_pro_node = [(1, {'element': 'C', 'connect': 3}),
                  (2, {'element': 'O', 'connect': 1}),
                  (3, {'element': 'N', 'connect': 3})]
    amide_pro_edge = [(1,2), (1,3)]
    amide_pro_graph = build_substructure_graph(amide_pro_node, amide_pro_edge)
    amide_pro_count = count_substructures(str_graph, amide_pro_graph, connect=True)
    # glucose
    glucose_node = [(1, {'element': 'C'}),
                     (2, {'element': 'C'}),
                     (3, {'element': 'C'}),
                     (4, {'element': 'C'}),
                     (5, {'element': 'C'}),
                     (6, {'element': 'C'}),
                     (7, {'element': 'O'}),
                     (8, {'element': 'O'}),
                     (9, {'element': 'O'}),
                     (10, {'element': 'O'}),
                     (11, {'element': 'O'}),
                     (12, {'element': 'O'})]
    glucose_edge = [(1,7), (1,2), (2,8), (2,3), (3,9), (3,4), (4,10), (4,5), (5,6), (6,11), (5,12), (1,12)]
    glucose_graph = build_substructure_graph(glucose_node, glucose_edge)
    glucose_count = count_substructures(str_graph, glucose_graph)
    # pentose/ribose (RNA)
    pentose_node = [(1, {'element': 'C'}),
                     (2, {'element': 'C'}),
                     (3, {'element': 'C'}),
                     (4, {'element': 'C'}),
                     (5, {'element': 'C'}),
                     (6, {'element': 'O'}),
                     (7, {'element': 'O'}),
                     (8, {'element': 'O'}),
                     (9, {'element': 'O'}),
                     (10, {'element': 'O'}),
                     (11, {'element': 'O'})]
    pentose_edge = [(1,2), (2,7), (2,3), (3,8), (3,4), (4,9), (4,5), (5,10), (5,11), (1,11)]
    pentose_graph = build_substructure_graph(pentose_node, pentose_edge)
    pentose_count = count_substructures(str_graph, pentose_graph)
    # Deoxyribose (DNA)
    deoxyribose_node = [(1, {'element': 'C'}),
                     (2, {'element': 'C'}),
                     (3, {'element': 'C'}),
                     (4, {'element': 'C'}),
                     (5, {'element': 'C'}),
                     (6, {'element': 'O'}),
                     (7, {'element': 'O'}),
                     (8, {'element': 'O'}),
                     (9, {'element': 'O'})]
    deoxyribose_edge = [(1,6), (1,2), (2,3), (3,7), (3,4), (4,5), (5,8), (4,9), (1,9)]
    deoxyribose_graph = build_substructure_graph(deoxyribose_node, deoxyribose_edge)
    deoxyribose_count = count_substructures(str_graph, deoxyribose_graph)
    # C3 linker
    c3_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2})]
    c3_edge = [(1,2), (2,3)]
    c3_graph = build_substructure_graph(c3_node, c3_edge)
    c3_count = count_substructures(str_graph, c3_graph, connect=True)
    # C5 linker
    c5_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 2})]
    c5_edge = [(1,2), (2,3), (3,4), (4,5)]
    c5_graph = build_substructure_graph(c5_node, c5_edge)
    c5_count = count_substructures(str_graph, c5_graph, connect=True)
    # C7 linker
    c7_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'C', 'connect': 2})]
    c7_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7)]
    c7_graph = build_substructure_graph(c7_node, c7_edge)
    c7_count = count_substructures(str_graph, c7_graph, connect=True)
    # c9 linker
    c9_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'C', 'connect': 2}),
                  (8, {'element': 'C', 'connect': 2}),
                  (9, {'element': 'C', 'connect': 2})]
    c9_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9)]
    c9_graph = build_substructure_graph(c9_node, c9_edge)
    c9_count = count_substructures(str_graph, c9_graph, connect=True)
    # c11 linker
    c11_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'C', 'connect': 2}),
                  (8, {'element': 'C', 'connect': 2}),
                  (9, {'element': 'C', 'connect': 2}),
                  (10, {'element': 'C', 'connect': 2}),
                  (11, {'element': 'C', 'connect': 2})]
    c11_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11)]
    c11_graph = build_substructure_graph(c11_node, c11_edge)
    c11_count = count_substructures(str_graph, c11_graph, connect=True)
    # c13 linker
    c13_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'C', 'connect': 2}),
                  (8, {'element': 'C', 'connect': 2}),
                  (9, {'element': 'C', 'connect': 2}),
                  (10, {'element': 'C', 'connect': 2}),
                  (11, {'element': 'C', 'connect': 2}),
                  (12, {'element': 'C', 'connect': 2}),
                  (13, {'element': 'C', 'connect': 2})]
    c13_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13)]
    c13_graph = build_substructure_graph(c13_node, c13_edge)
    c13_count = count_substructures(str_graph, c13_graph, connect=True)
    # C3 ring
    c3_ring_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2})]
    c3_ring_edge = [(1,2), (2,3), (1,3)]
    c3_ring_graph = build_substructure_graph(c3_ring_node, c3_ring_edge)
    c3_ring_count = count_substructures(str_graph, c3_ring_graph, connect=False)
    # C4 ring
    c4_ring_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2})]
    c4_ring_edge = [(1,2), (2,3), (3,4), (1,4)]
    c4_ring_graph = build_substructure_graph(c4_ring_node, c4_ring_edge)
    c4_ring_count = count_substructures(str_graph, c4_ring_graph, connect=False)
    # C5 ring
    c5_ring_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 2})]
    c5_ring_edge = [(1,2), (2,3), (3,4), (4,5), (1,5)]
    c5_ring_graph = build_substructure_graph(c5_ring_node, c5_ring_edge)
    c5_ring_count = count_substructures(str_graph, c5_ring_graph, connect=False)
    # C6 ring
    c6_ring_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2})]
    c6_ring_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (1,6)]
    c6_ring_graph = build_substructure_graph(c6_ring_node, c6_ring_edge)
    c6_ring_count = count_substructures(str_graph, c6_ring_graph, connect=False)
    # C7 ring
    c7_ring_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'C', 'connect': 2})]
    c7_ring_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (1,7)]
    c7_ring_graph = build_substructure_graph(c7_ring_node, c7_ring_edge)
    c7_ring_count = count_substructures(str_graph, c7_ring_graph, connect=False)
    # X3 ring
    X3_ring_node = [(1, {'element': 'X', 'connect': 2}),
                  (2, {'element': 'X', 'connect': 2}),
                  (3, {'element': 'X', 'connect': 2})]
    X3_ring_edge = [(1,2), (2,3), (1,3)]
    X3_ring_graph = build_substructure_graph(X3_ring_node, X3_ring_edge)
    X3_ring_count = count_substructures(str_graph, X3_ring_graph, connect=False)
    # X4 ring
    X4_ring_node = [(1, {'element': 'X', 'connect': 2}),
                  (2, {'element': 'X', 'connect': 2}),
                  (3, {'element': 'X', 'connect': 2}),
                  (4, {'element': 'X', 'connect': 2})]
    X4_ring_edge = [(1,2), (2,3), (3,4), (1,4)]
    X4_ring_graph = build_substructure_graph(X4_ring_node, X4_ring_edge)
    X4_ring_count = count_substructures(str_graph, X4_ring_graph, connect=False)
    # C5 ring
    X5_ring_node = [(1, {'element': 'X', 'connect': 2}),
                  (2, {'element': 'X', 'connect': 2}),
                  (3, {'element': 'X', 'connect': 2}),
                  (4, {'element': 'X', 'connect': 2}),
                  (5, {'element': 'X', 'connect': 2})]
    X5_ring_edge = [(1,2), (2,3), (3,4), (4,5), (1,5)]
    X5_ring_graph = build_substructure_graph(X5_ring_node, X5_ring_edge)
    X5_ring_count = count_substructures(str_graph, X5_ring_graph, connect=False)
    # C6 ring
    X6_ring_node = [(1, {'element': 'X', 'connect': 2}),
                  (2, {'element': 'X', 'connect': 2}),
                  (3, {'element': 'X', 'connect': 2}),
                  (4, {'element': 'X', 'connect': 2}),
                  (5, {'element': 'X', 'connect': 2}),
                  (6, {'element': 'X', 'connect': 2})]
    X6_ring_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (1,6)]
    X6_ring_graph = build_substructure_graph(X6_ring_node, X6_ring_edge)
    X6_ring_count = count_substructures(str_graph, X6_ring_graph, connect=False)
    # C7 ring
    X7_ring_node = [(1, {'element': 'X', 'connect': 2}),
                  (2, {'element': 'X', 'connect': 2}),
                  (3, {'element': 'X', 'connect': 2}),
                  (4, {'element': 'X', 'connect': 2}),
                  (5, {'element': 'X', 'connect': 2}),
                  (6, {'element': 'X', 'connect': 2}),
                  (7, {'element': 'X', 'connect': 2})]
    X7_ring_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (1,7)]
    X7_ring_graph = build_substructure_graph(X7_ring_node, X7_ring_edge)
    X7_ring_count = count_substructures(str_graph, X7_ring_graph, connect=False)
    # benzyl ring
    benzyl_ring_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 3}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'C', 'connect': 2})]
    benzyl_ring_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (2,7)]
    benzyl_ring_graph = build_substructure_graph(benzyl_ring_node, benzyl_ring_edge)
    benzyl_ring_count = count_substructures(str_graph, benzyl_ring_graph, connect=True)
    # tolyl ring
    tolyl_ring_node = [(1, {'element': 'C', 'connect': 3}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 3}),
                  (5, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'C', 'connect': 1})]
    tolyl_ring_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,1), (4,7)]
    tolyl_ring_graph = build_substructure_graph(tolyl_ring_node, tolyl_ring_edge)
    tolyl_ring_count = count_substructures(str_graph, tolyl_ring_graph, connect=True)
    # Naphthyl
    Naphthyl_ring_node = [(1, {'element': 'C', 'connect': 3}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 3}),
                  (5, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'C', 'connect': 1}),
                  (8, {'element': 'C', 'connect': 3}),
                  (9, {'element': 'C', 'connect': 2}),
                  (10, {'element': 'C', 'connect': 2})]
    Naphthyl_ring_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,1), 
                          (2,7), (7,8), (8,9), (9,10), (10,3)]
    Naphthyl_ring_graph = build_substructure_graph(Naphthyl_ring_node, Naphthyl_ring_edge)
    Naphthyl_ring_count = count_substructures(str_graph, Naphthyl_ring_graph, connect=False)
    # Imidazole
    imidazole_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 2}),
                (3, {'element': 'C', 'connect': 2}),
                (4, {'element': 'N', 'connect': 2}),
                (5, {'element': 'C', 'connect': 2})]
    imidazole_edge = [(1,2), (2,3), (3,4), (4,5), (1,5)]
    imidazole_graph = build_substructure_graph(imidazole_node, imidazole_edge)
    imidazole_count = count_substructures(str_graph, imidazole_graph, connect=False)
    # thioether 
    thioether_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'S', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),]
    thioether_edge = [(1,2), (2,3), (1,3)]
    thioether_graph = build_substructure_graph(thioether_node, thioether_edge)
    thioether_count = count_substructures(str_graph, thioether_graph, connect=False)
    # epoxides
    epoxides_node = [(1, {'element': 'C', 'connect': 3}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'O', 'connect': 2}),]
    epoxides_edge = [(1,2), (2,3), (1,3)]
    epoxides_graph = build_substructure_graph(epoxides_node, epoxides_edge)
    epoxides_count = count_substructures(str_graph, epoxides_graph, connect=True)
    # tetrahydrofuran, THF
    tetrahydrofuran_node = [(1, {'element': 'O'}),
                     (2, {'element': 'C'}),
                     (3, {'element': 'C'}),
                     (4, {'element': 'C'}),
                     (5, {'element': 'C'})]
    tetrahydrofuran_edge = [(1,2), (2,3), (3,4), (4,5), (1,5)]
    tetrahydrofuran_graph = build_substructure_graph(tetrahydrofuran_node, tetrahydrofuran_edge)
    tetrahydrofuran_count = count_substructures(str_graph, tetrahydrofuran_graph)
    # Piperidine
    piperidine_node = [(1, {'element': 'C', 'connect': 2}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'N', 'connect': 2})]
    piperidine_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (1,6)]
    piperidine_graph = build_substructure_graph(piperidine_node, piperidine_edge)
    piperidine_count = count_substructures(str_graph, piperidine_graph, connect=True)
    # piperazine
    piperazine_node = [(1, {'element': 'N', 'connect': 2}),
                (2, {'element': 'C', 'connect': 2}),
                (3, {'element': 'C', 'connect': 2}),
                (4, {'element': 'N', 'connect': 2}),
                (5, {'element': 'C', 'connect': 2}),
                (6, {'element': 'C', 'connect': 2})]
    piperazine_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (1,6)]
    piperazine_graph = build_substructure_graph(piperazine_node, piperazine_edge)
    piperazine_count = count_substructures(str_graph, piperazine_graph, connect=True)
    # glyceraldehyde/dihydroxyacetone (3c)
    glyceraldehyde_node = [(1, {'element': 'C'}),
                     (2, {'element': 'C'}),
                     (3, {'element': 'C'}),
                     (6, {'element': 'O'}),
                     (7, {'element': 'O'}),
                     (8, {'element': 'O'})]
    glyceraldehyde_edge = [(1,2), (2,3), (1,6), (2,7), (3,8)]
    glyceraldehyde_graph = build_substructure_graph(glyceraldehyde_node, glyceraldehyde_edge)
    glyceraldehyde_count = count_substructures(str_graph, glyceraldehyde_graph)
    # tetrose
    tetrose_node = [(1, {'element': 'C'}),
                     (2, {'element': 'C'}),
                     (3, {'element': 'C'}),
                     (4, {'element': 'C'}),
                     (6, {'element': 'O'}),
                     (7, {'element': 'O'}),
                     (8, {'element': 'O'}),
                     (9, {'element': 'O'})]
    tetrose_edge = [(1,2), (2,3), (3,4), (1,6), (2,7), (3,8), (4,9)]
    tetrose_graph = build_substructure_graph(tetrose_node, tetrose_edge)
    tetrose_count = count_substructures(str_graph, tetrose_graph)
    # adenosine
    adenosine_node = [(1, {'element': 'C', 'connect': 2}),
                     (2, {'element': 'N', 'connect': 2}),
                     (3, {'element': 'C', 'connect': 3}),
                     (4, {'element': 'C', 'connect': 3}),
                     (5, {'element': 'N', 'connect': 2}),
                     (6, {'element': 'C', 'connect': 2}),
                     (7, {'element': 'N', 'connect': 3}),
                     (8, {'element': 'C', 'connect': 3}),
                     (9, {'element': 'N', 'connect': 2}),
                     (10, {'element': 'N', 'connect': 1})]
    adenosine_edge = [(1,2), (2,3), (3,4), (4,5), (4,8), (5,6), (6,7), (7,8), (8,9), (1,9), (3,10)]
    adenosine_graph = build_substructure_graph(adenosine_node, adenosine_edge)
    adenosine_count = count_substructures(str_graph, adenosine_graph, connect= True)
    # guanine
    guanine_node = [(1, {'element': 'C', 'connect': 3}),
                     (2, {'element': 'N', 'connect': 2}),
                     (3, {'element': 'C', 'connect': 3}),
                     (4, {'element': 'C', 'connect': 3}),
                     (5, {'element': 'N', 'connect': 2}),
                     (6, {'element': 'C', 'connect': 2}),
                     (7, {'element': 'N', 'connect': 3}),
                     (8, {'element': 'C', 'connect': 3}),
                     (9, {'element': 'N', 'connect': 2}),
                     (10, {'element': 'N', 'connect': 1}),
                     (11, {'element': 'O', 'connect': 1})]
    guanine_edge = [(1,2), (2,3), (3,4), (4,5), (4,8), (5,6), (6,7), (7,8), (8,9), (1,9), (1,10), (3,11)]
    guanine_graph = build_substructure_graph(guanine_node, guanine_edge)
    guanine_count = count_substructures(str_graph, guanine_graph, connect= True)
    # uracil
    uracil_node = [(1, {'element': 'C', 'connect': 3}),
                     (2, {'element': 'N', 'connect': 3}),
                     (3, {'element': 'C', 'connect': 2}),
                     (4, {'element': 'C', 'connect': 2}),
                     (5, {'element': 'C', 'connect': 3}),
                     (6, {'element': 'N', 'connect': 2}),
                     (7, {'element': 'O', 'connect': 1}),
                     (8, {'element': 'O', 'connect': 1})]
    uracil_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (1,6), (1,7), (5,8)]
    uracil_graph = build_substructure_graph(uracil_node, uracil_edge)
    uracil_count = count_substructures(str_graph, uracil_graph, connect= True)
    # cytosine
    cytosine_node = [(1, {'element': 'C', 'connect': 3}),
                     (2, {'element': 'N', 'connect': 3}),
                     (3, {'element': 'C', 'connect': 2}),
                     (4, {'element': 'C', 'connect': 2}),
                     (5, {'element': 'C', 'connect': 3}),
                     (6, {'element': 'N', 'connect': 2}),
                     (7, {'element': 'O', 'connect': 1}),
                     (8, {'element': 'N', 'connect': 1})]
    cytosine_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (1,6), (1,7), (5,8)]
    cytosine_graph = build_substructure_graph(cytosine_node, cytosine_edge)
    cytosine_count = count_substructures(str_graph, cytosine_graph, connect= True)
    # thymine
    thymine_node = [(1, {'element': 'C', 'connect': 3}),
                     (2, {'element': 'N', 'connect': 3}),
                     (3, {'element': 'C', 'connect': 2}),
                     (4, {'element': 'C', 'connect': 3}),
                     (5, {'element': 'C', 'connect': 3}),
                     (6, {'element': 'N', 'connect': 2}),
                     (7, {'element': 'O', 'connect': 1}),
                     (8, {'element': 'O', 'connect': 1}),
                     (9, {'element': 'C', 'connect': 1})]
    thymine_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (1,6), (1,7), (4,9), (5,8)]
    thymine_graph = build_substructure_graph(thymine_node, thymine_edge)
    thymine_count = count_substructures(str_graph, thymine_graph, connect= True)
    # phenol
    phenol_ring_node = [(1, {'element': 'C', 'connect': 3}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 2}),
                  (6, {'element': 'C', 'connect': 2}),
                  (7, {'element': 'O', 'connect': 2})]
    phenol_ring_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (1,6), (1,7)]
    phenol_ring_graph = build_substructure_graph(phenol_ring_node, phenol_ring_edge)
    phenol_ring_count = count_substructures(str_graph, phenol_ring_graph, connect=False)
    # indole
    indole_ring_node = [(1, {'element': 'C', 'connect': 3}),
                  (2, {'element': 'C', 'connect': 2}),
                  (3, {'element': 'C', 'connect': 2}),
                  (4, {'element': 'C', 'connect': 2}),
                  (5, {'element': 'C', 'connect': 3}),
                  (6, {'element': 'C', 'connect': 3}),
                  (7, {'element': 'C', 'connect': 2}),
                  (8, {'element': 'C', 'connect': 2}),
                  (9, {'element': 'N', 'connect': 2})]
    indole_ring_edge = [(1,2), (2,3), (3,4), (4,5), (5,6), (1,6), (6,7), (7,8), (8,9), (5,9)]
    indole_ring_graph = build_substructure_graph(indole_ring_node, indole_ring_edge)
    indole_ring_count = count_substructures(str_graph, indole_ring_graph, connect=False)
    #
    features = [count_all, count_c, count_o, count_n, count_p, count_s, count_halogen, count_alkali, #8
    count_less_36, count_class_2, count_class_3, count_class_4, count_class_5, count_class_6, count_class_7, #7
    num_edges, O_1, O_2, C_1, C_2, C_3, C_4, N_1, N_2, N_3, S_1, S_2, P_3, P_5, #14
    gly_count, ala_count, pro_count, val_count, leu_count, ile_count, ser_count, thr_count, gln_count, asn_count, met_count, #11
    cys_count, phe_count, tyr_count, trp_count, asp_count, glu_count, arg_count, lys_count, his_count, #9
    gly_end_count, ala_end_count, pro_end_count, val_end_count, leu_end_count, ile_end_count, ser_end_count, thr_end_count, gln_end_count, asn_end_count, met_end_count,  #11
    cys_end_count, phe_end_count, tyr_end_count, trp_end_count, asp_end_count, glu_end_count, arg_end_count, lys_end_count, his_end_count, #9
    phosphoric_count, phosphate_count, phosphoryl_count, phosphite_count, #4
    iso_bytyl_count, tert_bytyl_count, nropentyl_count, carboxyl_count, carbonyl_count,  #5
    anhydride_count, peroxy_count, Hydroxymethyl_count, carbonate_count,  #4
    secondary_amino_count, tertiary_amino_count, nitro_count, cyano_count, isocyano_count, azido_count, nitroso_count, #7
    sulfonic_count, sulfinic_count, sulfonyl_count, thioester_count, thiocyanate_count, isothiocyanate_count, #6
    hydroxyamino_count, ester_count,    # 2
    guanidino_count,amide_count, amide_ac_count, amide_pro_count, glucose_count, pentose_count, deoxyribose_count, #7
    c3_count, c5_count, c7_count, c9_count, c11_count, c13_count, #6
    c3_ring_count, c4_ring_count, c5_ring_count, c6_ring_count, c7_ring_count,  #4
    X3_ring_count, X4_ring_count, X5_ring_count, X6_ring_count, X7_ring_count,  #4
    benzyl_ring_count, tolyl_ring_count, Naphthyl_ring_count,  #3
    imidazole_count, thioether_count, epoxides_count, tetrahydrofuran_count, piperidine_count, piperazine_count,  #6
    glyceraldehyde_count, tetrose_count, adenosine_count, guanine_count, uracil_count, cytosine_count, thymine_count,  #7
    phenol_ring_count, indole_ring_count]  #2
    return features
    
def count_nearby_atoms(protein, ligand, min_cut_off, max_cut_off):
    parser = PDBParser()
    str_protein = parser.get_structure('prot', protein)
    str_ligand = parser.get_structure('pept', ligand)
    
    protein_atoms = [ atom for model in str_protein
                     for chain in model
                     for residue in chain
                     for atom in residue
                     if atom.element != 'H' ]
    ligand_atoms = [ atom for model in str_ligand
                    for chain in model
                    for residue in chain
                    for atom in residue
                    if atom.element != 'H' ]
    protein_coords = np.array([atom.coord for atom in protein_atoms])
    ligand_coords = np.array([atom.coord for atom in ligand_atoms])

    deltas = protein_coords[:, None, :] - ligand_coords[None, :, :]
    distances = np.linalg.norm(deltas, axis= 2)

    valid_mask = (distances > min_cut_off) & (distances < max_cut_off)
    nearby_indices = np.any(valid_mask, axis= 1)
    nearby_atoms = [protein_atoms[i] for i in np.where(nearby_indices)[0]]
    residue_counts = {residue: [0, 0] for residue in 
                      ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                       'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                       'THR', 'TRP', 'TYR', 'VAL']}
    for atom in nearby_atoms:
        atom_name, residue = atom.get_name(), atom.get_parent().get_resname()
        if residue in residue_counts:
            if atom_name in ['N', 'C', 'CA', 'O']:
                residue_counts[residue][1] += 1
            else:
                residue_counts[residue][0] += 1
    features = [count for residue in residue_counts.values() for count in residue]
    return features

def hbond_features(protein, ligand, cut_off= 3.5):
    parser = PDBParser()
    protein_str = parser.get_structure('protein', protein)
    ligand_str = parser.get_structure('ligand', ligand)
    polar_protein = [atom.coord for atom in Selection.unfold_entities(protein_str, 'A')
                     if atom.element in ['N', 'O']
                     ]
    polar_ligand = [atom.coord for atom in Selection.unfold_entities(ligand_str, 'A')
                    if atom.element in ['N', 'O']
                    ]
    if not polar_protein or not polar_ligand:
        return[0]
    protein_coords = np.array(polar_protein)
    ligand_coords = np.array(polar_ligand)

    distance = np.linalg.norm(
        protein_coords[:, None, :] - ligand_coords[None, :, :],
        axis= 2
    )
    contact_count = np.sum(distance < cut_off)
    return [contact_count]

def hydrophobic_environment(protein, ligand, cut_off= 3.0):
    parser = PDBParser()
    protein_str = parser.get_structure('protein', protein)
    ligand_str = parser.get_structure('ligand', ligand)
    protein_atoms = [ (atom.element, atom.coord)
                     for atom in Selection.unfold_entities(protein_str, 'A') 
                     if atom.element != 'H']
    ligand_atoms = [ (atom.element, atom.coord) 
                     for atom in Selection.unfold_entities(ligand_str, 'A') 
                     if atom.element != 'H' ]
    if not protein_atoms or not ligand_atoms:
        return [0, 0]
    
    polar_element = {'O', 'N', 'F', 'CL', 'BR', 'I', 'P'}
    protein_elements, protein_coords = zip(*protein_atoms)
    protein_coords = np.array(protein_coords)
    protein_polar = np.array([e in polar_element for e in protein_elements], dtype= bool)

    ligand_elements, ligand_coords = zip(*ligand_atoms)
    ligand_coords = np.array(ligand_coords)
    ligand_polar = np.array([ e in polar_element for e in ligand_elements], dtype= bool)

    deltas = protein_coords[:, None, :] - ligand_coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)

    valid_pairs = distances < cut_off

    ligand_polar_mask = ligand_polar[None, :]
    protein_polar_mask = protein_polar[:, None]

    mismatch_condition = np.bitwise_xor(ligand_polar_mask, protein_polar_mask)
    hydrophobic_mismatch = np.sum(valid_pairs & mismatch_condition)

    contact_condition = ~mismatch_condition
    hydrophobic_contact = np.sum(valid_pairs & contact_condition)
    return [hydrophobic_mismatch, hydrophobic_contact ]

def electrostatic_environment(protein, ligand, cut_off= 6.0):
    parser = PDBParser()
    protein_str = parser.get_structure('protein', protein)
    ligand_str = parser.get_structure('ligand', ligand)

    positive_elements = { 'N', 'FE', 'ZN', 'CU', 'MN', 'CO', 'NI', 'MG', 'CA', 'K', 'NA', 'LI'}
    negative_elements = { 'O', 'S', 'CL', 'BR', 'I' , 'P'}

    protein_data = [ (atom.element.upper(), atom.coord) 
                    for atom in Selection.unfold_entities(protein_str, 'A')
                    if atom.element != 'H' ]
    ligand_data = [ (atom.element.upper(), atom.coord)
                   for atom in Selection.unfold_entities(ligand_str, 'A')
                   if atom.element != 'H' ]
    if not protein_data or not ligand_data:
        return [0, 0]
    
    protein_elements, protein_coords = zip(*protein_data)
    ligand_elements, ligand_coords = zip(*ligand_data)

    protein_coords = np.array(protein_coords)
    ligand_coords = np.array(ligand_coords)

    protein_positive = np.array([e in positive_elements for e in protein_elements], dtype= bool)
    protein_negative = np.array([e in negative_elements for e in protein_elements], dtype= bool)
    ligand_positive = np.array([e in positive_elements for e in ligand_elements], dtype= bool)
    ligand_negative = np.array([e in negative_elements for e in ligand_elements], dtype= bool)

    deltas = protein_coords[:, None, :] - ligand_coords[None, :, :]
    distances = np.linalg.norm(deltas, axis= 2)

    valid_dist = (distances > 3.0) & (distances < cut_off)

    positive_positive = protein_positive[:, None] & ligand_positive[None, :]
    negative_negative = protein_negative[:, None] & ligand_negative[None, :]
    mismatch_status = positive_positive | negative_negative

    positive_negaitve = protein_positive[:, None] & ligand_negative[None, :]
    negative_positive = protein_negative[:, None] & ligand_positive[None, :]
    contact_status = positive_negaitve | negative_positive

    electrostatic_mismatch = np.sum(valid_dist & mismatch_status)
    electrostatic_contact = np.sum(valid_dist & contact_status)

    return [ electrostatic_mismatch, electrostatic_contact]
    # return [0, 0]
def interaction_features(protein, ligand):
    interaction_features = []
    features_0_2 = count_nearby_atoms(protein, ligand, 0, 2)
    features_2_3 = count_nearby_atoms(protein, ligand, 2, 3)
    features_3_4 = count_nearby_atoms(protein, ligand, 3, 4)
    features_4_5 = count_nearby_atoms(protein, ligand, 4, 5)
    interaction_features = features_0_2 + features_2_3 + features_3_4 + features_4_5
    return interaction_features

def ligand_classification_features(protein_pdb, input_pdb):
    bio_str = read_file(input_pdb)
    ligand_features = get_features(bio_str)
    interaction_info = interaction_features(protein_pdb, input_pdb)
    hbond_info = hbond_features(protein_pdb, input_pdb, cut_off= 3.5)
    hydrophobic_features = hydrophobic_environment(protein_pdb, input_pdb, cut_off= 3.0)
    electrostatic_features = electrostatic_environment(protein_pdb, input_pdb, cut_off= 6.0)
    all_features = ligand_features + interaction_info + hbond_info + hydrophobic_features + electrostatic_features
    return all_features
    
def load_model_and_pred(input_pdb, LGBM_Model_package):
    '''
    0: ions
    1: ligand
    2: mem
    3: dna
    4: gly
    5: organic
    6: peptide
    7: rna
    '''  
    try:
        from ligandexplorer.workflow import ModelContainer
        if ModelContainer.model_1 is None:
            raise RuntimeError('model was not load0')
        ligand_str = read_file(input_pdb)
        features = get_features(ligand_str)
        features = np.array(features).reshape(1,-1)
        scaled_features = ModelContainer.scaler_1.transform(features)
        prediction = ModelContainer.model_1.predict(scaled_features)
        if int(prediction[0]) == 2:
            scaled_2_features = ModelContainer.scaler_2.transform(features)
            prediction = ModelContainer.model_2.predict(scaled_2_features)
            if int(prediction[0]) == 0:
                return 'dna'
            elif int(prediction[0]) == 1:
                return 'gly'
            elif int(prediction[0]) == 2:
                return 'organic'
            elif int(prediction[0]) == 3:
                return 'peptide'
            elif int(prediction[0]) == 4:
                return 'rna'
        elif int(prediction[0]) == 0:
            return 'ions'
        elif int(prediction[0]) == 1:
            return 'men'
        # return prediction[0]
    except ImportError as e:
        print(e)
    except AttributeError as e:
        print(e)

def load_model_and_pred_ligand(protein_pdb, ligand_pdb, LGBM_Model_package):
    '''
    0: not a ligand
    1: ligand
    '''
    try:
        from ligandexplorer.workflow import ModelContainer
        if ModelContainer.model_1 is None:
            raise RuntimeError('model was not load0')
        # model_3 = LGBM_Model_package[4]
        # scaler_3 = LGBM_Model_package[5]
        features = ligand_classification_features(protein_pdb, ligand_pdb)
        features = np.array(features).reshape(1,-1)
        scaler_features = ModelContainer.scaler_3.transform(features)
        prediction = ModelContainer.model_3.predict(scaler_features)
        return prediction[0]
    except ImportError as e:
        print(e)
    except AttributeError as e:
        print(e)




    
