import os
from Bio.PDB import NeighborSearch
import networkx as nx
from ligandexplorer.utilities.save_ligand import save_ligand
from ligandexplorer.utilities.formating import get_parser
from ligandexplorer.utilities.ligand_discriminate import clean_alt_structure

def non_covalent_workflow(init_pdb, work_path, debug):
    input_pdb = os.path.join(work_path, init_pdb)
    parser = get_parser(input_pdb, QUIET=True)
    structure = parser.get_structure('str', input_pdb)
    structure = clean_alt_structure(structure)
    graph = nx.Graph()
    for model in structure:
        for chain in model:
            chain_id_cache = chain.get_id()
            if chain_id_cache == " ":
                chain_id = '!' # '!' is a placeholder if not have chain id
            else:
                chain_id = chain_id_cache
            for residue in chain:
                hetatm, resseq, icode = residue.id
                if icode != " ":
                    residue_id = f'{resseq}{icode}'
                else:
                    residue_id = resseq
                residue_name = residue.get_resname()
                node_id = f"{residue_name}_{chain_id}_{residue_id}"
                graph.add_node(node_id)
    
    distance_thresholds = {
        ( "C", "C"): 1.84,
        ( "C", "N"): 1.76,
        ( "N", "C"): 1.76,
        ( "C", "O"): 1.72,
        ( "O", "C"): 1.72,
        ( "C", "S"): 2.10,
        ( "S", "C"): 2.10,
        ( "O", "P"): 1.95,
        ( "P", "O"): 1.95,
        ( "C", "CL"): 1.90,
        ( "CL", "C"): 1.90,
        ( "C", "BR"): 2.10,
        ( "BR", "C"): 2.10,
        ( "C", "I"): 2.30,
        ( "I", "C"): 2.30,
        ( "NA", "C"): 2.30,
        ( "C", "NA"): 2.30
    }

    atoms = [ atom for atom in structure.get_atoms() if atom.element != 'H' ]
    search = NeighborSearch(atoms)
    bonds = search.search_all(2.6)
    
    for atom1,atom2 in bonds:
        atom1_residue = atom1.get_parent().get_resname()
        atom2_residue = atom2.get_parent().get_resname()
        atom1_chain_cache = atom1.get_parent().get_parent().get_id()
        atom2_chain_cache = atom2.get_parent().get_parent().get_id()
        if atom1_chain_cache == " ":
            atom1_chain = "!"
        else:
            atom1_chain = atom1_chain_cache
            
        if atom2_chain_cache == " ":
            atom2_chain = "!"
        else:
            atom2_chain = atom2_chain_cache

        hetatm1,resseq1,icode1 = atom1.get_parent().id
        hetatm2,resseq2,icode2 = atom2.get_parent().id
        if icode1 != " ":
            atom1_id = f'{resseq1}{icode1}'
        else:
            atom1_id = resseq1
        if icode2 != " ":
            atom2_id = f'{resseq2}{icode2}'
        else:
            atom2_id = resseq2
        atom1_symbol = atom1.get_id()
        atom2_symbol = atom2.get_id()
        atom1_element = atom1.element
        atom2_element = atom2.element
        if (not (atom1_symbol == 'SG' and atom2_symbol == 'SG')
            and not (atom1_element == 'H' or atom2_element =='H')):
            node1_id = f'{atom1_residue}_{atom1_chain}_{atom1_id}'
            node2_id = f'{atom2_residue}_{atom2_chain}_{atom2_id}'
            if (atom1_element, atom2_element) in distance_thresholds:
                distance_limit = distance_thresholds[(atom1_element, atom2_element)]
                if atom1 - atom2 <= distance_limit:
                    graph.add_edge(node1_id,node2_id) 
            elif atom1 - atom2 <= 1.5:
                graph.add_edge(node1_id,node2_id) 
            else:
                pass
    
    connected_subgraphs = [ graph.subgraph(c).copy() for c in nx.connected_components(graph) ]
    small_subgraphs = [subgraph for subgraph in connected_subgraphs if subgraph.number_of_nodes() < 15]
    final_graph = graph.copy()
    if not small_subgraphs:
        print(f'No non-covalent ligand in {input_pdb}')
        return final_graph, False
    else:
        for i, subgraph in enumerate(small_subgraphs):
            save_ligand(input_pdb, subgraph, work_path)
            final_graph.remove_nodes_from(subgraph.nodes())
        return final_graph, True