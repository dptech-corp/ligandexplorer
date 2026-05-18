import os
import copy
import networkx as nx
from ligandexplorer.utilities.save_ligand import save_ligand_covalent

def check_covalent(res_name, cha_id, res_id, str_graph= None):
    if cha_id == " ":
        check_node = f'{res_name}_!_{res_id}'
    else:
        check_node = f'{res_name}_{cha_id}_{res_id}'

    graph = str_graph
    res_id = res_id
    standard_residue = [ "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", 
                        "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
                        "THR", "TRP", "TYR", "VAL", "ACE", "NME"]
                    
    covalent_node = []

    for graph_node in graph.nodes():
        neighbors = list(graph.neighbors(graph_node))
        if len(neighbors) > 3:
            node_residue_name = graph_node.split('_')[0]
            if node_residue_name in standard_residue:
                covalent_node.append(graph_node)
                print('covalent residue:', graph_node)
    
    covalent_stat = False
    covalent_residue = ""
    interaction = [[], []]

    if covalent_node:
        covalent_res = []
        for cov_node in covalent_node:
            if nx.has_path(graph, check_node, cov_node):
                interaction = [check_node, cov_node]
                covalent_res.append(cov_node)
                print('covalent interaction:',check_node, cov_node)
                covalent_stat = True

    if covalent_stat:
        for cov_res in covalent_res:
            graph.remove_node(cov_res)
            
        covalent_residue = set(nx.bfs_tree(graph, source=interaction[0], reverse=True, depth_limit=None).nodes)
        if len(covalent_residue) < 15:
            return (covalent_stat, covalent_residue, interaction[1])
        else:
            covalent_stat = True
            covalent_residue = []
            return (covalent_stat, covalent_residue, [])
    else:
        return (covalent_stat, [], [])

def find_covalent_ligand(input_pdb, work_path, str_graph= None):
    initial_graph = copy.deepcopy(str_graph)
    covalent_nodes = []
    covalent_limit = 15
    standard_residue = [ "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", 
                        "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
                        "THR", "TRP", "TYR", "VAL"]
    for graph_node in initial_graph.nodes():
        neighbors = list(str_graph.neighbors(graph_node))
        if len(neighbors) > 3:
            node_residue_name = graph_node.split('_')[0]
            if node_residue_name in standard_residue:
                covalent_nodes.append(graph_node)
                print('covalent residue:', graph_node)

    if covalent_nodes != []:
        for covalent_node in covalent_nodes:
            str_graph.remove_node(covalent_node)
        connected_subgraphs = [ str_graph.subgraph(c).copy() for c in nx.connected_components(str_graph)]
        subgraphs = [ sg for sg in connected_subgraphs if sg.number_of_nodes() < covalent_limit]
        for i, subgraph in enumerate(subgraphs):
            init_pdb = os.path.join(work_path, input_pdb)
            save_ligand_covalent(init_pdb, subgraph, work_path)
        return True
    else:
        return False