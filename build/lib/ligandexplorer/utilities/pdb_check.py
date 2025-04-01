import os
from pymol import cmd
def pdb_legality_check(work_path= None, input_pdb= None):
    pdb_path = os.path.join(work_path, input_pdb)
    file_size_bytes = os.path.getsize(pdb_path)
    file_size_mb = file_size_bytes / ( 1024 * 1024)
    # file size check 
    if file_size_mb > 10:
        print(f'{pdb_path}: size {file_size_mb} MB, is larger than 10 MB')
        return False
    # element check
    cmd.delete('all')
    cmd.load(pdb_path, 'init_str')
    pymol_space = {'ele_name': [],
                   'ele_num': []}
    cmd.iterate('(init_str)','ele_name.append(name)',space=pymol_space)
    cmd.iterate('(init_str)','ele_num.append(index)',space=pymol_space)
    if len(set(pymol_space['ele_name'])) < 4:
        return False
    if len(set(pymol_space['ele_num'])) < 7:
        return False
    print('pdb legality check: pass')
    return True

if __name__ == "__main__":
    work_path='../test'
    input_pdb='8pfn'
    state = pdb_legality_check(work_path=work_path, input_pdb=input_pdb)
    print(work_path,input_pdb,state)
    
