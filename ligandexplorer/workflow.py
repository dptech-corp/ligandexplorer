import os
import sys
import zipfile
import tarfile
import rarfile
import shutil
import pickle
import pkg_resources
import argparse
import queue
import multiprocessing
import subprocess
from multiprocessing import Pool
from ligandexplorer.utilities.sanitize_pdb import is_empty_file
from ligandexplorer.utilities.formating import format_pdb_file
from ligandexplorer.utilities.ligand_discriminate import ligand_identify
from ligandexplorer.utilities.calculated_docking_grid import calculated_docking_grid
from ligandexplorer.utilities.non_covalent import non_covalent_workflow
from ligandexplorer.utilities.covalent import find_covalent_ligand
from ligandexplorer.utilities.remove_ligand import save_protein_without_ligand
from ligandexplorer.utilities.model_wrapper import ModelWrapper
# package: networkx, biopython, numpy, lightGBM, scikit-learn, pdbfixer

sys.modules['__main__'].ModelWrapper = ModelWrapper

def ligandexplorer_workflow(work_path, pdb_file, search_mode, identify_lig, box_size= 10, fix_pdb= False, LGBM_Model_package= None, debug= False ):
    input_pdb = pdb_file + '.pdb'
    # if file is empty, kill
    if is_empty_file(os.path.join(work_path, input_pdb)):
        return 999
    if debug:
            format_pdb_file(work_path= work_path, input_file= input_pdb, fix_pdb= fix_pdb, debug= True)
    else:
        format_pdb_file(work_path= work_path, input_file= input_pdb, fix_pdb= fix_pdb, debug= False)
    # NMR structure have many frame, use the first frame
    print(f'>>> Working at {work_path}, input pdb {pdb_file} ')
    # non-covalent ligand workflow
    # non_covalent_state: 1 found non_covalent ligand
    #                     0 non_covalent ligand not found
    print(f">>> Searching non covalent ligand in {work_path}")
    if debug:
        final_graph, non_covalent_state = non_covalent_workflow(input_pdb, work_path, debug= True)
    else:
        final_graph, non_covalent_state = non_covalent_workflow(input_pdb, work_path, debug= False)
    print(f">>> Searching covalent ligand in {work_path}")
    covalent_state = find_covalent_ligand(input_pdb, work_path, final_graph)
    if not non_covalent_state and not covalent_state:
        print(f">>> Ligand not found in {work_path}, {input_pdb}")
    else:
        save_protein_without_ligand(work_path, input_pdb )
        if identify_lig:
            ligand_identify(work_path, input_pdb, search_mode, LGBM_Model_package= LGBM_Model_package, add_size=box_size)
        else:
            other_pdb_file = [input_pdb, 'protein.pdb', 'water.pdb']
            all_ligand = [ f for f in os.listdir(work_path) if os.path.splitext(f)[-1] == '.pdb' and f not in other_pdb_file ]
            for final_f in all_ligand:
                calculated_docking_grid(work_path, final_f, add_size= box_size)

def extract_archive(archive_path, output_path):  
    """  
    Extracts PDB files from various archive formats (zip, tar, rar, 7z).  

    Args:  
        archive_path: Path to the archive file.  
        output_path: Path to the directory where PDB files will be extracted.  
    """  
    def is_pdb_file(filename):  
        return filename.lower().endswith('.pdb')  

    def extract_from_zip(archive_path, output_path):  
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:  
            for member in zip_ref.infolist():  
                if is_pdb_file(member.filename):  
                    with zip_ref.open(member) as source, open(os.path.join(output_path, os.path.basename(member.filename)), 'wb') as target:  
                        shutil.copyfileobj(source, target)  

    def extract_from_tar(archive_path, output_path):  
        with tarfile.open(archive_path, 'r:*') as tar_ref:  
            for member in tar_ref.getmembers():  
                if is_pdb_file(member.name):  
                    with tar_ref.extractfile(member) as source, open(os.path.join(output_path, os.path.basename(member.name)), 'wb') as target:  
                        shutil.copyfileobj(source, target)  
    
    def extract_from_rar(archive_path, output_path):  
        with rarfile.RarFile(archive_path) as rar_ref:  
            for member in rar_ref.infolist():  
                if is_pdb_file(member.filename):  
                     with rar_ref.open(member) as source, open(os.path.join(output_path, os.path.basename(member.filename)), 'wb') as target:  
                        shutil.copyfileobj(source, target)  

    def extract_from_7z(archive_path, output_path):  
      try:  
          subprocess.run(['7z', 'e', archive_path, '-o' + output_path, '*.pdb'], check=True, capture_output=True, text=True) # Extract only PDBs for 7zip  
      except FileNotFoundError:  
          raise RuntimeError("7z is not installed. Please install p7zip.")  
      except subprocess.CalledProcessError as e:  
          raise RuntimeError(f"7z extraction failed with error:\n{e.stderr}")  
    try:  
        if zipfile.is_zipfile(archive_path):  
            extract_from_zip(archive_path, output_path)  
        elif tarfile.is_tarfile(archive_path):  
            extract_from_tar(archive_path, output_path)  
        elif rarfile.is_rarfile(archive_path):  
            extract_from_rar(archive_path, output_path)  
        elif archive_path.lower().endswith(('.7z', '.7zip')):  
            extract_from_7z(archive_path, output_path)  
        else:  
            raise ValueError(f"Unsupported archive format: {archive_path}")  

    except Exception as e:  
        raise RuntimeError(f"Extraction error for file '{archive_path}': {e}")    
    
def worker(real_work_path, pdb_name, search_mode, identify_lig, boxsize, fix_pdb, debug_mode):
    try:
        ligandexplorer_workflow(real_work_path, pdb_name, search_mode, identify_lig, boxsize, fix_pdb, debug_mode )  
    except Exception as e:
        with open('error.txt', 'a') as f_err:
            str_output = str(real_work_path) + '|' + str(e)
            f_err.write(str_output + '\n') 
        print(f"!!!!! Error in process: {e}")

def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
def worker_wrapper(task):  
    src_file, dest_dir, pdb_name, search_mode, identify_ligand, boxsize, fix_pdb, debug_mode = task
    if not ModelContainer.models_loaded:
        script_dir = pkg_resources.resource_filename('ligandexplorer', 'model')
        model_1_pkl_file = os.path.join(script_dir, 'model_1.pkl')
        model_2_pkl_file = os.path.join(script_dir, 'model_2.pkl')
        scaler_1_file = os.path.join(script_dir, 'scaler_1.pkl')
        scaler_2_file = os.path.join(script_dir, 'scaler_2.pkl')
        model_3_pkl_file = os.path.join(script_dir, 'model_3.pkl')
        scaler_3_file = os.path.join(script_dir, 'scaler_3.pkl')
        with open(model_1_pkl_file, 'rb') as f_in:
            ModelContainer.model_1 = pickle.load(f_in)
        with open(model_2_pkl_file, 'rb') as f_in:
            ModelContainer.model_2 = pickle.load(f_in)
        with open(scaler_1_file, 'rb') as f_in:
            ModelContainer.scaler_1 = pickle.load(f_in)
        with open(scaler_2_file, 'rb') as f_in:
            ModelContainer.scaler_2 = pickle.load(f_in)
        with open(model_3_pkl_file, 'rb') as f_in:
            ModelContainer.model_3 = pickle.load(f_in)
        with open(scaler_3_file, 'rb') as f_in:
            ModelContainer.scaler_3 = pickle.load(f_in)
        ModelContainer.models_loaded = True
        print('Models loaded successfully')
    os.makedirs(dest_dir, exist_ok= True)
    shutil.move(src_file, os.path.join(dest_dir, os.path.basename(src_file)))
    worker(dest_dir, pdb_name, search_mode, identify_ligand, boxsize, fix_pdb, debug_mode)

class ModelContainer:
    models_loaded = False
    model_1 = None
    scaler_1 = None
    model_2 = None
    scaler_2 = None
    model_3 = None
    scaler_3 = None

def main():
    parser = argparse.ArgumentParser(description='Ligand explorer, version: v0.0.1')
    parser.add_argument('-o','--output_dir', 
                        default= 'output', 
                        help='defined the output path', 
                        required= True)
    parser.add_argument('-i','--input_zip', 
                        default=None, 
                        help='Path of the input zip file (rar, zip, tar, tar.gz, tar.bz2, bz2)', 
                        required=True)
    parser.add_argument('-f', '--fix_pdb_file',
                        default= False, type= str2bool,
                        help='Fix missing main chain atom by PDBFixer (if PDBFixer is install)',
                        required=False)
    parser.add_argument('-b','--box_size', 
                        default=10, type=int, 
                        help='defined the docking grid box size (angstrom), default is 10',
                        required= False)
    parser.add_argument('-c','--core',
                        default=None, type=int, 
                        help='defined the number of mult-process core, default is None <will use all core>', 
                        required=False)
    parser.add_argument('-s','--strict_mode', 
                        default= 'True', type= str2bool, 
                        help= 'If strict mode is on, workflow will remove identical ligands. default is True', 
                        required= False)
    parser.add_argument('-l','--lig_identify',
                        default= 'True', type= str2bool, 
                        help= 'The workflow will determine the type of all ligand (ions, solvent, nucleic acid, ligand, peptide. default is True)', 
                        required= False)
    parser.add_argument('-e','--silence_mode',
                        default= 'False', type= str2bool, 
                        help= 'Disable all output information. The workflow will run in silence. default is False)', 
                        required= False)
    parser.add_argument('-t', '--debug',
                        default= False, type= str2bool,
                        help= 'debug mode. output some debug information. default is False',
                        required= False)
    args = parser.parse_args()
    output_path = args.output_dir
    pdb_zip = args.input_zip
    boxsize = args.box_size
    core = args.core
    strict_mode = args.strict_mode
    identify_ligand = args.lig_identify
    silence_mode = args.silence_mode
    fix_pdb = args.fix_pdb_file
    debug_mode = args.debug

    try:
        from pdbfixer import PDBFixer 
        from openmm.app import PDBFile
    except ImportError:
        fix_pdb = False
        print('PDBFixer not found, protein will not be fixed')

    if core == None:
        core = multiprocessing.cpu_count()
    search_mode = 'strict' if strict_mode else 'normal'
    print(f'Running the workflow in {search_mode} mode with {core} of core')
    
    if silence_mode:
        print(f'Running the workflow in {silence_mode} mode')
        sys.stdout = open(os.devnull, 'w')
    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    print(f"Extracting file {pdb_zip}, output path is {output_path}")
    extract_archive(pdb_zip, output_path)
    tasks = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith('.pdb'):
                pdb_name = os.path.splitext(file)[0]
                src_path = os.path.join(root, file)
                dest_dir = os.path.join(output_path, pdb_name)
                tasks.append((
                    src_path,
                    dest_dir,
                    pdb_name,
                    search_mode,
                    identify_ligand,
                    boxsize,
                    fix_pdb,
                    # LGBM_Model_package,
                    debug_mode
                ))

    print(f"Total tasks to process: {len(tasks)}") 
    try:
        with Pool(processes= core) as pool:
            results = []
            chunk_size = 1
            for result in pool.imap_unordered(worker_wrapper, tasks, chunksize= chunk_size):
                results.append(result)
    except KeyboardInterrupt:  
        print("\nInterrupted by user. Terminating workers...")  
        pool.terminate()  
    finally:  
        pool.close()  
        pool.join()  
    import time
    print(time.time())
    print('====== ALL JOB DONE ======')

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()

