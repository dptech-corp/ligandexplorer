import os
# Cap OpenMP threads to 1 *before* numpy / lightgbm / pytorch get imported.
# The LGBM backend forks worker processes so the parent's class-attribute
# models are inherited via copy-on-write; under fork, a multi-threaded
# OpenMP runtime initialised in the parent deadlocks the child the first
# time it calls into LGBM's C++ predict path. Each worker already runs in
# its own process so process-level parallelism replaces the lost
# thread-level parallelism with no throughput loss.
os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys
import zipfile
import tarfile
import rarfile
import shutil
import traceback
import pkg_resources
import argparse
import multiprocessing
import subprocess
from ligandexplorer.utilities.sanitize_pdb import is_empty_file
from ligandexplorer.utilities.formating import format_pdb_file
from ligandexplorer.utilities.ligand_discriminate import ligand_identify
from ligandexplorer.utilities.calculated_docking_grid import calculated_docking_grid
from ligandexplorer.utilities.non_covalent import non_covalent_workflow
from ligandexplorer.utilities.covalent import find_covalent_ligand
from ligandexplorer.utilities.remove_ligand import save_protein_without_ligand

def ligandexplorer_workflow(work_path, pdb_file, search_mode, identify_lig, box_size= 10, fix_pdb= False, LGBM_Model_package= None, debug= False ):
    if os.path.exists(os.path.join(work_path, pdb_file + '.cif')):
        input_pdb = pdb_file + '.cif'
    elif os.path.exists(os.path.join(work_path, pdb_file + '.mmcif')):
        input_pdb = pdb_file + '.mmcif'
    else:
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
            ext = '.cif' if input_pdb.endswith('.cif') or input_pdb.endswith('.mmcif') else '.pdb'
            other_pdb_file = [input_pdb, 'protein' + ext, 'water' + ext]
            all_ligand = [ f for f in os.listdir(work_path) if f.endswith(ext) and f not in other_pdb_file ]
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
        return filename.lower().endswith('.pdb') or filename.lower().endswith('.cif') or filename.lower().endswith('.mmcif')  

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
          subprocess.run(['7z', 'e', archive_path, '-o' + output_path, '*.pdb', '*.cif', '*.mmcif'], check=True, capture_output=True, text=True) # Extract PDBs and CIFs for 7zip  
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
    
def worker(real_work_path, pdb_name, search_mode, identify_lig, boxsize, fix_pdb, LGBM_Model_package, debug_mode):
    try:
        ligandexplorer_workflow(real_work_path, pdb_name, search_mode, identify_lig, boxsize, fix_pdb, LGBM_Model_package, debug_mode )  
    except Exception as e:
        error_file = os.path.join(real_work_path, 'error.txt')
        with open(error_file, 'a') as f_err:
            str_output = str(real_work_path) + '|' + str(e)
            f_err.write(str_output + '\n') 
        print(f"!!!!! Error in process: {e}")
        traceback.print_exc()

def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
def _load_gnn_models_cpu():
    """Load GNN models on CPU for direct inference in worker processes."""
    import torch
    from ligandexplorer.utilities.gnn_models import MoleculeClassifier, LigandBinaryClassifier
    script_dir = pkg_resources.resource_filename('ligandexplorer', 'model')
    ModelContainer.device = torch.device("cpu")
    mol_path = os.path.join(script_dir, 'mol_classifier.pt')
    mol_model = MoleculeClassifier(num_classes=8)
    mol_model.load_state_dict(torch.load(mol_path, map_location='cpu', weights_only=True))
    mol_model.eval()
    ModelContainer.mol_classifier = mol_model
    lig_path = os.path.join(script_dir, 'ligand_classifier.pt')
    if os.path.exists(lig_path):
        try:
            lig_model = LigandBinaryClassifier()
            lig_model.load_state_dict(torch.load(lig_path, map_location='cpu', weights_only=True))
            lig_model.eval()
            ModelContainer.ligand_classifier = lig_model
        except Exception as e:
            print(f'[WARNING] Failed to load ligand_classifier: {e}. Task B disabled.')
            ModelContainer.ligand_classifier = None
    else:
        ModelContainer.ligand_classifier = None
    # Peptide sub-classifier
    import numpy as np
    from ligandexplorer.utilities.gnn_models import PeptideSubClassifier
    sub_path = os.path.join(script_dir, "peptide_sub_classifier.pt")
    norm_path = os.path.join(script_dir, "peptide_sub_norm.npz")
    if os.path.exists(sub_path) and os.path.exists(norm_path):
        try:
            sub_model = PeptideSubClassifier()
            sub_model.load_state_dict(torch.load(sub_path, map_location="cpu", weights_only=True))
            sub_model.eval()
            ModelContainer.peptide_sub_classifier = sub_model
            norm_data = np.load(norm_path)
            ModelContainer.peptide_sub_norm_mean = norm_data["mean"]
            ModelContainer.peptide_sub_norm_std = norm_data["std"]
        except Exception as e:
            print(f"[WARNING] Failed to load peptide_sub_classifier: {e}")
            ModelContainer.peptide_sub_classifier = None


def _gpu_inference_loop(request_queue, response_queue):
    """Daemon: holds models on GPU, serves two-stage inference requests.

    Models are loaded once and kept resident on GPU. After each inference
    request, intermediate tensors are explicitly deleted and CUDA cache
    is cleared periodically to prevent memory fragmentation.
    """
    import torch
    import numpy as np
    from ligandexplorer.utilities.gnn_models import (
        MoleculeClassifier, LigandBinaryClassifier, PeptideSubClassifier)
    from ligandexplorer.utilities.gnn_workflow import (
        CATEGORIES, SUB_CATEGORIES, PEPTIDE_INDICES,
        _apply_chemical_rules, _run_sub_model)
    from torch_geometric.data import Data
    script_dir = pkg_resources.resource_filename('ligandexplorer', 'model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load models (resident on GPU) ---
    mol_model = MoleculeClassifier(num_classes=8)
    mol_model.load_state_dict(torch.load(
        os.path.join(script_dir, 'mol_classifier.pt'), map_location=device, weights_only=True))
    mol_model.to(device).eval()

    lig_model = None
    lig_path = os.path.join(script_dir, 'ligand_classifier.pt')
    if os.path.exists(lig_path):
        try:
            lig_model = LigandBinaryClassifier()
            lig_model.load_state_dict(torch.load(lig_path, map_location=device, weights_only=True))
            lig_model.to(device).eval()
        except Exception as e:
            print(f'[WARNING] Failed to load ligand_classifier: {e}. Task B disabled.')
            lig_model = None

    sub_model = None
    norm_mean = norm_std = None
    sub_path = os.path.join(script_dir, 'peptide_sub_classifier.pt')
    norm_path = os.path.join(script_dir, 'peptide_sub_norm.npz')
    if os.path.exists(sub_path) and os.path.exists(norm_path):
        try:
            sub_model = PeptideSubClassifier()
            sub_model.load_state_dict(torch.load(sub_path, map_location=device, weights_only=True))
            sub_model.to(device).eval()
            norm_data = np.load(norm_path)
            norm_mean = norm_data['mean']
            norm_std = norm_data['std']
        except Exception as e:
            print(f'[WARNING] Failed to load peptide_sub_classifier: {e}')
            sub_model = None

    # Force release loading temporaries
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    request_count = 0
    CACHE_CLEAR_INTERVAL = 50

    # --- Inference loop ---
    while True:
        msg = request_queue.get()
        if msg is None:
            break
        req_id, task_type, payload, reply_queue = msg
        try:
            z = torch.tensor(payload['z'], dtype=torch.long, device=device)
            pos = torch.tensor(payload['pos'], dtype=torch.float, device=device)
            edge_index = torch.tensor(payload['edge_index'], dtype=torch.long, device=device)
            edge_attr = torch.tensor(payload['edge_attr'], dtype=torch.float, device=device)
            batch = torch.zeros(z.size(0), dtype=torch.long, device=device)
            data = Data(z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            if 'edge_type' in payload:
                data.edge_type = torch.tensor(payload['edge_type'], dtype=torch.long, device=device)

            if task_type == 'mol':
                amide_count = payload.get('amide_count', 0)
                is_cyclic = payload.get('is_backbone_cyclic', 0)
                elem_ratios = payload.get('elem_ratios', np.zeros(5))
                func_groups = payload.get('func_groups', np.zeros(32))
                data.amide_count = torch.tensor([amide_count], dtype=torch.long, device=device)
                data.is_backbone_cyclic = torch.tensor([is_cyclic], dtype=torch.long, device=device)
                data.elem_ratios = torch.tensor(elem_ratios, dtype=torch.float, device=device)
                data.func_groups = torch.tensor(func_groups, dtype=torch.float, device=device)
                with torch.no_grad():
                    out = mol_model(data)
                    probs = torch.softmax(out, dim=-1).cpu().squeeze().numpy()
                ruled_probs = _apply_chemical_rules(probs, amide_count, elem_ratios, func_groups)
                pred = int(ruled_probs.argmax())
                pred_label = CATEGORIES[pred]
                if pred in PEPTIDE_INDICES and sub_model is not None:
                    pred_label = _run_sub_model(
                        sub_model, norm_mean, norm_std,
                        func_groups, elem_ratios, amide_count, is_cyclic, device)
                reply_queue.put((req_id, pred_label))

            elif task_type == 'ligand':
                if lig_model is None:
                    reply_queue.put((req_id, 0))
                else:
                    data.node_type = torch.tensor(payload['node_type'], dtype=torch.long, device=device)
                    data.ligand_mask = torch.tensor(payload['ligand_mask'], dtype=torch.bool, device=device)
                    with torch.no_grad():
                        pred = lig_model(data).argmax(-1).item()
                    reply_queue.put((req_id, pred))
            else:
                reply_queue.put((req_id, None))

        except Exception as e:
            print(f'GPU inference error: {e}')
            reply_queue.put((req_id, "organic" if task_type == 'mol' else 0))
        finally:
            # Release intermediate tensors immediately
            del data, z, pos, edge_index, edge_attr, batch
            if 'out' in dir():
                del out
            request_count += 1
            if device.type == 'cuda' and request_count % CACHE_CLEAR_INTERVAL == 0:
                torch.cuda.empty_cache()


def _load_lgbm_models():
    """Load the LGBM v2 cascade (models 1, 2, 3 + matching scalers).

    The v2 model pickles are dicts produced by joblib in the training
    sandbox; they carry an explicit ``schema_version`` string so any
    accidental shape / column-set mismatch surfaces immediately rather
    than silently producing wrong predictions.
    """
    import joblib
    from ligandexplorer.utilities.lgbm_featurizer import (
        FEATURE_SCHEMA_VERSION,
        FEATURE_SCHEMA_VERSION_FULL,
    )

    script_dir = pkg_resources.resource_filename('ligandexplorer', 'model')

    def _load_packaged(path, expected_version):
        obj = joblib.load(path)
        if not (isinstance(obj, dict) and 'model' in obj):
            raise RuntimeError(
                f"{os.path.basename(path)} is not a v2 packaged model "
                f"(expected dict with 'model' + 'schema_version' keys). "
                f"Re-train with lgbm_featurizer.train and place a "
                f"matching pickle in {script_dir!r}."
            )
        version = obj.get('schema_version')
        if version != expected_version:
            raise RuntimeError(
                f"{os.path.basename(path)} schema_version mismatch: "
                f"got {version!r}, expected {expected_version!r}."
            )
        return obj['model']

    ModelContainer.model_1 = _load_packaged(
        os.path.join(script_dir, 'model_1.pkl'),
        FEATURE_SCHEMA_VERSION,
    )
    ModelContainer.model_2 = _load_packaged(
        os.path.join(script_dir, 'model_2.pkl'),
        FEATURE_SCHEMA_VERSION,
    )
    ModelContainer.model_3 = _load_packaged(
        os.path.join(script_dir, 'model_3.pkl'),
        FEATURE_SCHEMA_VERSION_FULL,
    )
    ModelContainer.scaler_1 = joblib.load(
        os.path.join(script_dir, 'scaler_1.pkl')
    )
    ModelContainer.scaler_2 = joblib.load(
        os.path.join(script_dir, 'scaler_2.pkl')
    )
    ModelContainer.scaler_3 = joblib.load(
        os.path.join(script_dir, 'scaler_3.pkl')
    )


def _init_worker(backend, device_request, req_q, resp_q):
    """Pool initializer: set up ModelContainer once per worker process.

    For the LGBM backend the parent has already loaded the cascade
    (see :func:`main`) and fork's copy-on-write makes those
    ``ModelContainer.model_*`` / ``scaler_*`` class attributes visible
    here for free -- the ``if not models_loaded`` guard below is the
    fallback path for any caller that bypasses :func:`main` (e.g. unit
    tests that spawn a single worker directly).
    """
    ModelContainer.backend = backend
    ModelContainer.device_request = device_request
    if backend == 'gnn' and device_request == 'cuda':
        ModelContainer.device = 'cuda_proxy'
        ModelContainer.request_queue = req_q
        ModelContainer.response_queue = resp_q
        ModelContainer.worker_reply_queue = multiprocessing.Queue()
    elif backend == 'gnn':
        if not ModelContainer.models_loaded:
            _load_gnn_models_cpu()
    else:
        if not ModelContainer.models_loaded:
            _load_lgbm_models()
    ModelContainer.models_loaded = True


def worker_wrapper(task):  
    src_file, dest_dir, pdb_name, search_mode, identify_ligand, boxsize, fix_pdb, LGBM_Model_package, debug_mode = task
    os.makedirs(dest_dir, exist_ok= True)
    shutil.move(src_file, os.path.join(dest_dir, os.path.basename(src_file)))
    worker(dest_dir, pdb_name, search_mode, identify_ligand, boxsize, fix_pdb, LGBM_Model_package, debug_mode)


class ModelContainer:
    backend = 'gnn'
    device_request = 'cpu'
    models_loaded = False
    # GNN
    mol_classifier = None
    peptide_sub_classifier = None
    peptide_sub_norm_mean = None
    peptide_sub_norm_std = None
    ligand_classifier = None
    device = None
    # CUDA proxy
    request_queue = None
    response_queue = None  # kept for backward compat, unused in fixed path
    worker_reply_queue = None
    # LGBM
    model_1 = None
    scaler_1 = None
    model_2 = None
    scaler_2 = None
    model_3 = None
    scaler_3 = None

def main():
    # The start method is selected per-backend below via get_context()
    # so we no longer touch the global default.
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
    parser.add_argument('-m', '--backend',
                        default='gnn', choices=['gnn', 'lgbm'],
                        help='Model backend for ligand identification: gnn (default) or lgbm',
                        required=False)
    parser.add_argument('-d', '--device',
                        default='cpu', choices=['cpu', 'cuda'],
                        help='Device for GNN inference: cpu (default) or cuda',
                        required=False)
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
    ModelContainer.backend = args.backend
    if args.backend == 'lgbm' and args.device == 'cuda':
        args.device = 'cpu'
    ModelContainer.device_request = args.device

    try:
        # Soft-dependency probe -- PDBFixer / openmm.app are only used in
        # ligandexplorer.utilities.pdbfixer_wf, which is lazily imported.
        from pdbfixer import PDBFixer  # noqa: F401
        from openmm.app import PDBFile  # noqa: F401
    except ImportError:
        fix_pdb = False
        print('PDBFixer not found, protein will not be fixed')

    if core == None:
        core = multiprocessing.cpu_count()
    search_mode = 'strict' if strict_mode else 'normal'
    print(f'Running the workflow in {search_mode} mode with {core} of core, backend={ModelContainer.backend}')

    # Pick the multiprocessing start method based on backend:
    #   * LGBM is pure CPU. We load the v2 cascade in the parent ONCE
    #     and rely on fork's copy-on-write so the N workers share the
    #     same model objects without per-worker reloads or pickle
    #     round-trips.
    #   * GNN keeps 'spawn' because the CUDA proxy daemon (and worker
    #     CUDA contexts) are unsafe under fork().
    if ModelContainer.backend == 'lgbm':
        mp_ctx = multiprocessing.get_context('fork')
        _load_lgbm_models()
        ModelContainer.models_loaded = True
        from ligandexplorer.utilities.lgbm_featurizer import (
            FEATURE_SCHEMA_VERSION,
            FEATURE_SCHEMA_VERSION_FULL,
        )
        print(
            f'LGBM models loaded (mol schema={FEATURE_SCHEMA_VERSION}, '
            f'complex schema={FEATURE_SCHEMA_VERSION_FULL}); '
            f'shared with {core} workers via fork'
        )
    else:
        mp_ctx = multiprocessing.get_context('spawn')

    if silence_mode:
        print('Running the workflow in silence mode')
        sys.stdout = open(os.devnull, 'w')
    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    print(f"Extracting file {pdb_zip}, output path is {output_path}")
    extract_archive(pdb_zip, output_path)
    tasks = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith('.pdb') or file.lower().endswith('.cif') or file.lower().endswith('.mmcif'):
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
                    None, # LGBM model
                    debug_mode
                ))

    print(f"Total tasks to process: {len(tasks)}")

    gpu_process = None
    request_queue = None
    response_queue = None
    if ModelContainer.backend == 'gnn' and ModelContainer.device_request == 'cuda':
        request_queue = mp_ctx.Queue()
        response_queue = mp_ctx.Queue()
        gpu_process = mp_ctx.Process(target=_gpu_inference_loop,
                                     args=(request_queue, response_queue),
                                     daemon=True)
        gpu_process.start()

    pool = None
    try:
        pool = mp_ctx.Pool(
            processes=core,
            initializer=_init_worker,
            initargs=(ModelContainer.backend, ModelContainer.device_request,
                      request_queue, response_queue),
        )
        results = []
        chunk_size = 1
        for result in pool.imap_unordered(worker_wrapper, tasks, chunksize=chunk_size):
            results.append(result)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating workers...")
        if pool is not None:
            pool.terminate()
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if gpu_process is not None:
            request_queue.put(None)
            gpu_process.join(timeout=5)
            if gpu_process.is_alive():
                gpu_process.kill()
    # import time
    # print(time.time())
    print('====== ALL JOB DONE ======')

if __name__ == '__main__':
    main()

