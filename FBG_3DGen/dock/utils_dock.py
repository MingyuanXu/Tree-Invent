import dockstream
import os
import json 
import sys
import warnings
import tempfile
from shutil import copyfile

## target preperation
from dockstream.core.pdb_preparator import PDBPreparator
from dockstream.utils.dockstream_exceptions import *
from dockstream.core.rDock.rDock_target_preparator import rDockTargetPreparator
from dockstream.core.OpenEye.OpenEye_target_preparator import OpenEyeTargetPreparator
from dockstream.core.AutodockVina.AutodockVina_target_preparator import AutodockVinaTargetPreparator
from dockstream.containers.target_preparation_container import TargetPreparationContainer
from dockstream.utils.entry_point_functions.header import initialize_logging, set_environment
from dockstream.utils.enums.target_preparation_enum import TargetPreparationEnum
from dockstream.utils.enums.OpenEye_enums import OpenEyeTargetPreparationEnum
from dockstream.utils.enums.Gold_enums import GoldTargetPreparationEnum
from dockstream.utils.enums.AutodockVina_enums import AutodockTargetPreparationEnum
from dockstream.utils.enums.logging_enums import LoggingConfigEnum
from dockstream.utils.general_utils import *
import logging.config as logging_config
from dockstream.loggers.interface_logger import InterfaceLogger
from dockstream.utils.files_paths import dict_from_json_file
from dockstream.utils.enums.logging_enums import LoggingConfigEnum
from dockstream.utils.general_utils import *

##dock 
from dockstream.containers.docking_container import DockingContainer
from dockstream.core.rDock.rDock_docker import rDock
from dockstream.core.OpenEye.OpenEye_docker import OpenEye
from dockstream.core.Schrodinger.Glide_docker import Glide
from dockstream.core.AutodockVina.AutodockVina_docker import AutodockVina
#from dockstream.core.OpenEyeHybrid.OpenEyeHybrid_docker import OpenEyeHybrid
from dockstream.utils.entry_point_functions.header import initialize_logging, set_environment
from dockstream.utils.entry_point_functions.embedding import embed_ligands
from dockstream.utils.entry_point_functions.write_out import handle_poses_writeout, handle_score_printing, \
                                                         handle_scores_writeout
from dockstream.utils.enums.docking_enum import DockingConfigurationEnum
from dockstream.utils.enums.ligand_preparation_enum import LigandPreparationEnum
from dockstream.utils.enums.logging_enums import LoggingConfigEnum
from dockstream.utils.files_paths import attach_root_path
from dockstream.utils.dockstream_exceptions import *

def init_dockstream_logger(config,task,_task_enum,log_conf_path):
    _LE = LoggingConfigEnum()
    if in_keys(config, [task, _task_enum.HEADER]):
        log_conf_dict = dict_from_json_file(log_conf_path)
        if in_keys(config, [task, _task_enum.HEADER, _task_enum.LOGGING]):
            if in_keys(config, [task, _task_enum.HEADER, _task_enum.LOGGING, _task_enum.LOGGING_LOGFILE]):
                try:
                    log_conf_dict["handlers"]["file_handler"]["filename"] = config[task][_task_enum.HEADER][_task_enum.LOGGING][_task_enum.LOGGING_LOGFILE]
                    log_conf_dict["handlers"]["file_handler_blank"]["filename"] = config[task][_task_enum.HEADER][_task_enum.LOGGING][_task_enum.LOGGING_LOGFILE]
                except KeyError:
                    pass
        logging_config.dictConfig(log_conf_dict)
    else:
        logging_config.dictConfig(dict_from_json_file(log_conf_path))
    logger = InterfaceLogger()
    logger.log(f"DockStream version used: Modified in TreeInvent",_LE.INFO)
    return logger 

def dockstream_prepare_target(conf,validation=True,silent=True,debug=False,log_conf=None,dockstream_root_path='/data/myxu/Tree_Invent/envs/DockStream/'):
    _LE=LoggingConfigEnum()
    #print(_LE.PATH_CONFIG_DEFAULT)
    _TP=TargetPreparationEnum()
    if log_conf is None:
        log_conf=dockstream_root_path+_LE.PATH_CONFIG_DEFAULT
    if debug:
        log_conf=dockstream_root_path+_LE.PATH_CONFIG_DEBUG

    try:
        config =  TargetPreparationContainer(conf=conf, validation=True)
    except Exception as e:
        raise TargetPreparationFailed() from e

    logger = init_dockstream_logger(config=config, task=_TP.TARGETPREP, _task_enum=_TP,log_conf_path=log_conf)
    set_environment(config=config, task=_TP.TARGETPREP, _task_enum=_TP, logger=logger)
    try:
        with warnings.catch_warnings(record=True) as w:
            from dockstream.core.Gold.Gold_target_preparator import GoldTargetPreparator
            if len(w) > 0:
                logger.log("Could not load CCDC / Gold target preparator - if another backend is being used, you can safely ignore this warning.", _LE.DEBUG)
    except:
        logger.log("Could not load CCDC / Gold target preparator - if another backend is being used, you can safely ignore this warning.", _LE.DEBUG)

    temp_files=[]

    input_pdb_path = config[_TP.TARGETPREP][_TP.INPUT_PATH]

    if config[_TP.TARGETPREP][_TP.FIX][_TP.FIX_ENABLED]:
        pdb_prep = PDBPreparator(conf=config)
        temp_pdb_file = gen_temp_file(suffix=".pdb")

        # apply the specified fixing and set the input PDB file
        pdb_prep.fix_pdb(input_pdb_file=input_pdb_path,
                         output_pdb_file=temp_pdb_file)
        temp_files.append(temp_pdb_file)

        # clean-up (and make copy in case specified)
        if nested_get(config, [_TP.TARGETPREP, _TP.FIX, _TP.FIX_PBDOUTPUTPATH], default=False):
            try:
                copyfile(src=temp_pdb_file, dst=config[_TP.TARGETPREP][_TP.FIX][_TP.FIX_PBDOUTPUTPATH])
                input_pdb_path = config[_TP.TARGETPREP][_TP.FIX][_TP.FIX_PBDOUTPUTPATH]
            except:
                logger.log("Could not write fixed intermediate PDB file.", _LE.WARNING)
        else:
            input_pdb_path = temp_pdb_file
        logger.log(f"Wrote fixed PDB to file {input_pdb_path}.", _LE.DEBUG)

    for run_number, run in enumerate(config[_TP.TARGETPREP][_TP.RUNS]):
        logger.log(f"Started preparation run number {run_number}.", _LE.INFO)
        try:
            if run[_TP.RUNS_BACKEND] == _TP.RUNS_BACKEND_RDOCK:
                prep = rDockTargetPreparator(conf=config, target=input_pdb_path, run_number=run_number)
                result = prep.specify_cavity()
                logger.log("Wrote rDock cavity files to folder specified.",
                           _LE.INFO)
            elif run[_TP.RUNS_BACKEND] == _TP.RUNS_BACKEND_OPENEYE:
                _OpenEye_TP = OpenEyeTargetPreparationEnum()
                prep = OpenEyeTargetPreparator(conf=config, target=input_pdb_path, run_number=run_number)
                prep.specify_cavity()
                prep.write_target(path=run[_TP.RUNS_OUTPUT][_OpenEye_TP.OUTPUT_RECEPTORPATH])
                logger.log(f"Wrote OpenEye receptor to file {run[_TP.RUNS_OUTPUT][_OpenEye_TP.OUTPUT_RECEPTORPATH]}.",
                           _LE.INFO)
            elif run[_TP.RUNS_BACKEND] == _TP.RUNS_BACKEND_GOLD:
                _Gold_TP = GoldTargetPreparationEnum()
                prep = GoldTargetPreparator(conf=config, target=input_pdb_path, run_number=run_number)
                prep.specify_cavity()
                prep.write_target(path=run[_TP.RUNS_OUTPUT][_Gold_TP.OUTPUT_RECEPTORPATH])
                logger.log(f"Wrote Gold target to file {run[_TP.RUNS_OUTPUT][_Gold_TP.OUTPUT_RECEPTORPATH]}.",
                           _LE.INFO)
            elif run[_TP.RUNS_BACKEND] == _TP.RUNS_BACKEND_AUTODOCKVINA:
                _AD_TP = AutodockTargetPreparationEnum()
                prep = AutodockVinaTargetPreparator(conf=config, target=input_pdb_path, run_number=run_number)
                prep.specify_cavity()
                prep.write_target(path=run[_AD_TP.RUNS_OUTPUT][_AD_TP.RECEPTOR_PATH])
                logger.log(f"Wrote AutoDock Vina target to file {run[_AD_TP.RUNS_OUTPUT][_AD_TP.RECEPTOR_PATH]}.",_LE.INFO)
            else:
                raise TargetPreparationFailed("Target preparation backend unknown.")
        except Exception as e:
            logger.log(f"Failed when target preparation run number {run_number}.", _LE.EXCEPTION)
            raise TargetPreparationFailed() from e
        else:
            logger.log(f"Completed target preparation run number {run_number}.", _LE.INFO)
    # clean-up
    for temp_file in temp_files:
        os.remove(temp_file)    
    return 

def dockstream_dock(conf,validation=True,silent=False,print_scores=True,print_all=False,debug=False,log_conf=None,output_prefix=None,dockstream_root_path='/data/myxu/Tree_Invent/envs/DockStream/'):
    _LE = LoggingConfigEnum()
    _LP = LigandPreparationEnum()
    _DE = DockingConfigurationEnum()
    if log_conf is None:
        log_conf=dockstream_root_path+_LE.PATH_CONFIG_DEFAULT
    if debug:
        log_conf=dockstream_root_path+_LE.PATH_CONFIG_DEBUG
    try:
        #print (conf)
        config = DockingContainer(conf=conf, validation=validation)
    except Exception as e:
        raise DockingRunFailed() from e
    
    logger=init_dockstream_logger(config=config, task=_DE.DOCKING, _task_enum=_DE,log_conf_path=log_conf)
    set_environment(config=config, task=_DE.DOCKING, _task_enum=_DE, logger=logger)
    try:
        with warnings.catch_warnings(record=True) as w:
            from dockstream.core.Gold.Gold_docker import Gold
            if len(w) > 0:
                logger.log("Could not load CCDC / Gold docker - if another backend is being used, you can safely ignore this warning.", _LE.DEBUG)
    except Exception as e:
        logger.log(f"Could not load CCDC / Gold docker - if another backend is being used, you can safely ignore this warning. The exception message reads: {get_exception_message(e)}", _LE.WARNING)
    dict_pools = {}
    if _LP.LIGAND_PREPARATION in config[_DE.DOCKING].keys():

        # If single element (from GUI), wrap in a list.
        if not isinstance(config[_DE.DOCKING][_LP.LIGAND_PREPARATION][_LP.EMBEDDING_POOLS], list):
            config[_DE.DOCKING][_LP.LIGAND_PREPARATION][_LP.EMBEDDING_POOLS] = [
                config[_DE.DOCKING][_LP.LIGAND_PREPARATION][_LP.EMBEDDING_POOLS]
            ]

        # ligand preparation is to be performed
        for pool_number, pool in enumerate(config[_DE.DOCKING][_LP.LIGAND_PREPARATION][_LP.EMBEDDING_POOLS]):
            logger.log(f"Starting generation of pool {pool[_LP.POOLID]}.", _LE.INFO)
            try:
                prep = embed_ligands(smiles=None,
                                     pool_number=pool_number,
                                     pool=pool,
                                     logger=logger,
                                     ligand_number_start=0)
                #print (prep.get_ligands())
                dict_pools[pool[_LP.POOLID]] = prep.get_ligands()
                #print (dict_pools)
            except Exception as e:
                logger.log(f"Failed in constructing pool {pool[_LP.POOLID]}.", _LE.EXCEPTION)
                logger.log(f"Exception reads: {get_exception_message(e)}.", _LE.EXCEPTION)
                raise LigandPreparationFailed
            else:
                logger.log(f"Completed construction of pool {pool[_LP.POOLID]}.", _LE.INFO)

    # docking: this is the actual docking step; ligands can be provided by the preparation step specified before or
    #          loaded from files
    # ---------
    dict_docking_runs = {}
    if _DE.DOCKING_RUNS in config[_DE.DOCKING].keys():

        # If single element (from GUI), wrap in a list.
        if not isinstance(config[_DE.DOCKING][_DE.DOCKING_RUNS], list):
            config[_DE.DOCKING][_DE.DOCKING_RUNS] = [config[_DE.DOCKING][_DE.DOCKING_RUNS]]

        # execute the docking runs specified
        for docking_run_number, docking_run in enumerate(config[_DE.DOCKING][_DE.DOCKING_RUNS]):
            logger.log(f"Starting docking run {docking_run[_DE.RUN_ID]}.", _LE.INFO)
            try:
                if docking_run[_DE.BACKEND] == _DE.BACKEND_RDOCK:
                    docker = rDock(**docking_run)
                elif docking_run[_DE.BACKEND] == _DE.BACKEND_OPENEYE:
                    docker = OpenEye(**docking_run)
                #elif docking_run[_DE.BACKEND] == _DE.BACKEND_OPENEYEHYBRID:
                #    docker = OpenEyeHybrid(**docking_run)
                elif docking_run[_DE.BACKEND] == _DE.BACKEND_GLIDE:
                    docker = Glide(**docking_run)
                elif docking_run[_DE.BACKEND] == _DE.BACKEND_GOLD:
                    docker = Gold(**docking_run)
                elif docking_run[_DE.BACKEND] == _DE.BACKEND_AUTODOCKVINA:
                    docker = AutodockVina(**docking_run)
                else:
                    raise Exception("Backend is unknown.")

                # merge all specified pools for this run together
                if isinstance(docking_run[_DE.INPUT_POOLS], str):
                    docking_run[_DE.INPUT_POOLS] = [docking_run[_DE.INPUT_POOLS]]
                for pool_id in docking_run[_DE.INPUT_POOLS]:
                    cur_pool = [lig.get_clone() for lig in dict_pools.get(pool_id)]
                    if cur_pool is None or len(cur_pool) == 0:
                        raise Exception("Could not find pool id during docking run or pool was empty.")
                    docker.add_molecules(molecules=cur_pool)

                # do the docking
                docker.dock()
                scores = docker.get_scores(best_only=not print_all)
                
                #print (scores)
                # if specified, save the poses and the scores and print the scores to "stdout"
                handle_poses_writeout(docking_run=docking_run, docker=docker, output_prefix=output_prefix)
                handle_scores_writeout(docking_run=docking_run, docker=docker, output_prefix=output_prefix)
            except Exception as e:
                scores=[]
                logger.log(f"Failed when executing run {docking_run[_DE.RUN_ID]}.", _LE.EXCEPTION)
                logger.log(f"Exception reads: {get_exception_message(e)}.", _LE.EXCEPTION)
                raise DockingRunFailed() from e
            else:
                logger.log(f"Completed docking run {docking_run[_DE.RUN_ID]}.", _LE.INFO)
    return scores

def prepare_target_in_vina_format(input_path,target_pdb_path,reflig_pdb_path,out_path,log_path,dockstream_root_path,bin_path):
    # specify the target preparation JSON file as a dictionary and write it out
    if not os.path.exists(out_path):
        os.system(f'mkdir -p {out_path}')
    fixed_target_pdb_path=target_pdb_path.strip('pdb')+'fix.pdb'
    processed_target_pdbqt_path=target_pdb_path.strip('pdb')+'fix.pdbqt'
    tp_dict = {
        "target_preparation": {
            "header": {
                "logging": {
                    "logfile": f'{out_path}/{log_path}'
                    }
                },
            "input_path": f'{input_path}/{target_pdb_path}',                    # this should be an absolute path
            "fixer": {
                "enabled": True,
                "standardize": True,                                            # enables standardization of residues
                "remove_heterogens": True,                                      # remove hetero-entries
                "fix_missing_heavy_atoms": True,                                # if possible, fix missing heavy atoms
                "fix_missing_hydrogens": True,                                  # add hydrogens, which are usually not present in PDB files
                "fix_missing_loops": False,                                     # add missing loops; CAUTION: the result is usually not sufficient
                "add_water_box": False,                                         # if you want to put the receptor into a box of water molecules
                "fixed_pdb_path": f'{out_path}/{fixed_target_pdb_path}'         # if specified and not "None", the fixed PDB file will be stored here
                },
            "runs": [
                {
                    "backend": "AutoDockVina",                                  # one of the backends supported ("AutoDockVina", "OpenEye", ...)
                    "output": {
                        "receptor_path": f'{out_path}/{processed_target_pdbqt_path}'      # the generated receptor file will be saved to this location
                        },
                    "parameters": {
                        "pH": 7.4,                                              # sets the protonation states (NOT used in Vina)
                        "extract_box": {
                            "reference_ligand_path": f'{input_path}/{reflig_pdb_path}',   # path to the reference ligand
                            "reference_ligand_format": "PDB"                              # format of the reference ligand
                            }
                        }
                }]
            }
        }
    jsonsetting=f'{out_path}/target_prep.json'
    with open(jsonsetting, 'w') as f:
        json.dump(tp_dict, f, indent="    ")
    dockstream_prepare_target(conf=jsonsetting)
    return 
def prepare_docking_in_vina_format(input_path,
                                    center,
                                    box_size,
                                    receptor_pdbqt_path,
                                    lig_smiles_csv_path,
                                    lig_conformers_path,
                                    vina_binary_location,
                                    out_path,
                                    log_path,
                                    pose_path,
                                    score_path,ncores=10,nposes=2):
    if not os.path.exists(out_path):
        os.system(f'mkdir -p {out_path}')
     
    ed_dict = {
                "docking": {
                    "header": {                                         # general settings
                        "logging": {                                      # logging settings (e.g. which file to write to)
                            "logfile": f'{out_path}/{log_path}'
                        }
                    },
                    "ligand_preparation": {                             # the ligand preparation part, defines how to build the pool
                        "embedding_pools": [
                            {
                                "pool_id": "RDkit_pool",                     # here, we only have one pool
                                "type": "RDkit",
                                "parameters": {
                                    "protonate":True,
                                    "prefix_execution": "module load RDkit"    # only required, if a module needs to be loaded to execute "Corina"
                                },
                                "input": {
                                    "standardize_smiles": False,
                                    "type": "smi",
                                    "input_path": f'{lig_smiles_csv_path}'
                                },
                                "output": {                                   # the conformers can be written to a file, but "output" is not required as the ligands are forwarded internally
                                    "conformer_path": f'{lig_conformers_path}', 
                                    "format": "sdf"
                                }
                            }
                        ]
                    },
                    "docking_runs": [
                        {
                            "backend": "AutoDockVina",
                            "run_id": "AutoDockVina",
                            "input_pools": ["RDkit_pool"],
                            "parameters": {
                                "binary_location": vina_binary_location,               # absolute path to the folder, where the "vina" binary can be found
                                "parallelization": {
                                    "number_cores": ncores
                                },
                                "seed": 42,                                            # use this "seed" to generate reproducible results; if varied, slightly different results will be produced
                                "receptor_pdbqt_path": [f'{receptor_pdbqt_path}'],     # paths to the receptor files
                                "number_poses": nposes,                                     # number of poses to be generated
                                "search_space": {                                      # search space (cavity definition); see text
                                    "--center_x": center[0],
                                    "--center_y": center[1],
                                    "--center_z": center[2],
                                    "--size_x": box_size,
                                    "--size_y": box_size,
                                    "--size_z": box_size                                
                                }
                            },
                            "output": {
                                "poses": { "poses_path": f'{out_path}/{pose_path}'},
                                "scores": { "scores_path": f'{out_path}/{score_path}'}
                            }
                        }
                    ]
                }
            }    
    jsonsetting=f'{out_path}/dock.json'
    with open(jsonsetting, 'w') as f:
        json.dump(ed_dict, f, indent="    ")        
    return 

def prepare_docking_in_glide_format(input_path,
                                        grid_path,
                                        smiles_path,
                                        glide_flags={},
                                        glide_keywords={},
                                        ncores=4,
                                        out_path='./glide_dock',
                                        log_path='dock.log',
                                        pose_path='pose.sdf',
                                        score_path='scode.log',
                                        glide_ver='2017'):

    if not os.path.exists(out_path):
        os.system(f'mkdir -p {out_path}')
    glide_input_keywords={
        "AMIDE_MODE": "trans",
        "EXPANDED_SAMPLING": "True",
        "GRIDFILE":f"{input_path}/{grid_path}", #absolute path of grid path
        "NENHANCED_SAMPLING": "2",
        "POSE_OUTTYPE": "ligandlib_sd",
        "POSES_PER_LIG": "3",
        "POSTDOCK_NPOSE": "15",
        "POSTDOCKSTRAIN": "True",
        "PRECISION": "HTVS",
        "REWARD_INTRA_HBONDS": "True"
        }
    glide_input_flags={
        "-HOST":"localhost"
        }
    for key in glide_keywords.keys():
        if key not in glide_input_keywords.keys():
            glide_input_keywords[key]=glide_keywords[key]
    for key in glide_flags.keys():
        if key not in glide_input_flags.keys():
            glide_input_flags[key]=glide_flags[key]

    ed_dict={
        "docking": {
            "header": {                                   # general settings
                "environment": {}
                },
            "ligand_preparation": {                       # the ligand preparation part, defines how to build the pool
                "embedding_pools": [
                    {
                        "pool_id": "Ligprep_pool",
                        "type": "Ligprep",
                        "parameters": {
                            "prefix_execution": f"module load schrodinger/{glide_ver}",
                            "parallelization": {
                                "number_cores": ncores
                            },
                            "use_epik": {
                                "target_pH": 7.0,
                                "pH_tolerance": 2.0
                            },
                            "force_field": "OPLS3e"
                        },
                        "input": {
                            "standardize_smiles": False,
                            "input_path": smiles_path,
                            "type": "smi"                                   # expected input is a text file with smiles
                        },
                        "output": {                                       # the conformers can be written to a file, but "output" is not required as the ligands are forwarded internally
                            "conformer_path": f"{out_path}/ligand.sdf", 
                            "format": "sdf"
                        }
                    }
                    ]
                },
            "docking_runs": [
                    {
                        "backend": "Glide",
                        "run_id": "Glide_run",
                        "input_pools": ["Ligprep_pool"],
                        "parameters": {
                            "prefix_execution": "module load schrodinger/2017", # will be executed before a program call
                            "parallelization": {                              # if present, the number of cores to be used can be specified
                                "number_cores": ncores
                                },
                            "glide_flags": glide_input_flags,
                            "glide_keywords": glide_input_keywords,                               # add all keywords for the "input.in" file here
                            },
                        "output": {
                                "poses" :{ "poses_path":  f"{out_path}/{pose_path}" },
                                "scores":{ "scores_path": f"{out_path}/{score_path}"}
                            }
                    }
                ]
            }
        }
    jsonsetting=f'{out_path}/dock.json'
    with open(jsonsetting, 'w') as f:
        json.dump(ed_dict, f, indent="    ")        
    return 

"""
prepare_target_in_vina_format(  input_path='/data/myxu/Tree_Invent/envs/docktest',
                                target_pdb_path='target.pdb',
                                reflig_pdb_path='reflig.pdb',
                                out_path='/data/myxu/Tree_Invent/envs/docktest/receptor',
                                log_path='target_prep.log')

prepare_docking_in_vina_format(  input_path='/data/myxu/Tree_Invent/envs/docktest',
                                center=[-14.3,-27.47,6.73],
                                box_size=[15,15,15],
                                receptor_pdbqt_path='/data/myxu/Tree_Invent/envs/docktest/receptor/target.fix.pdbqt',
                                lig_smiles_csv_path='ligands.smi',
                                lig_conformers_path='lig_conf.sdf',
                                vina_binary_location='/data/myxu/Tree_Invent/envs/autodock_vina_1_1_2_linux_x86/bin/',
                                out_path='vina_dock',
                                log_path='dock.log',
                                pose_path='pose.sdf',
                                score_path='score.log')

scores=dockstream_dock(conf='/data/myxu/Tree_Invent/envs/docktest/dock.json') 
print (scores)
"""
