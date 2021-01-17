import tvm
import os
import logging
import sys
import numpy as np

### IMPORT NECESSARY TVM LIBS
from tvm import relay 
import tvm
from tvm import te
from tvm.contrib import graph_runtime
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm import autotvm
import tvm.contrib.graph_runtime as runtime
from tvm.relay import testing
from tvm.contrib.debugger import debug_runtime 
from tvm.runtime import container
from tvm.runtime import vm as vm_rt
from tvm.relay import vm
from tvm import topi


from tvm import autotvm




def tune_tasks(tasks,
            measure_option,
            tuner='xgb',
            n_trial=1000,
            early_stopping=None,
            log_filename='tuning.log',
            use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    
    
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)
        
        
        if os.path.isfile(tmp_log_file):
            tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
                
        
        print("current tuning: ")
        print(tsk)
        
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                        early_stopping=early_stopping,
                        measure_option=measure_option,
                        callbacks=[
                            autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                            autotvm.callback.log_to_file(tmp_log_file)
                        ])
    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
