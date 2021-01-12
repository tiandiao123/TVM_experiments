import os
import logging
import sys
import numpy as np

import tvm
from tvm import te
from tvm import topi
from tvm.topi.testing import conv2d_nchw_python

from tvm import autotvm
from tune_func import tune_tasks



target = tvm.target.cuda()
log_file = "batch_matmul_tensorcore_cuda.log"

ctx = tvm.context(str(target), 0)
tuning_option = {
    'log_filename': log_file,

    'tuner': 'gridsearch',
    'n_trial': 1500,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        #runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        runner=autotvm.RPCRunner(
            "arnold",  # change the device key to your key
            "0.0.0.0", 
            9190,
            number=20, repeat=3, timeout=4, min_repeat_ms=150)
    ),
}
# tuning_option = {
#     'log_filename': log_file,

#     'tuner': 'gridsearch',
#     'n_trial': 2000,
#     'early_stopping': 600,

#     'measure_option': autotvm.measure_option(
#         builder=autotvm.LocalBuilder(timeout=10),
#         runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
#     ),
# }



# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))


batch_size = 1
M_List = [32, 16, 32, 64, 128]
K_List = [32, 32, 64, 128, 128]
N_List = [32, 32, 64, 128, 256]


candidates = []
tasks = []

for i in range(len(M_List)):
    M = M_List[i]
    K = K_List[i]
    N = N_List[i]
    data = te.placeholder((batch_size, M, K), name="data")
    weight = te.placeholder((batch_size, N, K), name = "weight")
    candidates.append((data, weight))
    task = autotvm.task.create("batch_matmul_tensorcore.cuda",
                                args=(data, weight),
                                target=target)
    tasks.append(task)


print("we have {} tasks to tune.......".format(len(tasks)))
for task in tasks:
    print(task)



tune_tasks(tasks, **tuning_option)
