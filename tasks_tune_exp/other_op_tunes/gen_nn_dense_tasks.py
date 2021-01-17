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
log_file = "nn_dense.log"

ctx = tvm.context(str(target), 0)
tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 1000,
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



# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))


small_batches = [1, 2, 4, 8, 16]
large_batches = [32, 64, 128]
in_dims = [256, 512, 128, 64]
out_dims = [10, 64, 128, 256]

candidates = []
for batch in small_batches:
    for in_dim in in_dims:
        data = te.placeholder((batch, in_dim), name="data")
        for out_dim in out_dims:
            weight = te.placeholder((out_dim, in_dim), name="weight")
            candidates.append((data, weight))

  



tasks = []
print("create tasks.............")
for candidate in candidates:
    data, weight = candidate[0], candidate[1]
    if data.shape[0] <= 16:
        task = autotvm.task.create("dense_small_batch.cuda",
                                    args=(data, weight, None, "float32"),
                                    target=target)
    else:
        task = autotvm.task.create("dense_large_batch.cuda", args = (data, weight, None, "float32"), 
                                        target=target)

    tasks.append(task)
    print(task)


print("OMG!!! we have so many tasks to tune: " + str(len(tasks)))


# data = te.placeholder((32, 64, 32, 32), name = "data")
# kernel = te.placeholder((32, 64, 3, 3), name = "kernel")
# task = autotvm.task.create("conv2d_nchw.cuda",
#                             args=(data, kernel, 1, (1,1), 1, "float32"),
#                             target=target)

# print(task)


#tasks = tasks[0:100]

tune_tasks(tasks, **tuning_option)





