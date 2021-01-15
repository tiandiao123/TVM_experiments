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



#target = tvm.target.cuda(model="T4")
target = "cuda -libs=cudnn"
tvm.autotvm.measure.measure_methods.set_cuda_target_arch("sm_75")
log_file = "conv2d_0_99_cuda_cudnn.log"
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"

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



batches = [2, 4, 8]
in_channels = [32, 64]
in_heights = [64, 128, 256]
filter_size_list = [3, 5]

num_filer_map = {}
for in_channel in in_channels:
    num_filer_map[in_channel] = [in_channel*2]


candidates = []
for batch_size in batches:
    for in_channel in in_channels:
        for in_height in in_heights:
            in_width = in_height
            data_shape = (batch_size, in_channel, in_height, in_width)
            data = te.placeholder(data_shape, name = "data")
            num_filters = num_filer_map[in_channel]
            for num_filter in num_filters:

                for filter_size in filter_size_list:
                    kernel_shape = (num_filter, in_channel, filter_size, filter_size)
                    kernel = te.placeholder(kernel_shape, name="kernel")
                    candidates.append((data, kernel))



tasks = []
print("create tasks.............")
for candidate in candidates:
    data, kernel = candidate[0], candidate[1]
    print(data)
    print(kernel)
    task = autotvm.task.create("conv2d_nchw.cuda",args=(data, kernel, 1, (1,1), 1, "float32"),target=target)

    tasks.append(task)
    print(task)


print("OMG!!! we have so many tasks to tune: " + str(len(tasks)))


# tasks = tasks[0:100]

tune_tasks(tasks, **tuning_option)






