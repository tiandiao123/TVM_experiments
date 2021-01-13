import os
import logging
import sys
import tvm
from tvm import te
from tvm import topi
from tvm.topi.testing import conv2d_nchw_python
from tvm import autotvm
from tune_func import tune_tasks
import os
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python



target = tvm.target.cuda(model="T4")
tvm.autotvm.measure.measure_methods.set_cuda_target_arch("sm_75")
log_file = "ansor_conv2d_0_99.json"
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
ctx = tvm.context(str(target), 0)


@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]


batches = [2, 4, 8]
in_channels = [32, 64]
in_heights = [64, 128, 256]
filter_size_list = [3, 5]

num_filer_map = {}
for in_channel in in_channels:
    num_filer_map[in_channel] = in_channel*2


create_tasks = []
for batch_size in batches:
    for in_channel in in_channels:
        for in_height in in_heights:
            in_width = in_height
            data_shape = (batch_size, in_channel, in_height, in_width)
            data = te.placeholder(data_shape, name = "data")
            out_channel = 2 * in_channel
            for filter_size in filter_size_list:
                N, H, W, CO, CI, KH, KW, strides, padding = batch_size, in_height, in_width, out_channel, in_channel, filter_size, filter_size, (1, 1), (1, 1)
                task = auto_scheduler.SearchTask(
                    func=conv2d_layer, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
                )
                create_tasks.append(task)
                # kernel_shape = (num_filter, in_channel, filter_size, filter_size)
                # kernel = te.placeholder(kernel_shape, name="kernel")
                # candidates.append((data, kernel))
            
for task in create_tasks:
    print("---------------")
    print(task.compute_dag)


measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

print("we have {} tasks to tune".format(str(len(create_tasks))))
index = 1
for task in create_tasks:
    index += 1
    print("current tuning task {} .................".format(index))
    task.tune(tune_option)















