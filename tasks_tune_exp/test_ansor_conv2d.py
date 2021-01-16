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
from multiprocessing import Pool

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


def resume_search(task, log_file):
    print("Resume search:")
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    task.tune(tune_option, search_policy=search_policy)

    # Kill the measurement process
    del measure_ctx



def evaluate_func(sch, args, N, H, W, CO, CI, KH, KW):
    func = tvm.build(sch, args, target)

    # Check correctness
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)
    out_np = conv_np

    ctx = tvm.gpu()
    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    weight_tvm = tvm.nd.array(weight_np, ctx=ctx)
    out_tvm = tvm.nd.empty(out_np.shape, ctx=ctx)
    func(data_tvm, weight_tvm, out_tvm)

    # Check results
    np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )




batches = [2, 4, 8]
in_channels = [32, 64]
in_heights = [64, 128, 256]
filter_size_list = [3, 5]

num_filer_map = {}
for in_channel in in_channels:
    num_filer_map[in_channel] = in_channel*2


create_tasks = []
input_info_list = []
for batch_size in batches:
    for in_channel in in_channels:
        for in_height in in_heights:
            in_width = in_height
            data_shape = (batch_size, in_channel, in_height, in_width)
            data = te.placeholder(data_shape, name = "data")
            out_channel = 2 * in_channel
            for filter_size in filter_size_list:
                kernel_shape = (out_channel, in_channel, filter_size, filter_size)
                input_info_list.append((data_shape, kernel_shape))
                N, H, W, CO, CI, KH, KW, strides, padding = batch_size, in_height, in_width, out_channel, in_channel, filter_size, filter_size, (1, 1), (1, 1)
                task = auto_scheduler.SearchTask(
                    func=conv2d_layer, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
                )
                create_tasks.append(task)

            
for task in create_tasks:
    print("---------------")
    print(task.compute_dag)


print("we have {} tasks to tune".format(str(len(create_tasks))))
index = 0
for task in create_tasks[:5]:
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    data_shape, kernel_shape = input_info_list[index][0], input_info_list[index][1]
    batch_size, in_channel, in_height, in_width = data_shape
    out_channel, in_channel, filter_size, filter_size = kernel_shape
    N, H, W, CO, CI, KH, KW, strides, padding = batch_size, in_height, in_width, out_channel, in_channel, filter_size, filter_size, (1, 1), (1, 1)
    evaluate_func(sch, args, N, H, W, CO, CI, KH, KW)
    index += 1