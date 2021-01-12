import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
import os
import numpy as np
import time
from tvm.runtime import vm as vm_rt
from tvm.relay import vm
import argparse

target = tvm.target.cuda()
ctx = tvm.context(str(target), 0)


dshape = (1, 128, 128)
kshape = (1, 256, 128)
dtype = "float32"
x = relay.var("x", shape=dshape, dtype=dtype)
w = relay.var("w", shape=kshape, dtype=dtype)
#y = relay.nn.conv2d(x, w, strides=(1,1), padding=(1,1,1,1), dilation=(1,1), channels=256, kernel_size=(3,3))
#y = relay.nn.conv2d(x, w, padding=padding, dilation=dilation, groups=groups)
y = relay.nn.batch_matmul(x, w)
func = relay.Function([x, w], y)
mod = tvm.IRModule()
mod["main"] = func

#
print("print non-tuning tvm op: ")
scale = 1
data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
weight = np.random.uniform(-scale, scale, size=kshape).astype(dtype)
data = tvm.nd.array(data, ctx)
weight = tvm.nd.array(weight, ctx)



with tvm.transform.PassContext(opt_level=3):
    print("Compiling...")
    graph, lib, params = tvm.relay.build(mod, target=target)


from tvm.contrib.debugger import debug_runtime as graph_runtime
module  = graph_runtime.create(graph, lib, ctx)
module.set_input("x", data)
module.set_input("w", weight)
module.set_input(**params)
print("testing non-tuning result of the op......")
ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
        (np.mean(prof_res), np.std(prof_res)))


print("\n")
print("now testing tuned op......")
with autotvm.apply_history_best("batch_matmul_cuda.log"):
    with tvm.transform.PassContext(opt_level=3):
        print("Compiling...")
        graph, lib, params = tvm.relay.build(mod, target=target)

module  = graph_runtime.create(graph, lib, ctx)
module.set_input("x", data)
module.set_input("w", weight)
module.set_input(**params)
print("testing tuning result of the op......")
ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
        (np.mean(prof_res), np.std(prof_res)))





print("\n")
print("now testing tuned op......")
with autotvm.apply_history_best("batch_matmul_tensorcore_cuda.log"):
    with tvm.transform.PassContext(opt_level=3):
        print("Compiling...")
        graph, lib, params = tvm.relay.build(mod, target=target)

module  = graph_runtime.create(graph, lib, ctx)
module.set_input("x", data)
module.set_input("w", weight)
module.set_input(**params)
print("testing tuning result of the op using tensorcore......")
ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
        (np.mean(prof_res), np.std(prof_res)))

print("finished programming")





