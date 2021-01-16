import os
import logging
import sys
import tvm
from tvm import te
from tvm import topi, relay
from tvm.topi.testing import conv2d_nchw_python
from tvm import autotvm
import os
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
from tvm.contrib.debugger import debug_runtime as graph_runtime


#target = "cuda -libs=cudnn"
target = tvm.target.cuda(model="T4")

tvm.autotvm.measure.measure_methods.set_cuda_target_arch("sm_75")
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
ctx = tvm.context(str(target), 0)



dshape = (8, 64, 128, 128)
kshape = (128, 64, 5, 5)
dtype = "float32"
padding = (1,1)
dilation = (1,1)
groups = 1
x = relay.var("x", shape=dshape, dtype=dtype)
w = relay.var("w", shape=kshape, dtype=dtype)
y = relay.nn.conv2d(x, w, strides=(1,1), padding=(1,1,1,1), dilation=(1,1), channels=128, kernel_size=(5,5))
#y = relay.nn.conv2d(x, w, padding=padding, dilation=dilation, groups=groups)
func = relay.Function([x, w], y)
mod = tvm.IRModule()
mod["main"] = func


print("print non-tuning tvm op: ")
scale = 1
data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype)
data = tvm.nd.array(data, ctx)
kernel = tvm.nd.array(kernel, ctx)



# with tvm.transform.PassContext(opt_level=3):
#     print("Compiling...")
#     graph, lib, params = tvm.relay.build(mod, target=target)


# from tvm.contrib.debugger import debug_runtime as graph_runtime
# module  = graph_runtime.create(graph, lib, ctx)
# module.set_input("x", data)
# module.set_input("w", kernel)
# module.set_input(**params)
# print("testing non-tuning result of the op......")
# ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
# prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
# print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
#         (np.mean(prof_res), np.std(prof_res)))



print("\n")
print("now testing tuned op......")
with autotvm.apply_history_best("/home/tiger/cuiqing.li/tiandiao123_tvm_proj/log_dir/conv2d_0_99.log"):
    with tvm.transform.PassContext(opt_level=3):
        print("Compiling...")
        graph, lib, params = tvm.relay.build(mod, target=target)

module  = graph_runtime.create(graph, lib, ctx)
module.set_input("x", data)
module.set_input("w", kernel)
module.set_input(**params)
print("testing tuning result of the op......")
ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
        (np.mean(prof_res), np.std(prof_res)))

print("finished programming")


