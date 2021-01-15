import os
import argparse
import numpy as np
import os.path
from test_config.test_config import config
import numpy as np
from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
# from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime as graph_runtime




params_dict = config



saved_dir = params_dict['saved_dir']

deploy_graph_path = os.path.join(saved_dir, "deploy_graph.json")
loaded_graph = open(deploy_graph_path).read()

lib_save_path = os.path.join(saved_dir, "deploy_lib.tar")
loaded_lib = tvm.runtime.load_module(lib_save_path)

deploy_param_file_path = os.path.join(saved_dir, "deploy_param.params")
loaded_params = bytearray(open(deploy_param_file_path, "rb").read())


# target = tvm.target.cuda()
target = params_dict['target']
ctx = tvm.context(str(target), 0)





print("testing and evaluating TVM performance")
dtype = params_dict['dtype']
module  = graph_runtime.create(loaded_graph, loaded_lib, ctx)
print(dtype)

# set inputs
inference_input_shapes = params_dict['inference_input_shapes']
inference_input_names = params_dict['inference_input_names']
for i in range(len(inference_input_shapes)):
	input_shape = tuple(inference_input_shapes[i])
	input_name = inference_input_names[i]
	print("input name:" + input_name)
	print("input shape: " + str(input_shape))
	temp_data = np.random.uniform(size=input_shape).astype(dtype)

	data_tvm = tvm.nd.array(temp_data)
	module.set_input(input_name, data_tvm)

print("loading params......")
#module.set_input(**loaded_params)
module.load_params(loaded_params)
# execute
module.run()


print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
		(np.mean(prof_res), np.std(prof_res)))



