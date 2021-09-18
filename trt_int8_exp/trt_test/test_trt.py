import tvm
from tvm import relay
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata

import os
import onnx
import numpy as np
from PIL import Image
import onnxruntime

# Tensorflow imports
import tvm.relay.testing.tf as tf_testing


def compare_tvm_torch_output(tvm_res, torch_res):
    tvm_res = tvm_res.flatten()
    torch_res = torch_res.flatten()
    return np.max(np.abs(tvm_res-torch_res))


def cosine_distance(matrix1 , matrix2):
    matrix1 = matrix1.flatten()
    matrix2 = matrix2.flatten()
    res = distance.cosine(matrix1, matrix2)
    return res


model_url = "".join(
    [
        "https://gist.github.com/zhreshold/",
        "bcda4716699ac97ea44f791c24310193/raw/",
        "93672b029103648953c4e5ad3ac3aadf346a4cdc/",
        "super_resolution_0.2.onnx",
    ]
)
model_path = download_testdata(model_url, "super_resolution.onnx", module="onnx")
# now you have super_resolution.onnx on disk
onnx_model = onnx.load(model_path)


dtype = "float32"
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
img_y, img_cb, img_cr = img_ycbcr.split()
x = np.array(img_y)[np.newaxis, np.newaxis, :, :].astype("float32")
input_name = "1"
shape_dict = {input_name: x.shape}
print("shape: ")
print(x.shape)
input_shape = x.shape
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# compile the model
target = "cuda"
dev = tvm.cuda(0)


from tvm.relay.op.contrib.tensorrt import partition_for_tensorrt
mod, config = partition_for_tensorrt(mod, params)

print("python script building --------------")
with tvm.transform.PassContext(opt_level=3, config={'relay.ext.tensorrt.options': config}):
    lib = relay.build(mod, target=target, params=params)
print("python script finsihed building -------------------")


dtype = "float32"
lib.export_library('compiled.so')
loaded_lib = tvm.runtime.load_module('compiled.so')
gen_module = tvm.contrib.graph_executor.GraphModule(loaded_lib['default'](dev))

num_cali_int8 = 0
try:
    num_cali_int8 = os.environ["TENSORRT_NUM_CALI_INT8"]
    print("we are going to set {} times calibration in this case".format(num_cali_int8))
except:
    print("no TENSORRT_NUM_CALI_INT8 found in this case ... ")

num_cali_int8 = int(num_cali_int8)
if num_cali_int8 != 0:
    print("started calibration steps ... ")
    for i in range(num_cali_int8):
        tvm_data = tvm.nd.array(x)
        gen_module.set_input(input_name, tvm_data)
        gen_module.run(data=tvm_data)
    print("finished calibration steps ... ")

print("create builder and testing to run ... ")
tvm_data = tvm.nd.array(x)
gen_module.set_input(input_name, tvm_data)
gen_module.run(data=x)
out = gen_module.get_output(0)
print(out)

# get output of onnx output
print("starting using onnx model and get output ... ")


# Evaluate
print("Evaluate inference time cost...")
ftimer = gen_module.module.time_evaluator("run", dev, repeat=10, min_repeat_ms=500)
prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
message = "Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
print(message)