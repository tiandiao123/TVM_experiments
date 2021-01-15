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
from tvm.runtime import vm as _vm
from tvm.relay import vm




def main():
    params_dict = config
    saved_dir = params_dict['saved_dir']

    #target = "cuda -libs=cudnn,cublas"
    target = params_dict['target']
    ctx = tvm.context(str(target), 0)
    path_lib = os.path.join(saved_dir, "lib.so")
    code_path = os.path.join(saved_dir, "code.ro")

    loaded_lib = tvm.runtime.load_module(path_lib)
    loaded_code = bytearray(open(code_path, "rb").read())

    # deserialize.
    des_exec = _vm.Executable.load_exec(loaded_code, loaded_lib)
    des_vm = _vm.VirtualMachine(des_exec, ctx)
    data = []
    dtype = params_dict['dtype']
    for input_shape in params_dict['inference_input_shapes']:
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        data.append(data_tvm)
    data = tuple(data)

    res = des_vm.run(*data)
    print("Evaluate vm inference cost of {} on {}".format("your testing model", repr(ctx)))
    ftimer_warmup = des_vm.module.time_evaluator("invoke", ctx, number=1,
                                        repeat=50)
                                        # Measure in millisecond.
    print("finished warming up and start testing vm compile performance")
    ftimer = des_vm.module.time_evaluator("invoke", ctx, number=1,
                                        repeat=600)
                                        # Measure in millisecond.
    prof_res = np.array(ftimer("main", *data).results) * 1000
    #prof_res = np.array(ftimer().results) * 1000 
    print("Mean vm inference time (std dev): %.2f ms (%.2f ms)" %
        (np.mean(prof_res), np.std(prof_res)))


def test_dynamic_bcast():
    import pytest
    import numpy as np

    import tvm
    from tvm.runtime import vm as _vm
    from tvm.relay import vm as rly_vm
    from tvm import relay

    from tvm.relay.scope_builder import ScopeBuilder
    from tvm.relay import transform
    from tvm.relay.prelude import Prelude
    from tvm.relay import testing
    def create_exec(f, target="llvm", params=None):
        if isinstance(f, relay.Expr):
            mod = tvm.IRModule()
            mod["main"] = f
            executable = rly_vm.compile(mod, target=target, params=params)
            return executable
        else:
            assert isinstance(f, tvm.IRModule), "expected mod as tvm.IRModule"
            executable = rly_vm.compile(f, target=target, params=params)
            return executable


    def get_serialized_output(mod, *data, params=None, target="llvm", ctx=tvm.cpu()):
        exe = create_exec(mod, target, params=params)
        code, lib = exe.save()
        des_exec = _vm.Executable.load_exec(code, lib)
        des_vm = _vm.VirtualMachine(des_exec, ctx)
        result = des_vm.run(*data)
        print(result)
        return result


    dtype = "float32"
    x = relay.var("x", shape=(1, 2), dtype=dtype)
    y = relay.var("y", shape=(relay.Any(), 2), dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], relay.add(x, y))
    x_data = np.random.uniform(size=(1, 2)).astype(dtype)
    y_data = np.random.uniform(size=(4, 2)).astype(dtype)
    res_np = np.add(x_data, y_data)
    for target, ctx in testing.enabled_targets():
        res = get_serialized_output(mod, *(x_data, y_data), target=target, ctx=ctx)
        tvm.testing.assert_allclose(res.asnumpy(), res_np)


def test_vm_onnx_process():
    import onnx
    onnx_model_path = "/data00/cuiqing.li/onnx_models/sr_dy.onnx"
    onnx_model = onnx.load(onnx_model_path)
    shape_dict = {"input.1":(1,1,640,360)}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    target = tvm.target.cuda()
    ctx = tvm.context(str(target), 0)
    
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
        exe = vm.compile(mod, target, params=params)
        code, lib = exe.save()
        saved_dir = "tmp"
        if os.path.isdir("./tmp") == False:
            os.system("mkdir {}".format(saved_dir))

        path_lib = os.path.join(saved_dir, "lib.so")
        lib.export_library(path_lib)

        code_path = os.path.join(saved_dir, "code.ro")
        with open(code_path, "wb") as fo:
            fo.write(code)
        


        loaded_lib = tvm.runtime.load_module(path_lib)
        loaded_code = bytearray(open(code_path, "rb").read())

        # deserialize.
        des_exec = _vm.Executable.load_exec(loaded_code, loaded_lib)
        des_vm = _vm.VirtualMachine(des_exec, ctx)


        input_shape = [1,1,640,360]
        dtype = "float32"
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        data = []
        data.append(data_tvm)
        data = tuple(data)
        res = des_vm.run(*data)

        print("Evaluate vm inference cost of {} on {}".format("your testing model", repr(ctx)))
        ftimer_warmup = des_vm.module.time_evaluator("invoke", ctx, number=1,
                                            repeat=50)
                                            # Measure in millisecond.
        print("finished warming up and start testing vm compile performance")
        ftimer = des_vm.module.time_evaluator("invoke", ctx, number=1,
                                            repeat=600)
                                            # Measure in millisecond.
        prof_res = np.array(ftimer("main", *data).results) * 1000
        #prof_res = np.array(ftimer().results) * 1000 
        print("Mean vm inference time (std dev): %.2f ms (%.2f ms)" %
            (np.mean(prof_res), np.std(prof_res)))



def vm_tensorflow_model_process():
    def normalize_node_name(nodes):
        from tensorflow.compat import as_text
        if isinstance(nodes, list):
            ret = [as_text(node.split(':', 1)[0], 'ascii') for node in nodes]
        else:
            ret = as_text(nodes.split(':', 1)[0], 'ascii')
            
        return ret

    import tensorflow as tf
    from tvm.relay.frontend.tensorflow_parser import TFParser
    TF_pb_path = "/home/tiger/cuiqing.li/models/TF_checkpoint/latest"
    graph_def = TFParser(TF_pb_path).parse()
    input_names = ["input_ids_1:0", "input_mask_1:0", "segment_ids_1:0"]
    output_names = ["loss/Softmax:0"]
    input_shapes = [[1,256],[1,256],[1,256]]


    input_names = [normalize_node_name(i) for i in input_names]
    output_names = [normalize_node_name(i) for i in output_names]
    mod, params = relay.frontend.from_tensorflow(graph_def,
                                        shape={k: v for k, v in zip(input_names, input_shapes)},
                                        layout=None, outputs=output_names)
                                            
    desired_layouts = {'nn.conv2d': ['NCHW', 'default']}
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                relay.transform.ConvertLayout(desired_layouts)])
    with tvm.ir.transform.PassContext(opt_level=3):
        mod = seq(mod)
    

    target = tvm.target.cuda()
    ctx = tvm.context(str(target), 0)
    
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
        exe = vm.compile(mod, target, params=params)
        code, lib = exe.save()
        saved_dir = "tmp"
        if os.path.isdir("./tmp") == False:
            os.system("mkdir {}".format(saved_dir))

        path_lib = os.path.join(saved_dir, "lib.so")
        lib.export_library(path_lib)

        code_path = os.path.join(saved_dir, "code.ro")
        with open(code_path, "wb") as fo:
            fo.write(code)
        


        loaded_lib = tvm.runtime.load_module(path_lib)
        loaded_code = bytearray(open(code_path, "rb").read())

        # deserialize.
        des_exec = _vm.Executable.load_exec(loaded_code, loaded_lib)
        des_vm = _vm.VirtualMachine(des_exec, ctx)



        data = []
        idx = 0
        for input_shape in input_shapes:
            dtype = "int32"
            data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype), ctx)
            data.append(data_tvm)
            idx += 1
        data = tuple(data)
        res = des_vm.run(*data)

        print("Evaluate vm inference cost of {} on {}".format("your testing model", repr(ctx)))
        ftimer_warmup = des_vm.module.time_evaluator("invoke", ctx, number=1,
                                            repeat=50)
                                            # Measure in millisecond.
        print("finished warming up and start testing vm compile performance")
        ftimer = des_vm.module.time_evaluator("invoke", ctx, number=1,
                                            repeat=100)
                                            # Measure in millisecond.
        prof_res = np.array(ftimer("main", *data).results) * 1000
        #prof_res = np.array(ftimer().results) * 1000 
        print("Mean vm inference time (std dev): %.2f ms (%.2f ms)" %
            (np.mean(prof_res), np.std(prof_res)))








if __name__ == "__main__":
    #main()

    #test_vm_onnx_process()
    #test_dynamic_bcast()
    vm_tensorflow_model_process()

