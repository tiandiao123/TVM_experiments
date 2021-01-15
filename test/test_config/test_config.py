import os


config = {

    # model saved directory
    "saved_dir": "/data00/cuiqing.li/ByteTuner/saved_tvm_model",

    # type input names and input shapes in this case
    
    "inference_input_shapes":[[1,1,640,360]],
    "inference_input_names": ["input.1"],
    
    # set data type of your input:
    "dtype": "float32",

    #set your target: "cuda -libs=cudnn,cublas", "cuda", "cuda -libs=cudnn" or "cuda -libs=cublas"
    "target": "cuda",
}
