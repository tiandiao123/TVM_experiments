import os
import pickle
import json
import logging
import csv
import logging
import sys
import tvm
from tvm import te
from tvm import topi
from tvm.topi.testing import conv2d_nchw_python
from tvm import autotvm
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
from multiprocessing import Pool



log_file = "./log_dir/conv2d_0_99_cuda_cudnn.log"
log_list = []
with open(log_file, "rb") as f:
    for line in f:
        log_list.append(line)



print("length of log_list: {}".format(len(log_list)))
cuda_cudnn_dict_list = []
for line in log_list:
    line = line.decode('ascii')
    log_task_dict = json.loads(str(line))
    cuda_cudnn_dict_list.append(log_task_dict)



cuda_dict_list = []
log_file = "./log_dir/conv2d_0_99.log"
log_list = []
with open(log_file, "rb") as f:
    for line in f:
        log_list.append(line)

print("new list:")
for line in log_list:
    line = line.decode('ascii')
    log_task_dict = json.loads(str(line))
    cuda_dict_list.append(log_task_dict)
print("length of log_list: {}".format(len(cuda_dict_list)))


def match_func(dict_list, input):
    for ele in dict_list:
        if input == ele['input'][2]:
            return ele["result"]
    return None


final_info_list = []
for cuda_cudnn_dict_ele in cuda_cudnn_dict_list:
    input = cuda_cudnn_dict_ele["input"][2]
    res = cuda_cudnn_dict_ele["result"]
    # print(input)
    # print(res)

    second_res = match_func(cuda_dict_list, input)
    if second_res != None:
        print(input)
        print(res)
        print(second_res)
        final_info_list.append([input, res, second_res])
        print("**********************************")

print("final_info_list length is : " + str(len(final_info_list)))



with open('stats.csv', mode='w') as w_file:
    w_writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    w_writer.writerow(['input_info', 'cuda T4', 'cuda T4 --libs==cudnn'])
    for line in final_info_list:
        w_writer.writerow(line)







