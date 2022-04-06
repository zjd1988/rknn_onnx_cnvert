# -*- coding: utf-8 -*-
import onnx
import time
import numpy as np
import onnxruntime as rt
from onnx import checker
from onnx import helper
from onnx import TensorProto

def conver_rknn_tensor_type_to_onnx_type(tensor_dtype):
    datatype = tensor_dtype.split("_")[-1]
    if datatype == "INT8":
        return TensorProto.INT8
    elif datatype == "UINT8":
        return TensorProto.UINT8
    elif datatype == "INT16":
        return TensorProto.INT16
    elif datatype == "UINT16":
        return TensorProto.UINT16
    elif datatype == "INT32":
        return TensorProto.INT32
    elif datatype == "UINT32":
        return TensorProto.UINT32
    elif datatype == "FLOAT16":
        return TensorProto.FLOAT16
    elif datatype == "FLOAT32":
        return TensorProto.FLOAT
    elif datatype == "BOOL8":
        return TensorProto.BOOL
    else:
        assert False, "unspoorted datatype {}, when convert rknn tensor type to onnx type".format(datatype)


def conver_rknn_tensor_type_to_numpy_type(tensor_dtype):
    datatype = tensor_dtype.split("_")[-1]
    if datatype == "INT8":
        return np.int8
    elif datatype == "UINT8":
        return np.uint8
    elif datatype == "INT16":
        return np.int16
    elif datatype == "UINT16":
        return np.uint16
    elif datatype == "INT32":
        return np.int32
    elif datatype == "UINT32":
        return np.uint32
    elif datatype == "FLOAT16":
        return np.float16
    elif datatype == "FLOAT32":
        return np.float32
    elif datatype == "BOOL8":
        return np.bool
    else:
        assert False, "unspoorted datatype {}, when convert rknn tensor type to np type".format(datatype)


def convert_onnxruntime_type_to_numpy_dtype(onnxruntime_type):
    if onnxruntime_type == "tensor(int8)":
        return np.int8
    elif onnxruntime_type == "tensor(uint8)":
        return np.uint8
    elif onnxruntime_type == "tensor(int16)":
        return np.int16
    elif onnxruntime_type == "tensor(uint16)":
        return np.uint16
    elif onnxruntime_type == "tensor(int32)":
        return np.int32
    elif onnxruntime_type == "tensor(uint32)":
        return np.uint32
    elif onnxruntime_type == "tensor(float16)":
        return np.float16
    elif onnxruntime_type == "tensor(float)":
        return np.float32
    elif onnxruntime_type == "tensor(bool)":
        return np.bool
    else:
        assert False, "unspoorted datatype {}, when convert rknn tensor type to np type".format(onnxruntime_type)

def construct_pool_attr(op_attr):
    node_attr = {}
    for key in op_attr.keys():
        if key == "round_type":
            round_type = op_attr[key]
            if round_type == "VSI_NN_ROUND_CEIL":
                node_attr["ceil_mode"] = 1
            elif round_type == "VSI_NN_ROUND_FLOOR":
                node_attr["ceil_mode"] = 0
            else:
                assert False, "unsupported pool round type {}".format(round_type)
        elif key == "type":
            continue
        elif key == "ksize":
            node_attr["kernel_shape"] = op_attr[key]
        elif key == "stride":
            node_attr["strides"] = op_attr[key]
        elif key == "pad":
            node_attr["pads"] = op_attr[key]
        else:
            assert False, "unsupported pool attribute {}".format(key)
    return node_attr


def construct_node_attr(node_info):
    node_attr = {}
    if node_info["type"] == "POOL":
        op_attr = node_info["attribute"]["pool"]
        node_attr = construct_pool_attr(op_attr)
    
    return node_attr


def get_rknn_tensor_info(rknn_model_info, tensor_name):
    tensors_info = rknn_model_info["tensors"]
    for key in tensors_info.keys():
        if key == tensor_name:
            return tensors_info[key]
    assert False, "rknn model have no tensor: {}".format(tensor_name)


def insert_dequant_node(quant_input_tensor_info, onnx_model_info, log_flag):
    tensor_name = quant_input_tensor_info["name"]
    node_name = tensor_name + "_dequant"
    dequant_out_name = tensor_name + "_dequant_out"
    inserted = False
    for node_index in range(len(onnx_model_info["nodes"])):
        node = onnx_model_info["nodes"][node_index]
        if node.name == node_name:
            inserted = True
            break
    if not inserted:
        op_inputs = []
        op_inputs.append(tensor_name)
        op_inputs.append(tensor_name+"_scale")
        op_inputs.append(tensor_name+"_zero_point")
        op_outputs = []
        op_outputs.append(dequant_out_name)
        node = helper.make_node("DequantizeLinear", inputs=op_inputs, outputs=op_outputs, name=node_name)
        onnx_model_info["nodes"].append(node)
        if log_flag:
            print("insert dequant node for {}".format(tensor_name))
    return dequant_out_name


def insert_quant_node(dequant_out_tensor_info, quant_input_name, onnx_model_info, log_flag):
    tensor_name = dequant_out_tensor_info["name"]
    node_name = tensor_name + "_quant"
    op_inputs = []
    op_inputs.append(quant_input_name)
    op_inputs.append(tensor_name+"_scale")
    op_inputs.append(tensor_name+"_zero_point")
    op_outputs = []
    op_outputs.append(tensor_name)
    node = helper.make_node("QuantizeLinear", inputs=op_inputs, outputs=op_outputs, name=node_name)
    onnx_model_info["nodes"].append(node)
    if log_flag:
        print("insert dequant node for {}".format(tensor_name))


def construct_with_dequant_quant_node(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag):
    node_info = rknn_model_info["nodes"][node_index]
    input_names = node_info["inputs"]
    output_names = node_info["outputs"]
    op_inputs = []
    for index in range(len(input_names)):
        op_inputs.append(input_names[index])
    op_outputs = []
    for index in range(len(output_names)):
        op_outputs.append(output_names[index])
    
    for index in range(len(op_inputs)):
        input_tensor_info = get_rknn_tensor_info(rknn_model_info, op_inputs[index])
        input_dtype = conver_rknn_tensor_type_to_onnx_type(input_tensor_info["dtype"]["vx_type"])
        if input_dtype == TensorProto.UINT8 or \
            input_dtype == TensorProto.INT8:
            quant_input_tensor_info = {}
            quant_input_tensor_info["name"] = op_inputs[index]
            quant_input_tensor_info["dtype"] = input_tensor_info["dtype"]
            dequant_out_name = insert_dequant_node(quant_input_tensor_info, onnx_model_info, log_flag)
            op_inputs[index] = dequant_out_name
    
    need_quant = []
    quant_inputs = []
    for index in range(len(op_outputs)):
        output_tensor_info = get_rknn_tensor_info(rknn_model_info, op_outputs[index])
        output_dtype = conver_rknn_tensor_type_to_onnx_type(output_tensor_info["dtype"]["vx_type"])
        need_quant.append(False)
        quant_inputs.append(None)
        if output_dtype == TensorProto.UINT8 or \
            output_dtype == TensorProto.INT8:
            need_quant[-1] = True
            op_outputs[index] = op_outputs[index] + "_quant_input"
            quant_inputs[-1] = op_outputs[index]
    
    node_name = node_info["name"]
    if log_flag:
        print("node_name  : {}".format(node_name))
        print("node_input : {}".format(op_inputs))
        print("node_output: {}".format(op_outputs))

    node_attr = construct_node_attr(node_info)
    if len(node_attr.keys()) > 0:
        node = helper.make_node(onnx_op_name, inputs=op_inputs, outputs=op_outputs, name=node_name, **node_attr)
    else:
        node = helper.make_node(onnx_op_name, inputs=op_inputs, outputs=op_outputs, name=node_name)
    onnx_model_info["nodes"].append(node)
    
    for index in range(len(need_quant)):
        if need_quant[index]:
            quant_input_name = quant_inputs[index]
            dequant_out_tensor_info = {}
            dequant_out_tensor_info["name"] = output_names[index]
            dequant_out_tensor_info["dtype"] = output_tensor_info["dtype"]
            insert_quant_node(dequant_out_tensor_info, quant_input_name, onnx_model_info, log_flag)


def construct_sigmoid_op(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag):
    node_info = rknn_model_info["nodes"][node_index]
    input_names = node_info["inputs"]
    output_names = node_info["outputs"]
    assert len(input_names) == 1, "rknn sigmoid op should have 1 input"
    assert len(output_names) == 1, "rknn sigmoid op should have 1 output"
    construct_with_dequant_quant_node(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag)
    return True


def construct_add_op(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag):
    node_info = rknn_model_info["nodes"][node_index]
    input_names = node_info["inputs"]
    output_names = node_info["outputs"]
    assert len(input_names) == 2, "rknn add op should have 2 input"
    assert len(output_names) == 1, "rknn add op should have 1 output"
    construct_with_dequant_quant_node(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag)
    return True


def construct_relu_op(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag):
    node_info = rknn_model_info["nodes"][node_index]
    input_names = node_info["inputs"]
    output_names = node_info["outputs"]
    assert len(input_names) == 1, "rknn relu op should have 1 input"
    assert len(output_names) == 1, "rknn relu op should have 1 output"
    construct_with_dequant_quant_node(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag)
    return True


def construct_conv2d_op(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag):
    node_info = rknn_model_info["nodes"][node_index]
    input_names = node_info["inputs"]
    output_names = node_info["outputs"]
    assert len(input_names) == 2 or len(input_names) == 3, "rknn conv2d op should have 2 or 3 input"
    assert len(output_names) == 1, "rknn conv2d op should have 1 output"
    op_inputs = []
    for index in range(len(input_names)):
        if index > 1:
            break
        op_inputs.append(input_names[index])
        op_inputs.append(input_names[index]+"_scale")
        op_inputs.append(input_names[index]+"_zero_point")
    op_inputs.append(output_names[0]+"_scale")
    op_inputs.append(output_names[0]+"_zero_point")
    if len(input_names) == 3:
        op_inputs.append(input_names[2])
        tensor_info = get_rknn_tensor_info(rknn_model_info, input_names[2])
        if tensor_info["dtype"]["zero_point"] != 0:
            print(tensor_info)
    op_outputs = []
    for index in range(len(output_names)):
        op_outputs.append(output_names[index])
    op_attr = node_info["attribute"]["conv2d"]
    if "ksize" in op_attr.keys():
        kernel_shape = op_attr["ksize"]

    if "group" in op_attr.keys():
        group = op_attr["group"]
        
    if "dilation" in op_attr.keys():
        dilations = op_attr["dilation"]
        
    if "stride" in op_attr.keys():
        strides = op_attr["stride"]

    if "pad" in op_attr.keys():
        pads = op_attr["pad"]

    node_name = node_info["name"]
    if log_flag:
        print("node_name  : {}".format(node_name))
        print("node_input : {}".format(op_inputs))
        print("node_output: {}".format(op_outputs))
        print("kenel_shape: {}".format(kernel_shape))
        print("group     : {}".format(group))
        print("dilations  : {}".format(dilations))
        print("strides    : {}".format(strides))
        print("pads       : {}".format(pads))

    node = helper.make_node(onnx_op_name, inputs=op_inputs, outputs=op_outputs, name=node_name,
        kernel_shape=kernel_shape,
        group=group,
        dilations=dilations,
        strides=strides,
        pads=pads)
    onnx_model_info["nodes"].append(node)
    return True


def construct_constant_op(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag):
    node_info = rknn_model_info["nodes"][node_index]
    input_names = node_info["inputs"]
    output_names = node_info["outputs"]
    assert len(input_names) == 1, "rknn variable op should have 1 input"
    assert len(output_names) == 1, "rknn variable op should have 1 output"
    # replace node input tensor name(virtural tensor) with virable node input tensor(const tensor), to simulate constant op
    replace_nodes = []
    nodes = rknn_model_info["nodes"]
    for index in range(len(nodes)):
        node = nodes[index]
        for input_index in range(len(node["inputs"])):
            if output_names[0] == node["inputs"][input_index]:
                node["inputs"][input_index] = input_names[0]
                replace_nodes.append(node["name"])
    if log_flag:
        for node in replace_nodes:
            print("replace node {} input with const tensor".format(node))
    return True


def construct_transpose_op(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag):
    node_info = rknn_model_info["nodes"][node_index]
    input_names = node_info["inputs"]
    output_names = node_info["outputs"]
    assert len(input_names) == 1, "rknn permute op should have 1 input"
    assert len(output_names) == 1, "rknn permute op should have 1 output"
    op_inputs = []
    for index in range(len(input_names)):
        op_inputs.append(input_names[index])
    op_outputs = []
    for index in range(len(output_names)):
        op_outputs.append(output_names[index])
    op_attr = node_info["attribute"]["permute"]
    assert "perm" in op_attr.keys(), "rknn permute op should have perm attribute"
    if "perm" in op_attr.keys():
        perm = op_attr["perm"]
    node_name = node_info["name"]
    node = helper.make_node(onnx_op_name, 
        inputs=op_inputs, 
        outputs=op_outputs, 
        name=node_name,
        perm=perm)
    onnx_model_info["nodes"].append(node)
    return True


def construct_reshape_op(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag):
    node_info = rknn_model_info["nodes"][node_index]
    input_names = node_info["inputs"]
    output_names = node_info["outputs"]
    assert len(input_names) == 1, "rknn reshape op should have 1 input"
    assert len(output_names) == 1, "rknn reshape op should have 1 output"
    op_inputs = []
    for index in range(len(input_names)):
        op_inputs.append(input_names[index])
    op_outputs = []
    for index in range(len(output_names)):
        op_outputs.append(output_names[index])
    op_attr = node_info["attribute"]["reshape"]
    assert "size" in op_attr.keys(), "rknn reshape op should have size attribute"
    assert "dim_num" in op_attr.keys(), "rknn reshape op should have dim_num attribute"
    dim_num = op_attr["dim_num"]
    size = op_attr["size"]
    assert dim_num == len(size), "rknn reshape op attribute dim_num not equal to len of attribute size"
    node_name = node_info["name"]
    reshape_shape_tesnor_name = node_name + "_shape"
    reshape_shape_tesnor_type = TensorProto.INT64
    reshape_shape_tesnor_shape = np.array(size, dtype=np.int64).shape
    reshape_shape_tesnor_data = size
    reshape_shape_tesnor = helper.make_tensor(reshape_shape_tesnor_name,
        reshape_shape_tesnor_type, 
        reshape_shape_tesnor_shape, 
        reshape_shape_tesnor_data)
    onnx_model_info["initializers"].append(reshape_shape_tesnor)
    op_inputs.append(reshape_shape_tesnor_name)
    node = helper.make_node(onnx_op_name, 
        inputs=op_inputs, 
        outputs=op_outputs, 
        name=node_name)
    onnx_model_info["nodes"].append(node)
    return True


def pre_run(onnx_model_info, out_tensor_name):
    ori_outputs = onnx_model_info["outputs"]
    onnx_model_info["outputs"] = []
    out_tensor_value_info = helper.make_tensor_value_info(out_tensor_name, TensorProto.UINT8, [-1,-1,-1,-1])
    onnx_model_info["outputs"].append(out_tensor_value_info)
    time_stamp = time.localtime()
    doc_string = "created on {}".format(time.strftime('%Y-%m-%d %H-%M-%S', time_stamp))
    onnx_initializer_list = onnx_model_info["initializers"]
    onnx_input_list = onnx_model_info["inputs"]
    onnx_output_list = onnx_model_info["outputs"]
    onnx_node_list = onnx_model_info["nodes"]
    onnx_graph_def = helper.make_graph(onnx_node_list, 'rknn_graph', onnx_input_list, onnx_output_list, initializer=onnx_initializer_list, doc_string=doc_string)
    op = onnx.OperatorSetIdProto()
    op.version = 11
    onnx_mode_def = helper.make_model(onnx_graph_def, producer_name='rknn_parser', opset_imports=[op])
    onnx_mode_def.ir_version = 6
    onnx.save_model(onnx_mode_def, "shape_infer.onnx")
    sess = rt.InferenceSession("shape_infer.onnx")
    onnx_input = {}
    for input in sess.get_inputs():
        input_name = input.name
        input_shape = input.shape
        input_dtype = convert_onnxruntime_type_to_numpy_dtype(input.type)
        input_data = np.array(np.zeros(input_shape)).astype(input_dtype)
        onnx_input[input_name] = input_data
    pred_out = sess.run([out_tensor_name,], onnx_input)
    onnx_model_info["outputs"] = ori_outputs
    return list(pred_out[0].shape)

def construct_resize_op(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag):
    node_info = rknn_model_info["nodes"][node_index]
    input_names = node_info["inputs"]
    output_names = node_info["outputs"]
    assert len(input_names) == 1, "rknn resize op should have 1 input"
    assert len(output_names) == 1, "rknn resize op should have 1 output"
    op_inputs = []
    for index in range(len(input_names)):
        op_inputs.append(input_names[index])
    op_outputs = []
    for index in range(len(output_names)):
        op_outputs.append(output_names[index])
    node_name = node_info["name"]
    op_attr = node_info["attribute"]["resize"]
    if op_attr["type"] == 'VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR':
        mode = "nearest"

    roi_init_tensor_name = node_name + "_roi"
    roi_init_tensor_type = TensorProto.FLOAT
    roi_init_tensor_shape = [0]
    roi_init_tensor_data = []
    roi_tesnor_def = helper.make_tensor(roi_init_tensor_name, 
        roi_init_tensor_type, 
        roi_init_tensor_shape, 
        roi_init_tensor_data)
    onnx_model_info["initializers"].append(roi_tesnor_def)
    op_inputs.append(roi_init_tensor_name)

    scales_init_tensor_name = node_name + "_scales"
    scales_init_tensor_type = TensorProto.FLOAT
    scales_init_tensor_shape = [0]
    scales_init_tensor_data = []
    scales_tesnor_def = helper.make_tensor(scales_init_tensor_name, 
        scales_init_tensor_type, 
        scales_init_tensor_shape, 
        scales_init_tensor_data)
    onnx_model_info["initializers"].append(scales_tesnor_def)
    op_inputs.append(scales_init_tensor_name)

    expect_shape = pre_run(onnx_model_info, op_inputs[0])
    sizes_init_tensor_name = node_name + "_sizes"
    sizes_init_tensor_type = TensorProto.INT64
    sizes_init_tensor_shape = [len(expect_shape),]
    sizes_init_tensor_data = expect_shape
    if len(op_attr["size"]) < 4:
        start = len(expect_shape) - len(op_attr["size"])
        sizes_init_tensor_data[start:] = op_attr["size"]

    sizes_tesnor_def = helper.make_tensor(sizes_init_tensor_name, 
        sizes_init_tensor_type, 
        sizes_init_tensor_shape, 
        sizes_init_tensor_data)
    onnx_model_info["initializers"].append(sizes_tesnor_def)
    op_inputs.append(sizes_init_tensor_name)

    node = helper.make_node(onnx_op_name, 
        inputs=op_inputs, 
        outputs=op_outputs, 
        name=node_name,
        mode=mode)
    onnx_model_info["nodes"].append(node)

    # expect_shape = pre_run(onnx_model_info, op_outputs[0])
    return True


def construct_mul_op(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag):
    node_info = rknn_model_info["nodes"][node_index]
    input_names = node_info["inputs"]
    output_names = node_info["outputs"]
    assert len(input_names) == 2, "rknn multiply op should have 2 input"
    assert len(output_names) == 1, "rknn multiply op should have 1 output"
    construct_with_dequant_quant_node(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag)
    return True


def construct_pool_op(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag):
    node_info = rknn_model_info["nodes"][node_index]
    input_names = node_info["inputs"]
    output_names = node_info["outputs"]
    assert len(input_names) == 1, "rknn pool op should have 1 input"
    assert len(output_names) == 1, "rknn pool op should have 1 output"
    construct_with_dequant_quant_node(rknn_op_name, onnx_op_name, rknn_model_info, node_index, onnx_model_info, log_flag)
    return True


def get_actual_onnx_name(onnx_op_name, rknn_model_info, node_index):
    node_info = rknn_model_info["nodes"][node_index]
    if onnx_op_name == "Pool":
        op_attr = node_info["attribute"]
        if op_attr["pool"]["type"] == "VX_CONVOLUTIONAL_NETWORK_POOLING_MAX":
            return "MaxPool"
        elif op_attr["pool"]["type"] == "VX_CONVOLUTIONAL_NETWORK_POOLING_AVG":
            return "AveragePool"
        else:
            assert False, "unspported {} pool type".format(op_attr["pool"]["type"])
    return onnx_op_name

class ConstructOnnxOp(object):
    op_construct_funcs = {}
    rknn_op_onnx_op_map = {}
    def register(self, rknn_op_name, onnx_op_name, op_func):
        if rknn_op_name not in self.rknn_op_onnx_op_map.keys():
            self.rknn_op_onnx_op_map[rknn_op_name] = onnx_op_name
            self.op_construct_funcs[onnx_op_name] = op_func
        else:
            print("already register {}".format(rknn_op_name))

    def construct_node(self, rknn_model_info, node_index, onnx_model_info, log_flag = False):
        node_info = rknn_model_info["nodes"][node_index]
        rknn_op_name = node_info['type']
        if rknn_op_name not in self.rknn_op_onnx_op_map.keys():
            print("have not register {}".format(rknn_op_name))
            return False
        else:
            onnx_op_name = self.rknn_op_onnx_op_map[rknn_op_name]
            actual_onnx_op_name = get_actual_onnx_name(onnx_op_name, rknn_model_info, node_index)
            if log_flag:
                print("{}: --------------- convert {} to {}".format(node_index, rknn_op_name, actual_onnx_op_name))
                node_input_list = node_info['input']
                node_output_list = node_info['output']
                for index in range(len(node_input_list)):
                    input_name = node_input_list[index]
                    print("input {}: {}".format(index, input_name))
                for index in range(len(node_output_list)):
                    output_name = node_output_list[index]
                    print("output {}: {}".format(index, output_name))

            convert_flag = self.op_construct_funcs[onnx_op_name](
                rknn_op_name, 
                actual_onnx_op_name, 
                rknn_model_info, 
                node_index, 
                onnx_model_info, 
                log_flag)
            return convert_flag


def init_onnx_construct_obj():
    onnx_obj = ConstructOnnxOp()   
    onnx_obj.register('SIGMOID', 'Sigmoid', construct_sigmoid_op)
    onnx_obj.register('ADD', 'Add', construct_add_op)
    onnx_obj.register('RELU', 'Relu', construct_relu_op)
    onnx_obj.register('CONV2D', 'QLinearConv', construct_conv2d_op)
    onnx_obj.register('VARIABLE', 'Constant', construct_constant_op)
    onnx_obj.register('PERMUTE', 'Transpose', construct_transpose_op)
    onnx_obj.register('RESHAPE', 'Reshape', construct_reshape_op)
    onnx_obj.register('RESIZE', 'Resize', construct_resize_op)
    onnx_obj.register('MULTIPLY', 'Mul', construct_mul_op)
    onnx_obj.register("POOL", "Pool", construct_pool_op)
    return onnx_obj


def constrtut_graph_initializer(rknn_model_info, layout="WHCN", log_flag=False):
    initializer_list = []
    tensor_info = rknn_model_info["tensors"]
    for tensor_name in tensor_info.keys():
        onnx_const_tensor_list = []
        tensor = tensor_info[tensor_name]
        if "norm_tensor" in tensor_name:
            continue
        onnx_dtype = conver_rknn_tensor_type_to_onnx_type(tensor["dtype"]["vx_type"])
        np_dtype = conver_rknn_tensor_type_to_numpy_type(tensor["dtype"]["vx_type"])
        if "const_tensor" in tensor_name:
            # init weight data
            shape = tensor["shape"][::-1]
            weight_data_arr = np.frombuffer(tensor["raw_data"], dtype=np_dtype).reshape(shape)
            # if layout == "WHCN":
            #     shape_len = len(tensor["shape"])
            #     transpose_axes = [shape_len - i - 1 for i in range(shape_len)]
            #     weight_data_arr = np.transpose(weight_data_arr, transpose_axes)

            onnx_const_tensor = {}
            onnx_const_tensor["name"] = tensor_name
            onnx_const_tensor["type"] = onnx_dtype
            onnx_const_tensor["shape"] = weight_data_arr.shape
            onnx_const_tensor["data"] = weight_data_arr
            onnx_const_tensor_list.append(onnx_const_tensor)
    
        # init scale 
        if isinstance(tensor["dtype"]["scale"], list):
            scale_data_arr = np.array(tensor["dtype"]["scale"]).astype(np.float32)
        else:
            scale_data_arr = np.array([tensor["dtype"]["scale"],]).astype(np.float32)
        # init zero point
        if isinstance(tensor["dtype"]["zero_point"], list):
            zero_point_data_arr = np.array(tensor["dtype"]["zero_point"]).astype(np_dtype)
        else:
            zero_point_data_arr = np.array([tensor["dtype"]["zero_point"],]).astype(np_dtype)
        
        onnx_const_tensor = {}
        onnx_const_tensor["name"] = tensor_name + "_scale"
        onnx_const_tensor["type"] = TensorProto.FLOAT
        onnx_const_tensor["shape"] = scale_data_arr.shape
        onnx_const_tensor["data"] = scale_data_arr
        onnx_const_tensor_list.append(onnx_const_tensor)

        onnx_const_tensor = {}
        onnx_const_tensor["name"] = tensor_name + "_zero_point"
        onnx_const_tensor["type"] = onnx_dtype
        onnx_const_tensor["shape"] = zero_point_data_arr.shape
        onnx_const_tensor["data"] = zero_point_data_arr
        onnx_const_tensor_list.append(onnx_const_tensor)
        
        for index in range(len(onnx_const_tensor_list)):
            init_tensor_name = onnx_const_tensor_list[index]["name"]
            init_tensor_type = onnx_const_tensor_list[index]["type"]
            init_tensor_shape = onnx_const_tensor_list[index]["shape"]
            init_tensor_data = onnx_const_tensor_list[index]["data"].flatten().tolist()
            if log_flag:
                print("construct const tensor with:")
                print("tensor name: {}".format(init_tensor_name))
                print("tensor shape: {}".format(init_tensor_shape))
                print("tensor type: {}".format(init_tensor_type))
                print("tensor data: {}".format(init_tensor_data))
            tesnor_def = helper.make_tensor(init_tensor_name, init_tensor_type, init_tensor_shape, init_tensor_data)
            initializer_list.append(tesnor_def)
    return initializer_list


def constrtut_graph_input_output(rknn_model_info, layout="WHCN", log_flag=False):
    input_list = []
    output_list = []
    initializer_list = []
    tensor_info = rknn_model_info["tensors"]
    input_names = []
    for input in rknn_model_info["inputs"]:
        input_names.append(input["tensor_info"]["url"])
    output_names = []
    for output in rknn_model_info["outputs"]:
        output_names.append(output["tensor_info"]["url"])
    for tensor_name in tensor_info.keys():
        tensor = tensor_info[tensor_name]
        if "norm_tensor" in tensor_name:
            onnx_dtype = conver_rknn_tensor_type_to_onnx_type(tensor["dtype"]["vx_type"])
            np_dtype = conver_rknn_tensor_type_to_numpy_type(tensor["dtype"]["vx_type"])
            # init scale 
            if isinstance(tensor["dtype"]["scale"], list):
                scale_data_arr = np.array(tensor["dtype"]["scale"]).astype(np.float32)
            else:
                scale_data_arr = np.array([tensor["dtype"]["scale"],]).astype(np.float32)
            # init zero point
            if isinstance(tensor["dtype"]["zero_point"], list):
                zero_point_data_arr = np.array(tensor["dtype"]["zero_point"]).astype(np_dtype)
            else:
                zero_point_data_arr = np.array([tensor["dtype"]["zero_point"],]).astype(np_dtype)

            if layout == "WHCN":
                tensor_shape = tensor["shape"][::-1]
            onnx_norm_tensor_value_info = helper.make_tensor_value_info(tensor_name, onnx_dtype, tensor_shape)
            if tensor['url'] in input_names:
                input_list.append(onnx_norm_tensor_value_info)
            elif tensor['url'] in output_names:
                output_list.append(onnx_norm_tensor_value_info)
            else:
                assert False, "unrecognize norm tensor {}".format(tensor_name)
            if log_flag:
                print("construct norm tensor with:")
                print("tensor name: {}".format(tensor_name))
                print("tensor shape: {}".format(tensor_shape))
                print("tensor type: {}".format(onnx_dtype))
            onnx_const_tensor_list = []
            onnx_const_tensor = {}
            onnx_const_tensor["name"] = tensor_name + "_scale"
            onnx_const_tensor["type"] = TensorProto.FLOAT
            onnx_const_tensor["shape"] = scale_data_arr.shape
            onnx_const_tensor["data"] = scale_data_arr
            onnx_const_tensor_list.append(onnx_const_tensor)

            onnx_const_tensor = {}
            onnx_const_tensor["name"] = tensor_name + "_zero_point"
            onnx_const_tensor["type"] = onnx_dtype
            onnx_const_tensor["shape"] = zero_point_data_arr.shape
            onnx_const_tensor["data"] = zero_point_data_arr
            onnx_const_tensor_list.append(onnx_const_tensor)
            for index in range(len(onnx_const_tensor_list)):
                init_tensor_name = onnx_const_tensor_list[index]["name"]
                init_tensor_type = onnx_const_tensor_list[index]["type"]
                init_tensor_shape = onnx_const_tensor_list[index]["shape"]
                init_tensor_data = onnx_const_tensor_list[index]["data"].flatten().tolist()
                if log_flag:
                    print("construct const tensor with:")
                    print("tensor name: {}".format(init_tensor_name))
                    print("tensor shape: {}".format(init_tensor_shape))
                    print("tensor type: {}".format(init_tensor_type))
                    print("tensor data: {}".format(init_tensor_data))
                tesnor_def = helper.make_tensor(init_tensor_name, init_tensor_type, init_tensor_shape, init_tensor_data)
                initializer_list.append(tesnor_def)
    return input_list, output_list, initializer_list
            

def constrtut_graph_node(rknn_model_info, onnx_model_info, onnx_construct_obj, log_flag=False):
    for index in range(len(rknn_model_info["nodes"])):
        convert_flag = onnx_construct_obj.construct_node(rknn_model_info, index, onnx_model_info, log_flag)
        assert convert_flag, "convert node {} fail!!!!".format(rknn_model_info["nodes"][index]["name"])


def construct_onnx_from_rknn_model_info(rknn_model_info, onnx_construct_obj, layout="WHCN", log_flag=False):
    graph_initializer = constrtut_graph_initializer(rknn_model_info, layout, log_flag)
    graph_input, graph_output, graph_input_output_initializer = constrtut_graph_input_output(rknn_model_info, layout, log_flag)
    graph_initializer.extend(graph_input_output_initializer)
    onnx_model_info = {}
    onnx_model_info["initializers"] = graph_initializer
    onnx_model_info["inputs"] = graph_input
    onnx_model_info["outputs"] = graph_output
    onnx_model_info["nodes"] = []
    constrtut_graph_node(rknn_model_info, onnx_model_info, onnx_construct_obj, log_flag)
    time_stamp = time.localtime()
    doc_string = "created on {}".format(time.strftime('%Y-%m-%d %H-%M-%S', time_stamp))
    onnx_initializer_list = onnx_model_info["initializers"]
    onnx_input_list = onnx_model_info["inputs"]
    onnx_output_list = onnx_model_info["outputs"]
    onnx_node_list = onnx_model_info["nodes"]
    onnx_graph_def = helper.make_graph(onnx_node_list, 'rknn_graph', onnx_input_list, onnx_output_list, initializer=onnx_initializer_list, doc_string=doc_string)
    op = onnx.OperatorSetIdProto()
    op.version = 11
    onnx_mode_def = helper.make_model(onnx_graph_def, producer_name='rknn_parser', opset_imports=[op])
    onnx_mode_def.ir_version = 6
    checker.check_model(onnx_mode_def)
    if log_flag:
        print('Model Graph:\n\n{}'.format(helper.printable_graph(onnx_mode_def.graph)))
    return onnx_mode_def


def format_onnx_nodes(rknn_model_info, layout):
    nodes_info = rknn_model_info["nodes"]
    for node in nodes_info:
        if node["type"] == "RESIZE":
            node_attr = node["attribute"]["resize"]
            if layout == "WHCN":
                node_attr["size"] = node_attr["size"][::-1]


def convert_rknn_to_onnx(rknn_model_info, save_file="convert.onnx", layout="WHCN", log_flag=False):
    construct_onnx_obj = init_onnx_construct_obj()
    format_onnx_nodes(rknn_model_info, layout)
    onnx_model = construct_onnx_from_rknn_model_info(rknn_model_info, construct_onnx_obj, layout, log_flag)
    onnx.save_model(onnx_model, save_file)