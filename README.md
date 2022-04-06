# rknn_onnx_cnvert
convert rknn to onnx model

## 支持的rknn算子
参照netron对rknn解析的代码，将对应的算子使用onnx算子代替，目前支持的rknn算子
VSI_NN_OP_SIGMOID  
VSI_NN_OP_ADD  
VSI_NN_OP_RELU  
VSI_NN_OP_CONV2D  
VSI_NN_OP_VARIABLE  
VSI_NN_OP_PERMUTE  
VSI_NN_OP_RESHAPE  
VSI_NN_OP_RESIZE  
VSI_NN_OP_MULTIPLY  
VSI_NN_OP_POOL  

## 进展
写这个代码的初衷是为了在本地推理rknn模型，因为rk官方提供的simulator推理速度有些慢，  
要验证转换后的rknn模型的话，往往需要编写c++代码，比较耗费时间，因此才考虑能不能把模型  
转成onnx，进行推理，目前我测试发现从rknn转换成onnx模型在推理时一些层结果不一致，所以  
实际并没有达到最初的目的，但是有用的一点是将rknn模型信息通过onnx展示了出来，可以方便大  
家学习下rk的模型网络结构，因为使用netron打开rknn模型是看不到scale和zero_point等信息的。  


## 计划
因为rk用的芯原微的npu ip核，因此考虑看能不能通过使用tim-vx来复现rk网络的方式来进行推理
，很可惜目前tim-vx暂不提供python接口，所以后期在考虑要不要自己封装下python api