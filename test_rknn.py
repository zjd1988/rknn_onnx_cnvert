from parse_rknn import parse_rknn_model
from onnx_convert import convert_rknn_to_onnx

if __name__ == "__main__":
    rknn_model = "./test.rknn"
    with open(rknn_model, "rb") as f:
        rknn_data = f.read()
    rknn_model_info = parse_rknn_model(rknn_data)
    convert_rknn_to_onnx(rknn_model_info, "./convert.onnx")