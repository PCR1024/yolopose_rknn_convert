#!/bin/bash

yolo_version='v8'
yolo_size='x'
batch_size=3

ultralytics_pt_path=./model/yolo${yolo_version}${yolo_size}-pose.pt
ultralytics_onnx_path=./model/yolo${yolo_version}${yolo_size}-pose.onnx
modified_onnx_path=./model/modified_yolo${yolo_version}${yolo_size}-pose.onnx
rknn_output_path=./model/modified_yolo${yolo_version}${yolo_size}-pose.rknn
target=rk3588
quant_algorithm=normal
data_path=./dataset.txt

yolo "export" model=${ultralytics_pt_path} format=onnx
python modify_onnx.py --model_path ${ultralytics_onnx_path} --output_path ${modified_onnx_path} --model_version yolo${yolo_version}
python convert.py ${modified_onnx_path} ${target} i8 ${rknn_output_path} ${quant_algorithm} ${batch_size} ${data_path}
rm ${modified_onnx_path} ${ultralytics_onnx_path}