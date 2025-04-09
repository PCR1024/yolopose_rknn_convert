## yolopose_rknn_convert
一个简易的将ultralytics官方YOLOV8/YOLO11的各大小Pose模型转换为RKNN的脚本，同时适配RKNN官方modelzoo https://github.com/airockchip/rknn_model_zoo/tree/main 里的yolov8_pose例程

运行前需pip安装ultralytics, rknn_toolkit2

## 运行/RUN
```sh
bash convert.sh
```
参数在convert.sh中修改
## convert.sh 参数说明
| param | dtype | description | 
| -- | -- | -- |
| ultralytics_pt_path | `str` | ultralytics官方模型路径 |
| ultralytics_onnx_path | `str` | ultralytics导出onnx路径(自选) |
| modified_onnx_path | `str` | 修改onnx路径 |
| rknn_output_path | `str` | rknn输出路径 |
| target | `str` | 运行目标 |
| quant_algorithm | `str` | rknn量化算法 |
| data_path | `str` | 量化数据集路径txt |
