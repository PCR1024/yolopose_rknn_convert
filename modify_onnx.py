import onnx
from onnx import helper, shape_inference
import argparse

def remove_node_and_children(model, target_node_name):
    graph = model.graph
    node_map = {node.name: node for node in graph.node}

    def get_descendants(node_name, visited=None):
        if visited is None:
            visited = set()
        visited.add(node_name)
        for node in graph.node:
            if any(inp in node_map[node_name].output for inp in node.input) and node.name not in visited:
                get_descendants(node.name, visited)
        return visited

    to_remove = get_descendants(target_node_name)
    new_nodes = [node for node in graph.node if node.name not in to_remove]
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    print(f"Removed node '{target_node_name}' and its child nodes: {to_remove}")

def remove_output_node(model, output_name):
    graph = model.graph
    new_outputs = [output for output in graph.output if output.name != output_name]
    if len(new_outputs) == len(graph.output):
        print(f"Output '{output_name}' not found in the model.")
    else:
        graph.ClearField("output")
        graph.output.extend(new_outputs)
        print(f"Removed output node: {output_name}")

def add_output_node(model, new_output_name):
    graph = model.graph
    existing_outputs = {value_info.name for value_info in graph.output}
    
    if new_output_name in existing_outputs:
        print(f"Output '{new_output_name}' already exists in the model.")
        return
    
    for node in graph.node:
        if new_output_name in node.output:
            output_type = helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, shape=None)
            new_output = helper.make_value_info(new_output_name, output_type)
            graph.output.append(new_output)
            print(f"Added new output node: {new_output_name}")
            return
    
    print(f"Node output '{new_output_name}' not found in the model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modify Ultralytics Yolo Pose ONNX for RKNN Convert', add_help=True)
    # basic params
    parser.add_argument('--model_path', type=str, required=True,
                        help='model path, could be .onnx file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='output model path, could be .onnx file')
    parser.add_argument('--model_version', type=str, required=True,
                        help='model version, choose from [yolov8, yolov11]')
    args = parser.parse_args()

    layer_num = 23

    if args.model_version == 'yolov8':
        layer_num = 22
    elif args.model_version == 'yolo11':
        layer_num = 23
    else:
        print('model version should be chosen from [yolov8, yolov11]')
        exit(1)

    # 加载onnx模型
    model_path = '/hy-tmp/model/onnx/yolo11l-pose.onnx'
    model = onnx.load(args.model_path)
    print("Loaded ONNX model.")
    
    # 移除指定节点及其子节点
    target_nodes = [f'/model.{layer_num}/Reshape_3', f'/model.{layer_num}/Reshape_4', f'/model.{layer_num}/Reshape_5', f'/model.{layer_num}/Reshape_7']
    for node in target_nodes:
        remove_node_and_children(model, node)
    
    # 移除已有输出节点
    remove_output_node(model, 'output0')
    
    # 添加输出节点
    new_outputs = [f'/model.{layer_num}/Concat_1_output_0', f'/model.{layer_num}/Concat_2_output_0', f'/model.{layer_num}/Concat_3_output_0', f'/model.{layer_num}/Concat_6_output_0']
    for output in new_outputs:
        add_output_node(model, output)
    
    # 推断输出节点结构
    model = shape_inference.infer_shapes(model)
    print("Shape inference completed.")
    
    # 保存模型
    onnx.save(model, args.output_path)
    print(f"Modified ONNX model saved as {args.output_path}.")
