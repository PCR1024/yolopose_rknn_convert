import sys
from rknn.api import RKNN
import os
import argparse

DATASET_PATH = './dataset.txt'
DEFAULT_QUANT = True
DEFAULT_BATCH_SIZE=1

def parse_arg():
    if len(sys.argv) < 7:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)] [quantized_algorithm]".format(sys.argv[0]));
        print("       platform choose from [rk3562,rk3566,rk3568,rk3576,rk3588,rk1808,rv1109,rv1126]")
        print("       dtype choose from [i8, fp] for [rk3562,rk3566,rk3568,rk3576,rk3588]")
        print("       dtype choose from [u8, fp] for [rk1808,rv1109,rv1126]")
        print("       quantized_algorithm choose from [normal, mmse, kl_divergence]")
        print("       batch_size default is 1")
        print("       data_path default is ./dataset.txt")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]
    quantized_algorithm = sys.argv[5]
    batch_size = 1
    data_path = DATASET_PATH

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        exit(1)

    if len(sys.argv) > 5:
        quantized_algorithm = sys.argv[5]
    else:
        quantized_algorithm = "normal"
        
    if len(sys.argv) > 6:
        batch_size = int(sys.argv[6])
    else:
        batch_size = DEFAULT_BATCH_SIZE
        
    if len(sys.argv) > 7:
        data_path = sys.argv[7]
    else:
        data_path = DATASET_PATH
        
    name, ext = os.path.splitext(output_path)
    output_path = f"{name}_{quantized_algorithm}{ext}"

    return model_path, platform, do_quant, output_path, quantized_algorithm, batch_size, data_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path, quantized_algorithm, batch_size, data_path = parse_arg()

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform=platform, quantized_algorithm=quantized_algorithm, model_pruning=True) 
    #quantized_algorithm: currently support: normal, mmse (Min Mean Square Error), kl_divergence. default is normal.
    
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=data_path, rknn_batch_size=batch_size)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
