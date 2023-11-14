import h5py
import numpy as np
import keras
from pathlib import Path
from yolov3.yolov4 import Create_Yolo

def peak_h5_weights():
    import h5py
    filename = "./checkpoints/yolov3_custom_Tiny_val_loss_  10.01.h5"

    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        # print(type(f[a_group_key])) 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        # data = list(f[a_group_key])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]      # returns as a h5py dataset object
        # ds_arr = f[a_group_key][()]  # returns as a numpy array


def load_h5_weights(weights_file: Path):
    range1 = 13
    range2 = [9, 12]
    yolo = Create_Yolo(input_size=416, CLASSES="./dataset/dataset.names")
    yolo.load_weights("/home/plamedi/Documents/repos/pocket/TensorFlow-2.x-YOLOv3/checkpoints/yolov3_custom_Tiny_val_loss_  10.01.h5") # use keras weights
    
    model = yolo
    model.summary()
    weights_array = np.array([], dtype=np.float32)
    conv_bias = np.array([])
    bn_weights = np.array([])
    with open(weights_file, 'wb') as wf:
        start_buffer = np.array((0, 1, 2, 3, 4), dtype=np.int32)
        start_buffer.tofile(wf, sep="")

        j = 0
        for i in range(range1):
            if i > 0:
                conv_layer_name = 'conv2d_%d' %i
            else:
                conv_layer_name = 'conv2d'
                
            if j > 0:
                bn_layer_name = 'batch_normalization_%d' %j
            else:
                bn_layer_name = 'batch_normalization'
            
            conv_layer = model.get_layer(conv_layer_name)
            if i not in range2:
                # darknet weights: [beta, gamma, mean, variance]
                # tf weights: [gamma, beta, mean, variance]
                bn_layer = model.get_layer(bn_layer_name)
                bn_weights = np.array(bn_layer.get_weights(), dtype=np.float32)
                print(bn_weights)
                j += 1
            else:
                conv_bias = np.array(conv_layer.get_weights()[1], dtype=np.float32)

            # darknet shape (out_dim, in_dim, height, width)
            conv_weights = np.array(conv_layer.get_weights()[0], dtype=np.float32).transpose([3, 2, 0, 1])
            # print(f"kernels: \n{conv_weights}\n")
            # tf shape (height, width, in_dim, out_dim)

            if i not in range2:
                weights_array = np.append(weights_array, bn_weights.flatten())
                weights_array = np.append(weights_array, conv_weights.flatten())
            else:
                weights_array = np.append(weights_array, conv_bias.flatten())
                weights_array = np.append(weights_array, conv_weights.flatten())

        print(f"weights length = {len(weights_array)}")
        # assert len(weights_array) == 8680864, 'failed to write all weights'
        weights_array.tofile(wf, sep="")

if __name__ == "__main__":
    load_h5_weights(Path("~/Downloads/weights/debris_extremem.weights").expanduser())
    # peak_h5_weights()
    print("\nweights convertion complete")
