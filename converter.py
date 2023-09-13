# Script converter_h5-2-wts.py
# -*- coding: utf-8 -*-
# ''' yolov3_keras_to_darknet.py'''
# import argparse
# import numpy
# import numpy as np
# import keras
# from keras.models import load_model
# from keras import backend as K
# from yolov3.yolov4 import Create_Yolo
# from yolov3.configs import *
#
# def parser():
#     parser = argparse.ArgumentParser(description="Darknet\'s yolov3.cfg and yolov3.weights converted into Keras\'s yolov3.h5!")
#     parser.add_argument('-cfg_path', help='yolov3.cfg')
#     parser.add_argument('-h5_path', help='yolov3.h5')
#     parser.add_argument('-output_path', help='yolov3.weights')
#     return parser.parse_args()
#
# class WeightSaver(object):
#
#     def __init__(self,h5_path,output_path):
#         yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
#         yolo.load_weights("/home/plamedi/Documents/repos/pocket/TensorFlow-2.x-YOLOv3/checkpoints/yolov3_custom_Tiny.data-00000-of-00001") # use keras weights
#         self.model = yolo
#         self.layers = {weight.name:weight for weight in self.model.weights}
#         # self.sess = K.get_session()
#         self.fhandle = open(output_path,'wb')
#         self._write_head()
#
#     def _write_head(self):
#         numpy_data = numpy.ndarray(shape=(3,),
#                       dtype='int32',
#                       buffer=np.array([0,2,0],dtype='int32') )
#         self.save(numpy_data)
#         numpy_data = numpy.ndarray(shape=(1,),
#                       dtype='int64',
#                       buffer=np.array([320000],dtype='int64'))
#         self.save(numpy_data)
#
#     def get_bn_layername(self,num):
#         layer_name = 'batch_normalization_{num}'.format(num=num)
#         bias = self.layers['{0}/beta:0'.format(layer_name)]
#         scale = self.layers['{0}/gamma:0'.format(layer_name)]
#         mean = self.layers['{0}/moving_mean:0'.format(layer_name)]
#         var = self.layers['{0}/moving_variance:0'.format(layer_name)]
#         bias_np = self.get_numpy(bias)
#         scale_np = self.get_numpy(scale)
#         mean_np = self.get_numpy(mean)
#         var_np = self.get_numpy(var)
#         return bias_np,scale_np,mean_np,var_np
#
#     def get_convbias_layername(self,num):
#         layer_name = 'conv2d_{num}'.format(num=num)
#         bias = self.layers['{0}/bias:0'.format(layer_name)]
#         bias_np = self.get_numpy(bias)
#         return bias_np
#
#     def get_conv_layername(self,num):
#         layer_name = 'conv2d_{num}'.format(num=num)
#         conv = self.layers['{0}/kernel:0'.format(layer_name)]
#         conv_np = self.get_numpy(conv)
#         return conv_np
#
#     def get_numpy(self,layer_name):
#         numpy_data = layer_name.numpy()
#         return numpy_data
#
#     def save(self,numpy_data):
#         bytes_data = numpy_data.tobytes()
#         self.fhandle.write(bytes_data)
#         self.fhandle.flush()
#
#     def close(self):
#         self.fhandle.close()
#
# class KerasParser(object):
#
#     def __init__(self, cfg_path, h5_path, output_path):
#         self.block_gen = self._get_block(cfg_path)
#         self.weights_saver = WeightSaver(h5_path, output_path)
#         self.count_conv = 0
#         self.count_bn = 0
#
#     def _get_block(self,cfg_path):
#
#         block = {}
#         with open(cfg_path,'r', encoding='utf-8') as fr:
#             for line in fr:
#                 line = line.strip()
#                 if '[' in line and ']' in line:
#                     if block:
#                         yield block
#                     block = {}
#                     block['type'] = line.strip(' []')
#                 elif not line or '#' in line:
#                     continue
#                 else:
#                     key,val = line.strip().replace(' ','').split('=')
#                     key,val = key.strip(), val.strip()
#                     block[key] = val
#
#             yield block
#
#     def close(self):
#         self.weights_saver.close()
#
#     def conv(self, block):
#         self.count_conv += 1
#         batch_normalize = 'batch_normalize' in block
#         print('handing.. ',self.count_conv)
#
#         # If bn exists, process bn first, in order of bias, scale, mean, var
#         if batch_normalize:
#             bias,scale,mean,var = self.bn()
#             self.weights_saver.save(bias)
#             
#             scale = scale.reshape(1,-1)
#             mean = mean.reshape(1,-1)
#             var = var.reshape(1,-1)
#             remain = np.concatenate([scale,mean,var],axis=0)
#             self.weights_saver.save(remain)
#
#         # biase
#         else:
#             conv_bias = self.weights_saver.get_convbias_layername(self.count_conv)
#             self.weights_saver.save(conv_bias)
#
#         # weights
#         conv_weights = self.weights_saver.get_conv_layername(self.count_conv)
#         # (height, width, in_dim, out_dim) (out_dim, in_dim, height, width)
#         conv_weights = np.transpose(conv_weights,[3,2,0,1])
#         self.weights_saver.save(conv_weights)
#
#     def bn(self):
#         self.count_bn += 1
#         bias,scale,mean,var = self.weights_saver.get_bn_layername(self.count_bn) 
#         return bias,scale,mean,var
#
# def main():
#     args = parser()
#     keras_loader = KerasParser(args.cfg_path, args.h5_path, args.output_path)
#
#     for block in keras_loader.block_gen:
#         if 'convolutional' in block['type']:
#             keras_loader.conv(block)
#     keras_loader.close()
#
#
# if __name__ == "__main__":
#     main()
import h5py
import numpy as np
import keras
from pathlib import Path
from yolov3.yolov4 import Create_Yolo

def load_h5_weights(weights_file: Path):
    range1 = 13
    range2 = [9, 12]
    yolo = Create_Yolo(input_size=416, CLASSES="./dataset/dataset.names")
    yolo.load_weights("/home/plamedi/Documents/repos/pocket/TensorFlow-2.x-YOLOv3/checkpoints/yolov3_custom_Tiny.data-00000-of-00001.h5") # use keras weights
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
    # conv_layer: keras.Model = model.get_layer("conv2d_12")
    # conv_weights = conv_layer.get_weights()
    # print(np.array(conv_weights[0]).shape)
    # print(np.array(conv_weights)[0].transpose((3, 2, 1, 0)).flatten().shape)

    # bn_layer = model.get_layer("batch_normalization_3")
    # bn_weights = bn_layer.get_weights()
    # print(np.array(bn_weights)[[1, 0, 2, 3]].flatten().shape)
            if i not in range2:
                # darknet weights: [beta, gamma, mean, variance]
                # tf weights: [gamma, beta, mean, variance]
                bn_layer = model.get_layer(bn_layer_name)
                bn_weights = np.array(bn_layer.get_weights(), dtype=np.float32)[[1, 0, 2, 3]]
                j += 1
            else:
                conv_bias = np.array(conv_layer.get_weights()[1], dtype=np.float32)

            # darknet shape (out_dim, in_dim, height, width)
            conv_weights = np.array(conv_layer.get_weights()[0], dtype=np.float32).transpose([3, 2, 0, 1])
            # tf shape (height, width, in_dim, out_dim)

            if i not in range2:
                weights_array = np.append(weights_array, conv_weights.flatten())
                weights_array = np.append(weights_array, bn_weights.flatten())
            else:
                weights_array = np.append(weights_array, conv_bias.flatten())
                weights_array = np.append(weights_array, conv_weights.flatten())

        print(f"weights length = {len(weights_array)}")
        assert len(weights_array) == 8680864, 'failed to write all weights'
        weights_array.tofile(wf, sep="")

if __name__ == "__main__":
    load_h5_weights(Path("~/Downloads/weights/debris.weights").expanduser())
    print("\nweights convertion complete")

#
# # Open the HDF5 file containing the weights
# with h5py.File('/home/plamedi/Downloads/weights/yolov3_custom_Tiny', 'r') as file:
#     # List the groups (layers) in the HDF5 file
#     print("Layers in the HDF5 file:")
#     for layer_name in file.keys():
#         print(layer_name)
#
#     # Display the weights for a specific layer (change 'layer_name' accordingly)
#     layer_name = 'conv2d_1'  # Replace with the name of the layer you want to inspect
#     weights = file[layer_name][layer_name]["kernel:0"][:]
#     weights = np.array(weights)
#     print(f"Weights for layer '{layer_name}':")
#     print(weights)
#     print(f"weights info: shape: {weights.shape} dtype: {weights.dtype} ndim: {weights.ndim}")

# import h5py
# keys = []
# with h5py.File("/home/plamedi/Downloads/weights/yolov3_custom_Tiny",'r') as f: # open file
#     f.visit(keys.append) # append all keys to list
#     for key in keys:
#             if ':' in key: # contains data if ':' in key
#                 print(f[key].name)

# import numpy as np  #Define the info to extract
# f = h5py.File("/home/plamedi/Downloads/weights/yolov3_custom_Tiny",'r')
# group = f[key]
# b4 = group['/dense_5/dense_5/bias:0'].value
# k4= group['/dense_5/dense_5/kernel:0'].value
# K4=np.transpose(np.array(k4)) #Process
#    
# f.close() #Close key
#
# from numpy import savetxt #Save as human readable
# savetxt('b4.csv', b4, delimiter=',')
# savetxt('w4.csv', K4, delimiter=',')
