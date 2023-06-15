from utils.datastruct import CNNData
import numpy as np

def conv3D_output_size(args, img_size):

    if not isinstance(args, CNNData):
            raise TypeError("input must be a ParserClass")
            
    (cin, h , w) = img_size
    # compute output shape of conv3D
    for idx , chan in enumerate(args.kernel_size):
        padding = args.paddings[idx]
        stride = args.strides[idx]
        (cin, h , w) = (np.floor((cin + 2 * padding[0] - chan[0] ) / stride[0] + 1).astype(int),
                    np.floor((h + 2 * padding[1] - chan[1] ) / stride[1] + 1).astype(int),
                    np.floor((w + 2 * padding[2] - chan[2] ) / stride[2] + 1).astype(int))
        
        
    final_dim = int(args.n_f[-1] * cin * h * w)
    
    return final_dim