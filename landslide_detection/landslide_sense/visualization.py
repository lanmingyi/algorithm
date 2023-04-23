import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

file = h5py.File("./test_map/mask_1.h5", "r+")
# file = h5py.File("./test_map/image_801.h5", "r+")

for key in file.keys(): 
    img = file[key][:] if file[key].ndim == 2 else file[key][:,:,1]
    plt.imshow(img) 
    plt.axis('off') 
    plt.show()
    plt.close()
    
file.close()


