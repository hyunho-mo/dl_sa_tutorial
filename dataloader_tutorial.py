import os
import numpy as np
import pandas as pd

import time
import sys
import argparse
import h5py

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "experiments/data/whas")

data_filepath =os.path.join(data_dir, "whas_train_test.h5")
whas_dirpath = os.path.join(current_dir, "whas")

def h5_to_array(hdf5_group):

    x_array = np.array(hdf5_group['x'])
    t_array = np.array(hdf5_group['t'])
    e_array = np.array(hdf5_group['e'])

    return x_array, t_array, e_array


def main():
    

    h5_raw = h5py.File(data_filepath,'r')

    print (list(h5_raw.keys()))

    # print (h5_raw)

    train_data = h5_raw['train']  # train data from the raw file of WHAS
    test_data = h5_raw['test']  # test data from the raw file of WHAS 

    # h5_raw.close()


    print (train_data)
    print (train_data.keys())
    print (test_data)
    
	# 'x': (n,d) observations (dtype = float32), 
 	# 't': (n) event times (dtype = float32),
 	# 'e': (n) event indicators (dtype = int32)



    # data = hf.get('dataset_name').value # `data` is now an ndarray.
    # n1 = np.array(hf["dataset_name"][:]) #dataset_name is same as hdf5 object name

    # train_x_array = np.array(train_data['x'])
    # print ("train_x_array", train_x_array)
    # print ("train_x_array.shape", train_x_array.shape)

    train_x_array, train_t_array, train_e_array = h5_to_array(train_data)
    print ("train_x_array", train_x_array)
    print ("train_t_array", train_t_array)
    print ("train_e_array", train_e_array)

   
    whas_csv_file_path = os.path.join(whas_dirpath, "whas_train.csv")
    train_x_df = pd.DataFrame(train_x_array, columns=['case','age','sex','bmi',
    'chf','miord'])

    train_x_df.to_csv(whas_csv_file_path, index=False)

    train_t_array = np.reshape(train_t_array, (train_t_array.shape[0],1))
    train_e_array = np.reshape(train_e_array, (train_e_array.shape[0],1))

    train_full_array = np.concatenate((train_x_array, train_t_array, train_e_array), axis=1)
    whas_full_file_path = os.path.join(whas_dirpath, "whas_train_full.csv")
    train_x_full_df = pd.DataFrame(train_full_array, columns=['case','age','sex','bmi',
    'chf','miord','time','indicator'])

    train_x_full_df.to_csv(whas_full_file_path, index=False)


    


if __name__ == '__main__':
    main()    

