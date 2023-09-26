import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
from scipy import io



def Vol2Patch(img, patch_size, stride=1):
    img_torch = torch.tensor(img).unsqueeze(0)
    patch = img_torch.unfold(1, patch_size, stride).unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    return(patch.squeeze().reshape(-1, 1, patch_size, patch_size, patch_size))


def prepare_data(patch_size, stride, aug_times=1, image_type='BSD'):
    # train
    print('process training data')

    files = glob.glob(f'./volumes/train/*.mat')


    files.sort()

    output_directory_train = f'preprocessed_3d/'
    if not os.path.exists(output_directory_train):
        os.makedirs(output_directory_train)
    h5f = h5py.File(f'{output_directory_train}/train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        Vol = make_volume(files[i])
        patches = Vol2Patch(Vol, patch_size, stride)
        for i in range(patches.shape[0]):
            data = np.float32(patches[i].numpy())
            h5f.create_dataset(str(train_num), data=data)
            train_num += 1
    h5f.close()

    output_directory_val = f'preprocessed_3d/'
    if not os.path.exists(output_directory_train):
        os.makedirs(output_directory_train)
    h5f = h5py.File(f'{output_directory_train}/val.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        Vol = np.float32(make_volume(files[i])[None, :, :, :])
        h5f.create_dataset(str(val_num), data=Vol)
        val_num += 1
    h5f.close()

    #     h, w, c = img.shape
    #     for k in range(len(scales)):
    #         Vol = np.float32(Vol)
    #         patches = Im2Patch(Img, win=patch_size, stride=stride)
    #         print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
    #         for n in range(patches.shape[3]):
    #             data = patches[:,:,:,n].copy()
    #             h5f.create_dataset(str(train_num), data=data)
    #             train_num += 1
    #             for m in range(aug_times-1):
    #                 data_aug = data_augmentation(data, np.random.randint(1,8))
    #                 h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
    #                 train_num += 1
    # h5f.close()

    # # val
    # print('\nprocess validation data')
    # files.clear()
    # if image_type == 'BSD':
    #     files = glob.glob(f'./images/{image_type}/Set12/*.png')
    # else:
    #     files = glob.glob(f'./images/{image_type}/validation/*.png')
    
    # files.sort()
    # output_directory_val = f'preprocessed/{image_type}'
    # if not os.path.exists(output_directory_val):
    #     os.makedirs(output_directory_val)
    # h5f = h5py.File(f'{output_directory_val}/validation.h5', 'w')
    # val_num = 0
    # for i in range(len(files)):
    #     print("file: %s" % files[i])
    #     img = cv2.imread(files[i])
    #     img = np.expand_dims(img[:,:,0], 0)
    #     img = np.float32(normalize(img))
    #     h5f.create_dataset(str(val_num), data=img)
    #     val_num += 1
    # h5f.close()

    # # test
    # if image_type == 'BSD':
    #     print('\nprocess test data')
    #     files.clear()
    #     files = glob.glob(os.path.join('./images/BSD/Set68', '*.png'))
    #     files.sort()
    #     output_directory_test = f'preprocessed/{image_type}'
    #     if not os.path.exists(output_directory_test):
    #         os.makedirs(output_directory_test)
    #     h5f = h5py.File(f'{output_directory_test}/test.h5', 'w')
    #     test_num = 0
    #     for i in range(len(files)):
    #         print("file: %s" % files[i])
    #         img = cv2.imread(files[i])
    #         img = np.expand_dims(img[:,:,0], 0)
    #         img = np.float32(normalize(img))
    #         h5f.create_dataset(str(test_num), data=img)
    #         test_num += 1
    #     h5f.close()
    # else:
    #     test_num = 0

    # print('training set, # samples %d\n' % train_num)
    # print('val set, # samples %d\n' % val_num)
    # print('test set, # samples %d\n' % test_num)


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))

def make_volume(f_path):
    x = io.loadmat(f_path)

    B = np.moveaxis(x['B_ideal_resized'],[0, 1], [-1, -2])
    C = x['C_downsample']
    V = np.dot(B, C)

    V = (V - np.min(V))

    V = V / np.max(V) 

    return(V)

if __name__ == "__main__":
    prepare_data(patch_size=40, stride=20, aug_times=1)