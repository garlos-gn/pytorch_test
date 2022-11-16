import glob
import cv2
import SimpleITK as sitk
import numpy as np
from functions.data_treatment.indexes import get_indexes, delete_indexes, extend_images
'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''


def load_img(filename):
    'This function loads the .mhd images'
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def load_label(path):
    'This function loads the labels'
    path_lumen = glob.glob("O:\Carlos\Data\lumen/ivus_" + path + "\*.png")
    path_vessel = glob.glob("O:\Carlos\Data/vessel/ivus_" + path + "\*.png")

    images_lumen = [cv2.imread(file) for file in path_lumen]
    images_vessel = [cv2.imread(file) for file in path_vessel]
    images_lumen = np.array(images_lumen)
    images_vessel = np.array(images_vessel)
    return images_lumen, images_vessel


def load_everything():
    'This function loads all the available lumen and vessel images into memory'
    path = glob.glob("O:\Carlos\Data\lumen/*")
    index = np.arange(len(path)).tolist()
    i = 0
    for element in path:
        index[i] = element[-4:]
        i = i + 1

    path_vessel = glob.glob("O:\Carlos\Data/vessel/ivus_" + index[0] + "\*.png")
    path_lumen = glob.glob("O:\Carlos\Data/lumen/ivus_" + index[0] + "\*.png")
    ct_scan, origin, spacing = load_img('O:\Carlos\Data\mhds\ivus_' + index[0] + '.mhd')
    images_l, images_vessel = load_label(index[0])
    index_vessel = get_indexes(path_vessel)
    index_lumen = get_indexes(path_lumen)
    images_lumen = delete_indexes(images_l, index_lumen, index_vessel)
    inputs = extend_images(ct_scan, index_vessel, number_images=2)
    index.remove('0000')

    for element in index:
        path_vessel = glob.glob("O:\Carlos\Data/vessel/ivus_" + element + "\*.png")
        path_lumen = glob.glob("O:\Carlos\Data/lumen/ivus_" + element + "\*.png")
        ct_scan, origin, spacing = load_img('O:\Carlos\Data\mhds\ivus_' + element + '.mhd')
        images_l, images_v = load_label(element)
        index_vessel = get_indexes(path_vessel)
        index_lumen = get_indexes(path_lumen)
        images_l2 = delete_indexes(images_l, index_lumen, index_vessel)
        images_vessel = np.append(images_vessel, images_v, axis=0)
        images_lumen = np.append(images_lumen, images_l2, axis=0)
        inputs = np.append(inputs, extend_images(ct_scan, index_vessel, number_images=2), axis=0)

    labels = images_lumen[:, :, :, 0] / 255 + 2 * images_vessel[:, :, :, 0] / 255
    labels.astype('float32')
    return inputs, labels