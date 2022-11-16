import numpy as np
import math


def get_indexes(path):
    'Given a path, this function gives you the indexes of the labels corresponding to the original images'
    index = np.arange(len(path)).tolist()
    i = 0
    for element in path:
        index[i] = int(element[-8:-4])
        i = i + 1
    return index

def delete_indexes(images_lumen, index_lumen, index_vessel):
    'This function takes the images of the lumen and removes these indexes that are not in the vessel'
    # Should be change; not deleted, but created
    i = 0
    for e in index_lumen:
        if e not in index_vessel:
            images_lumen = np.delete(images_lumen, i, axis=0)
        i = i + 1
    return images_lumen

def extend_images(ct_scan, index_vessel, number_images = 0):
    'This function takes the original sample and turns it into an RGB image'
    i = 0
    until = len(index_vessel)
    for element in index_vessel:
        if (i > until * (number_images + 1) - 1):
            break
        for j in range(number_images):
            index_vessel.insert(i + j + 1, index_vessel[i] + j + 1)
        i = i + j + 2
    ct_scan = ct_scan[index_vessel]
    until = math.ceil(len(index_vessel) / (number_images + 1))
    shape = (math.ceil(ct_scan.shape[0] / (number_images + 1)),) + (ct_scan.shape[1],) + (ct_scan.shape[2],)
    image_modified = np.zeros((shape + (number_images + 1,)))
    for k in range(until):
        image_modified[k, :, :, :] = np.stack((ct_scan[k, :, :], ct_scan[k + 1, :, :], ct_scan[k + 2, :, :]), axis=-1)
        k += 3
    return image_modified
