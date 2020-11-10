from skimage import draw
import pydicom as pyd
import numpy as np
import suspect
import os
import copy
from skimage import draw
from skimage import feature
from skimage import filters
from scipy import sparse
import pickle
import matplotlib.pyplot as plt
import copy


########################################################################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
######################################################## RS LOADING  ###################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
########################################################################################################################

###############################
# GENERAL MULTI-PURPOSE TOOLS #
###############################


def save_dict_2_pkl(fp_pkl, load_dict):
    """
    Input:      fp_pkl = file path of pickle file to be saved (including name)
                load_dict = dictionary data to be saved into pickle file
    Output:     Pickle file saved to location
    Summary:    Loading of pickle files.
    Notes:      Protocol = 1 was selected to solve errors in saving dicom files within pkl. This has now stopped
                working, I believe due to update to python 3.7, RS files have now been removed from the
                dictionaries.
    """

    f = open(fp_pkl, "wb")
    pickle.dump(load_dict, f, protocol=1)
    f.close()


def load_dict_from_pkl(fp_pkl):
    """
    Input:    fp_pkl = file path of pickle file to load.
    Output:   load_dict = loaded pkl file.
    Summary:  Simple loading of pickle files.
    """
    load_dict = pickle.load(open(fp_pkl, "rb"))

    return load_dict


def create_flat_sparse(np_array):
    """
    Input:    np_array = numpy array to flatten and convert to sparse array
    Output:   f_sp_array = flattened sparse array
    Summary:  This flattens and converts numpy arrays into sparse arrays. It greatly reduces the sizes of 3D grids
             from sims for saving.
    """
    f_sp_array = copy.deepcopy(np_array)

    f_sp_array = f_sp_array.flatten()
    f_sp_array = sparse.csr_matrix(f_sp_array)

    return f_sp_array


def unravel_data(csr_array, shape):
    """
    Input:    csr_array = flattened sparse array
              shape = shape to unravel to
    Output:   load_dict = of pickle file to load loaded pkl file
    Summary:  This deflattens and converts spare arrays to numpy arrays.
    """
    map_data = copy.deepcopy(csr_array)

    map_data = map_data.toarray()
    map_data = map_data.reshape(shape)

    return map_data


def create_mask(base_map, masking_value=1):
    """
    Input:    base_map = map to make mask from, masking_value = percentage bound on mask
    Output:   mask = int mask
    Summary:  Creates a mask from a base map with a predefined masking value of 1%
    """

    mask = copy.deepcopy(base_map)

    mask = 100 * (mask / np.max(mask))
    mask[mask < masking_value] = 0
    mask[mask > (masking_value - 0.001)] = 1

    return mask


def resamp_mat_2_mat(ref_matrix, tar_matrix):
    """
    Input:    ref_matrix = matrix to match, tar_matrix = matrix to alter
    Output:   mask = int mask
    Summary:  Creates a mask from a base map with a predefined masking value of 1%
    """
    matrix_res = copy.deepcopy(tar_matrix)

    matrix_res = matrix_res.resample(ref_matrix.row_vector,
                                     ref_matrix.col_vector,
                                     ref_matrix.shape,
                                     ref_matrix.centre + ref_matrix.voxel_size / 2,
                                     ref_matrix.voxel_size)

    return matrix_res


#############################
# GENERAL DCM LOADING TOOLS #
#############################


def create_dcm_ct(fp_main):
    """
    Input:     fp_main = path to full AUTOMC output
    Output:    dicom ct file
    Summary:   Loads dicom ct file by searching folder.
    Note:      Will bug if more than one set of DICOM CT files
    """

    file_list = os.listdir(fp_main)

    ct_counter = 0

    while ct_counter == 0:
        for name in file_list:
            if name[:2] == 'CT' and name[-4:] =='.dcm':
                fp_dicom_ct = fp_main + name
                ct_counter += 1

    dicom_ct = suspect.image.load_dicom_volume(fp_dicom_ct)

    return dicom_ct


def create_dcm_rs(fp_main):
    """
    Input:     fp_main = path to full AUTOMC output
    Output:    pyd_rs_data = raw load of DICOM RS file
    Summary:   Loads DICOM RS file by searching folder.
    Note:      Should catch error of more than one RS file in folder
    """
    file_list = os.listdir(fp_main)

    str_start = 'RS.'
    str_end = '.dcm'
    fp_rs = ''

    rs_counter = 0

    for file in file_list:
        if file.startswith(str_start) and file.endswith(str_end):
            rs_counter += 1

    if rs_counter == 1:
        for file in file_list:
            if file.startswith(str_start) and file.endswith(str_end):
                fp_rs = fp_main + file

    else:
        print('There is either no or multiple RS files"')

    pyd_rs_data = pyd.read_file(fp_rs)

    return pyd_rs_data


##########################
# RS DICT BUILDING TOOLS #
##########################


def create_dict_rs_dcm(pyd_rs_data):
    """
    Input:      pyd_rs_data = rs data loaded by pydicom (function = create_dcm_rs)
    Output:     dict_rs_contours = a dictionary of each contour with its associated data for 'color', 'number', 'name'
                and 'contours' (contour point data)
    Summary:    Creates a dictionary of contours from DICOM RS data loaded by pydicom
    Note:       (function = create_dcm_rs)
    """
    dict_rs_contours = {}

    for i in range(len(pyd_rs_data.ROIContourSequence)):
        contour = {}
        contour['color'] = pyd_rs_data.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = pyd_rs_data.ROIContourSequence[i].ReferencedROINumber
        contour['name'] = pyd_rs_data.StructureSetROISequence[i].ROIName
        assert contour['number'] == pyd_rs_data.StructureSetROISequence[i].ROINumber

        contour['contours'] = []

        try:
            for j in np.arange(0, len(pyd_rs_data.ROIContourSequence[i].ContourSequence)):

                loop_contour = pyd_rs_data.ROIContourSequence[i].ContourSequence[j].ContourData
                new_length = int(len(loop_contour) / 3)
                loop_contour = np.reshape(loop_contour, [new_length, 3])

                contour['contours'].append(loop_contour)

        except AttributeError:
            print('Contour ', contour['name'], contour['number'], 'has no contour data')

        dict_rs_contours[str(contour['name'])] = contour

        print('contour: ', i, 'complete')

    return dict_rs_contours


def create_dict_rs_pix(dcm_ct, dict_rs_contours):
    """
    Input:      dcm_ct = DICOM ct file
                dict_rs_contours = a dictionary of each contour with its associated data for 'color', 'number', 'name'
                and 'contours' (contour point data).
    Output:     dict_pix_contours = the dict_rs_contours plus a key holding the converted
                pixel location of the contour point.
    Summary:    Creates a dictionary from the dict_rs_contours but with an additional key
                giving the pixel locations of DICOM contour points
                (function = create_dcm_rs)
    """
    dict_pix_contours = copy.deepcopy(dict_rs_contours)

    # DICOM CT Parameters

    dcm_centre_mm = np.array(dcm_ct.centre)

    dcm_voxelsize = np.array(dcm_ct.voxel_size)

    dcm_shape = dcm_ct.shape
    dcm_shape = np.array([dcm_shape[1], dcm_shape[2], dcm_shape[0]])
    # nibabel stores headers ike voxel size in diff x,y,z order to the image (WTF!. For my santiy I am changing shape
    # from z, x, y to x,y,z

    dcm_centre_pix = np.round((dcm_shape/2)-0.5)
    dcm_centre_pix = dcm_centre_pix.astype(int)

    #dcm_cen_cor = dcm_centre_pix - dcm_centre_mm

    # Loop

    for key in dict_pix_contours:

        dict_pix_contours[key]['contours_pix'] = []

        print(dict_pix_contours[key]['name'])

        for s in np.arange(0, len(dict_pix_contours[key]['contours'])):

            con_slice = dict_pix_contours[key]['contours'][s]

            temp_slice = []

            for point in con_slice:

                x = int(((point[0] - dcm_centre_mm[0]) / dcm_voxelsize[0]) + dcm_centre_pix[0])
                y = int(((point[1] - dcm_centre_mm[1]) / dcm_voxelsize[1]) + dcm_centre_pix[1])
                z = int(((point[2] - dcm_centre_mm[2]) / dcm_voxelsize[2]) + dcm_centre_pix[2])

                temp_point = [z, x, y]  # NOTE I HAVE CHANGED THE ORDER HERE TO MATCH NIBABEL EAKS

                temp_slice.append(temp_point)

            dict_pix_contours[key]['contours_pix'].append(np.array(temp_slice))

    return dict_pix_contours


#################################
# MASK & OUTLINE BUILDING TOOLS #
#################################


def create_contour_mask(dcm_ct, contour_set):
    """
    Input:      dcm_ct = DICOM ct file
                contour_set = dictionary for single structure with a key for 'contours_pix'
    Output:     mask = mask created from contours_pix
    Summary:    Creates a dictionary for structure of contours point (for DICOM points ('contours') and
                pixel points('contour_pix').
    """

    mask = np.zeros(dcm_ct.shape)

    for sli in contour_set['contours_pix']:

        s_coord = sli[:, 0][0]
        c_coords = sli[:, 1]
        r_coords = sli[:, 2]

        f_c_coords, f_r_coords = draw.polygon(r_coords, c_coords)

        f_s_coords = np.ones(f_r_coords.shape) * s_coord
        f_s_coords = f_s_coords.astype(int)

        mask[f_s_coords, f_c_coords, f_r_coords] = 1

    return mask


def create_outline(mask_struct):
    """
    Input:      mask_struct = mask of structure (np array (non-flattened or sparsed))
    Output:     outline = outline of structure
    Summary:    Creates a dictionary for structure of contours point (for DICOM points ('contours') and pixel
                points('contour_pix').
    """
    mask_struct = copy.deepcopy(mask_struct)

    temp_length = np.arange(0, len(mask_struct))
    temp_array = np.zeros(mask_struct.shape)

    for temp_slice in temp_length:
        temp_outline = feature.canny(mask_struct[temp_slice], sigma=5)
        temp_outline = temp_outline.astype(float)
        temp_outline = filters.gaussian(temp_outline)
        temp_outline[temp_outline > 0.2] = 1
        temp_outline[temp_outline < 0.3] = 0

        temp_array[temp_slice] = temp_outline

    outline_contour = temp_array

    return outline_contour
