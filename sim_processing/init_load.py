import numpy as np
# import dicom
# import scipy
import pydicom
import copy
import nibabel
import os
import suspect
import struct
import pickle
from scipy import sparse
# import matplotlib.pyplot as plt
# from dicompylercore import dicomparser

########################################################################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################################### AUTOMC LOADING  ###################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
########################################################################################################################

###############################
# GENERAL MULTI-PURPOSE TOOLS #
###############################


def load_dict_from_pkl(fp_pkl):
    """
    Input:      fp_pkl = file path of pickle file to load
    Output:     load_dict = loaded pkl file
    Summary:    Simple loading of pickle files.
    """

    load_dict = pickle.load(open(fp_pkl, "rb"))

    return load_dict


def save_dict_2_pkl(fp_pkl, load_dict):
    """
    Input:      fp_pkl = file path of pickle file to be saved (including name)
                load_dict = dictionary data to be saved into pickle file
    Output:     Pickle file saved to location
    Summary:    Simple loading of pickle files.
    Notes:      protocol = 1 was selected to solve errors in saving dicom files within pkl
    """

    f = open(fp_pkl, "wb")
    pickle.dump(load_dict, f, protocol=1)
    f.close()


def create_flat_sparse(np_array):
    """
    Input:      np_array = numpy array to flatten and convert to sparse array
    Output:     f_sp_array = flattened sparse array
    Summary:    This flattens and converts numpy arrays into sparse arrays. It greatly reduces the sizes of 3D grids
                from sims for saving.
    """

    f_sp_array = copy.deepcopy(np_array)

    f_sp_array = f_sp_array.flatten()
    f_sp_array = sparse.csr_matrix(f_sp_array)

    return f_sp_array


def unravel_data(csr_array, shape):
    """
    Input:      csr_array = flattened sparse array
                shape = shape to unravel to
    Output:     load_dict = of pickle file to load loaded pkl file
    Summary:    This de-flattens and converts spare arrays to numpy arrays.
    """

    map_data = copy.deepcopy(csr_array)

    map_data = map_data.toarray()
    map_data = map_data.reshape(shape)

    return map_data


def create_mask(base_map, masking_value=1):
    """
    Input:      base_map = map to make mask from, masking_value = percentage bound on mask
    Output:     mask = int mask
    Summary:    Creates a mask from a base map with a predefined masking value of 1%
    """

    mask = copy.deepcopy(base_map)

    mask = 100 * (mask / np.max(mask))
    mask[mask < masking_value] = 0
    mask[mask > (masking_value - 0.001)] = 1

    return mask


def resamp_mat_2_mat(ref_matrix, tar_matrix):
    """
    Input:      ref_matrix = matrix to match
                tar_matrix = matrix to alter
    Output:     matrix_res = tar_matrix resized to dimensions of ref_matrix
    Summary:    Creates new matrix where tar_matrix is resized to the dimensions of ref_matrix
    """

    matrix_res = copy.deepcopy(tar_matrix)

    matrix_res = matrix_res.resample(ref_matrix.row_vector,
                                     ref_matrix.col_vector,
                                     ref_matrix.shape,
                                     ref_matrix.centre + ref_matrix.voxel_size / 2,
                                     ref_matrix.voxel_size)

    return matrix_res


#############################
# GENERAL MAP LOADING TOOLS #
#############################

def create_lett_patient_dict(fp_main):
    """
    Input:     fp_main = path to full AUTOMC output
    # Output:    dictionary of als and dcm maps
    # Summary:   Loads als and dcm scored maps for additional LETt sims run for the LET uncertainty paper.
    #            Do not use for any other sims.
    """

    dcm_maps = create_dcm_maps(fp_main)
    print('DICOM Maps Loaded')
    als_maps = create_als_lett_maps(fp_main)
    print('AUTOMC Maps Loading')

    dict_all = {'dicom': dcm_maps, 'als': {}, 'shape': dcm_maps['CT'].shape}

    print('Processing AUTOMC Maps')

    # Resizing to CT size, flattening and sparsing

    for key in als_maps.keys():
        dict_all['als'][key] = resamp_mat_2_mat(dict_all['dicom']['CT'], als_maps[key])
        dict_all['als'][key] = create_flat_sparse(dict_all['als'][key])
        print('AUTOMC Processing: Completed', key)

    dict_all['dicom']['total_dose'] = resamp_mat_2_mat(dict_all['dicom']['CT'], dcm_maps['total_dose'])

    dict_all['dicom']['total_dose'] = create_flat_sparse(dict_all['dicom']['total_dose'])
    dict_all['dicom']['CT'] = create_flat_sparse(dict_all['dicom']['CT'])

    return dict_all


def create_letdprimmed_pat_dict(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dictionary of als and dcm maps
    Summary:    Loads als and dcm scored maps for additional letd prim med sims run for the LET uncertainty paper.
    Notes:      Do not use for any other sims.
    """

    dcm_maps = create_dcm_maps(fp_main)
    print('DICOM Maps Loaded')
    als_maps = create_als_letdprimmed_maps(fp_main)
    print('AUTOMC Maps Loading')

    dict_all = {'dicom': dcm_maps, 'als': {}, 'shape': dcm_maps['CT'].shape}

    print('Processing AUTOMC Maps')

    # Resizing to CT size, flattening and sparsing

    for key in als_maps.keys():
        dict_all['als'][key] = resamp_mat_2_mat(dict_all['dicom']['CT'], als_maps[key])
        dict_all['als'][key] = create_flat_sparse(dict_all['als'][key])
        print('AUTOMC Processing: Completed', key)

    dict_all['dicom']['total_dose'] = resamp_mat_2_mat(dict_all['dicom']['CT'], dcm_maps['total_dose'])

    dict_all['dicom']['total_dose'] = create_flat_sparse(dict_all['dicom']['total_dose'])
    dict_all['dicom']['CT'] = create_flat_sparse(dict_all['dicom']['CT'])

    return dict_all


def create_patient_orig_dict(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dictionary of als and dcm maps
    Summary:    Loads als and dcm scored maps for original sims run for the LET uncertainty paper.
    Notes:      Do not use for any other sims.
    """

    dcm_maps = create_dcm_maps(fp_main)
    print('DICOM Maps Loaded')
    als_maps = create_als_orig_maps(fp_main)
    print('AUTOMC Maps Loading')

    dict_all = {'dicom': dcm_maps, 'als': {}, 'shape': dcm_maps['CT'].shape}

    print('Processing AUTOMC Maps')

    # Resizing to CT size, flattening and sparsing

    for key in als_maps.keys():
        dict_all['als'][key] = resamp_mat_2_mat(dict_all['dicom']['CT'], als_maps[key])
        dict_all['als'][key] = create_flat_sparse(dict_all['als'][key])
        print('AUTOMC Processing: Completed', key)

    dict_all['dicom']['total_dose'] = resamp_mat_2_mat(dict_all['dicom']['CT'], dcm_maps['total_dose'])

    dict_all['dicom']['total_dose'] = create_flat_sparse(dict_all['dicom']['total_dose'])
    dict_all['dicom']['CT'] = create_flat_sparse(dict_all['dicom']['CT'])

    return dict_all


####################################################################################################################
#################################################### DICOM Functions ###############################################
####################################################################################################################
#######################
# DICOM Loading Tools #
#######################


def create_dcm_maps(fp_main):
    """
    Input:     fp_main = path to full AUTOMC output
    Output:    loads standard dicom files for a rt patient
    Summary:   loads total dose from dicom field files
    """

    list_dict = ['total_dose', 'CT']

    dict_dcm = dict.fromkeys(list_dict, 0)

    dict_dcm['total_dose'] = create_dcm_total_dose(fp_main)

    dict_dcm['CT'] = create_dcm_ct(fp_main)

    dict_dcm['RS'] = create_dcm_rs(fp_main)

    return dict_dcm


def create_dcm_rs(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     pyd_rs_data = raw load of DICOM RS file
    Summary:    Loads DICOM RS file by searching folder.
    Notes:      Should catch error of more than one RS file in folder
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


def create_dcm_total_dose(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     imagebase file of dicom total dose
    Summary:    loads total dose from dicom field files
    """

    # Find CT filename
    file_list = os.listdir(fp_main)

    str_dose_start = 'RD.'
    str_dose_end = '.dcm'

    list_fp_dcm_dose = []

    for file_name in file_list:
        if file_name.startswith(str_dose_start) and file_name.endswith(str_dose_end):
            list_fp_dcm_dose.append(fp_main + file_name)
            print(file_name)

    dose_counter = 0

    for fp_dcm_dose in list_fp_dcm_dose:
        if dose_counter == 0:
            dcm_total_dose = create_dcm_field_dose(fp_dcm_dose)
            dose_counter +=1
        else:
            dose_field = create_dcm_field_dose(fp_dcm_dose)
            dcm_total_dose = dcm_total_dose + dose_field

    return dcm_total_dose


def create_dcm_field_dose(fp_dicom_dose):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     imagebase file of dicom dose
    Summary:    Loads dicom dose field file
    """

    dose_1 = pydicom.read_file(fp_dicom_dose)

    raw_eclip_dose = dose_1.pixel_array * dose_1.DoseGridScaling

    eclip_dose_row_vec = np.array(dose_1.ImageOrientationPatient[:3])
    eclip_dose_col_vec = np.array(dose_1.ImageOrientationPatient[3:])
    eclip_dose_pos = np.array(dose_1.ImagePositionPatient)
    eclip_dose_slice_thic = dose_1.GridFrameOffsetVector[1] - dose_1.GridFrameOffsetVector[0]
    eclip_dose_vox_spac = list(dose_1.PixelSpacing)
    eclip_dose_vox_spac.append(eclip_dose_slice_thic)

    eclip_dose_trans = suspect.transformation_matrix(eclip_dose_row_vec, eclip_dose_col_vec, eclip_dose_pos, eclip_dose_vox_spac)

    eclipse_dose = suspect.base.ImageBase(raw_eclip_dose, eclip_dose_trans)

    return eclipse_dose


def create_dcm_ct(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     DICOM ct file
    Summary:    Loads dicom ct file by searching folder.
    Note:       Will bug if more than one set of DICOM CT files
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

####################################################################################################################
###################################################### ALS Functions ###############################################
####################################################################################################################
#########################################
### FUNCTIONS TO LOAD AUTOMC MAP SETS ###
#########################################


def create_als_letdprimmed_maps(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dictionary of als maps
    Summary:    Loads als scored maps for additional letd prim med sims run for the LET uncertainty paper.
    Notes:      Do not use for any other sims.
    """

    list_letdprimmed_scoring = ['letdprimmed']
    list_dose_scoring = ['total_dosetowater', 'total_dosetomed']
    list_image = ['CT', 'physdens']
    list_dict = list_dose_scoring + list_letdprimmed_scoring + list_image

    dict_allscore = dict.fromkeys(list_dict, 0)

    # Total dose

    dict_allscore['total_dosetowater'], dict_allscore['total_dosetomed'] = create_als_total_dose(fp_main)

    # Total LETt

    for score in list_letdprimmed_scoring:
        dict_allscore[score] = create_als_total_letd(fp_main, score)

    # CT and Physdens

    dict_allscore['CT'] = create_als_ct(fp_main)
    dict_allscore['physdens'] = create_als_physdens(fp_main)

    return dict_allscore


def create_als_lett_maps(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dictionary of als maps
    Summary:    Loads als scored maps for additional lett sims run for the LET uncertainty paper.
    Notes       Do not use for any other sims.
    """

    list_lett_scoring = ['lett']
    list_dose_scoring = ['total_dosetowater', 'total_dosetomed']
    list_image = ['CT', 'physdens']
    list_dict = list_dose_scoring + list_lett_scoring + list_image

    dict_allscore = dict.fromkeys(list_dict, 0)

    # Total dose

    dict_allscore['total_dosetowater'], dict_allscore['total_dosetomed'] = create_als_total_dose(fp_main)

    # Total LETt

    for score in list_lett_scoring:
        dict_allscore[score] = create_als_total_lett(fp_main, score)

    #CT and Physdens

    dict_allscore['CT'] = create_als_ct(fp_main)
    dict_allscore['physdens'] = create_als_physdens(fp_main)

    return dict_allscore


def create_als_orig_maps(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dictionary of als maps
    Summary:    Loads als maps for original sims run for the LET uncertainty paper.
    Notes:      Do not use for any other sims.
    """

    list_letd_scoring = ['letd', 'letdpost', 'letdpart', 'letdmed', 'letdprim']
    list_dose_scoring = ['total_dosetowater', 'total_dosetomed']
    list_image = ['CT', 'physdens', 'spr']
    list_dict = list_dose_scoring + list_letd_scoring + list_image

    dict_allscore = dict.fromkeys(list_dict, 0)

    # Total dose

    dict_allscore['total_dosetowater'], dict_allscore['total_dosetomed'] = create_als_total_dose(fp_main)

    #letd_mask = create_mask(dict_allscore['total_dosetowater'], masking_value=1)

    letd_mask = 1 # removing mask from data

    # Total LETd (all scoring options)

    for score in list_letd_scoring:
        dict_allscore[score] = letd_mask * create_als_total_letd(fp_main, score)

    #CT, Physdens and SPR

    dict_allscore['CT'] = create_als_ct(fp_main)
    dict_allscore['physdens'] = create_als_physdens(fp_main)
    dict_allscore['spr'] = create_als_spr(fp_main)

    # Creating LETd mass

    dict_allscore['letdmass'] = letd_mask * (create_als_letd_mass(dict_allscore['letdmed'], dict_allscore['physdens']))

    return dict_allscore


###########
### ALS ###
###########
###########################################################################
### FUNCTIONS TO LOAD AUTOMC CT, Stopping Power Ratio, Physical Density ###
###                               AND MISC                              ###
###########################################################################


def create_als_letd_mass(als_letd_med, als_physdens):
    """
    Input:      als_letd_med = als file for LETd to medium
                als_physdens = als file for physical density
    Output:     Physical density file
    Summary:    Creates letd to mass density from letd to medium and a physical density file.
    """

    letd_med = copy.deepcopy(als_letd_med)
    physdens = copy.deepcopy(als_physdens)

    physdens = resamp_mat_2_mat(letd_med, physdens)

    with np.errstate(divide='ignore'):
        letd_mass = np.divide(letd_med, physdens)

    return letd_mass


def create_als_physdens(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     Physical density file
    Summary:    Loads physical density
    """

    file_list = os.listdir(fp_main)

    str_physdens = '_PHYSDENS.hdr'

    for file_name in file_list:
        if file_name.endswith(str_physdens):
            fp_als_physdens = fp_main + file_name

    als_physdens_trans = create_als_trans(fp_als_physdens)
    als_physdens_perm = create_als_perm(fp_als_physdens)

    als_physdens = suspect.base.ImageBase(als_physdens_perm, als_physdens_trans)

    return als_physdens


def create_als_spr(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     Stopping Power Ratio file (SPR)
    Summary:    Loads SPR file created by AUTOMC
    """

    file_list = os.listdir(fp_main)

    str_spr = '_SPR.hdr'

    for file_name in file_list:
        if file_name.endswith(str_spr):
            fp_als_spr = fp_main + file_name

    als_spr_trans = create_als_trans(fp_als_spr)
    als_spr_perm = create_als_perm(fp_als_spr)

    als_spr = suspect.base.ImageBase(als_spr_perm, als_spr_trans)

    return als_spr


def create_als_ct(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     als_CT = CT file
    Summary:    Finds CT.hdr created by AUTOMC in main simulation folder
    """

    # Find CT filename
    file_list = os.listdir(fp_main)

    str_ct = '_CT.hdr'

    ct_counter = 0
    for root, dirs, files in os.walk(fp_main):
        for file in files:
            if file.endswith(str_ct):
                ct_counter += 1

    if ct_counter == 1:
        for name in file_list:
            if name[-7:] == '_CT.hdr':
                fp_CT = fp_main + name

    else:
        print('There is either no or multiple files ending with "_CT.hdr"')

    als_CT_trans = create_als_trans(fp_CT)
    als_CT_perm = create_als_perm(fp_CT)

    als_CT = suspect.base.ImageBase(als_CT_perm, als_CT_trans)

    return als_CT


###########
### ALS ###
###########
###################################################################
### FUNCTIONS TO LOAD TOTAL (ALL FIELDS) FOR AUTOMC SIMULATIONS ###
###################################################################

def create_als_total_lett(fp_main, str_init_lett):
    """
    Input:      fp_main = path to full AUTOMC output, str_init_lett = name of lett flavour
    Output:     total_lett
    Summary:    Calculates total lett from fp_main and the start string of the lett method
    Notes:      This takes into account how AUTOMC allocates particles between fields
    """

    # Loading physden
    als_physdens = create_als_physdens(fp_main) # creating patient als physdens file
    als_total_dose_w, _ = create_als_total_dose(fp_main)
    als_physdens = resamp_mat_2_mat(als_total_dose_w, als_physdens) # Need to resize physden

    list_field_names, list_fp_fields = create_als_field_names(fp_main) # finding names and fps of fields

    for fp_field, field_name in zip(list_fp_fields, list_field_names):

        _, num_energylayer = create_als_field_detail(fp_main, field_name)

        list_lett_num, list_lett_den = create_als_let_list(str_init_lett, num_energylayer)

        field_dose_w, _ = create_als_field_dose(fp_field, field_name)

        field_dose = field_dose_w

        field_lett = create_als_field_let(fp_field, list_lett_num, list_lett_den)

        flu_cor = field_dose * als_physdens / field_lett

        field_lett_cor = field_lett * flu_cor

        if fp_field == list_fp_fields[0]:
            als_total_lett_num = field_lett_cor
            tot_flu_cor = flu_cor
        else:
            als_total_lett_num= als_total_lett_num + field_lett_cor
            tot_flu_cor = tot_flu_cor + flu_cor

    total_lett = als_total_lett_num / tot_flu_cor

    total_lett = np.nan_to_num(total_lett)

    return total_lett


def create_als_total_letd_comb(fp_main, str_init_letd):
    """
    Input:      fp_main = path to full AUTOMC output, str_init_letd = name of letd flavour
    Output:     total_letd = total letd for all fields in AUTOMC output
    Summary:    Calculates total letd from fp_main and giving start string of the letd scoring method.
    Notes:      This method is used when AUTOMC has already combined the fields
    """

    list_field_names, list_fp_fields = create_als_field_names(fp_main)

    letd_if_case = ['letdmed', 'letdpart', 'letdprimmed']

    for fp_field, field_name in zip(list_fp_fields, list_field_names):

        if str_init_letd in letd_if_case:
            _, field_dose = create_als_field_dose(fp_field, field_name)
        else:
            field_dose, _ = create_als_field_dose(fp_field, field_name)

        fp_als = fp_field + str_init_letd + '_COMBINED-LET.hdr'

        field_letd = create_als_field_let_comb(fp_field, fp_als)

        if fp_field == list_fp_fields[0]:
            als_total_letd_num = field_dose * field_letd
        else:
            als_field_letd = field_dose * field_letd
            als_total_letd_num = als_total_letd_num + als_field_letd

    if str_init_letd in letd_if_case:
        _, total_dose = create_als_total_dose(fp_main)
    else:
        total_dose, _ = create_als_total_dose(fp_main)

    total_letd = als_total_letd_num / total_dose

    total_letd = np.nan_to_num(total_letd)

    return total_letd


def create_als_total_letd(fp_main, str_init_letd):
    """
    Input:      fp_main = path to full AUTOMC output, str_init_letd = name of letd flavour
    Output:     total_letd = total letd for all fields in AUTOMC output
    Summary:    Calculates total letd from fp_main and giving start string of the letd scoring method.
    """

    list_field_names, list_fp_fields = create_als_field_names(fp_main)

    letd_if_case = ['letdmed', 'letdpart', 'letdprimmed']

    for fp_field, field_name in zip(list_fp_fields, list_field_names):

        _, EnergyLayer = create_als_field_detail(fp_main, field_name)

        list_letd_num, list_letd_den = create_als_let_list(str_init_letd, EnergyLayer)

        if str_init_letd in letd_if_case:
            _, field_dose = create_als_field_dose(fp_field, field_name)
        else:
            field_dose, _ = create_als_field_dose(fp_field, field_name)

        field_letd = create_als_field_let(fp_field, list_letd_num, list_letd_den)

        if fp_field == list_fp_fields[0]:
            als_total_letd_num = field_dose * field_letd
        else:
            als_field_letd = field_dose * field_letd
            als_total_letd_num = als_total_letd_num + als_field_letd

    if str_init_letd in letd_if_case:
        _, total_dose = create_als_total_dose(fp_main)
    else:
        total_dose, _ = create_als_total_dose(fp_main)

    total_letd = als_total_letd_num / total_dose

    total_letd = np.nan_to_num(total_letd)

    return total_letd


def create_als_total_dose(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     total_dose2water & total_dose2med = total absorbed dose scored to med & water for patient
    Summary:    Adds dose to water and dose to medium from fields in the AUTOMC output folder
    """

    list_field_names, list_fp_fields = create_als_field_names(fp_main)

    for field_name, fp_field in zip(list_field_names, list_fp_fields):

        if fp_field == list_fp_fields[0]:
            total_dose2water, total_dose2med = create_als_field_dose(fp_field, field_name)
        else:
            field_dose2water, field_dose2med = create_als_field_dose(fp_field, field_name)
            total_dose2water = total_dose2water + field_dose2water
            total_dose2med = total_dose2med + field_dose2med

    total_dose2water = np.nan_to_num(total_dose2water)
    total_dose2med = np.nan_to_num(total_dose2med)

    return total_dose2water, total_dose2med


###########
### ALS ###
###########
################################################
### FUNCTIONS TO LOAD FIELDS FOR AUTOMC SIMS ###
################################################


def create_als_field_let_comb(fp_als):
    """
    Input:      fp_als = path to als let file
    Output:     als_let_field  = LET of a field
    Summary:    Creates LET for a single field for a given let scoring method (via the list_letd)
    Notes:      Chance of bug if als file doesn't have coordinates required for create_als_trans
    """

    als_let = create_als_field_comb(fp_als)

    als_let_field = np.nan_to_num(als_let)

    return als_let_field


def create_als_field_let(fp_field, list_let_num, list_let_den):
    """
    Input:      fp_field = path to field
                list_let_num, list_let_den = lists of GATE splits (num and den)
    Output:     als_let_field  = LET of a field
    Summary:    Creates LET for a single field for a given let scoring method (via the list_letd)
    """

    als_let_num = np.nan_to_numn(create_als_field_map(fp_field, list_let_num))
    als_let_den = np.nan_to_num(create_als_field_map(fp_field, list_let_den))

    als_let_field = np.nan_to_num(als_let_num / als_let_den)

    return als_let_field


def create_als_field_map(fp_field, list_map):  # loads and adds nums or dens together
    """
    Input:      fp_field = path to field,
                list_map = lists of GATE map splits
    Output:     als_map  = Score map for a field
    Summary:    Creates map for a single field from a group of splits
    """

    als_trans = create_als_trans(fp_field + list_map[0])

    temp_als_perm = create_als_perm((fp_field + list_map[0]))

    als_frac = suspect.base.ImageBase(temp_als_perm, als_trans)

    for name in list_map[1:]:
        temp_als_perm = create_als_perm(fp_field + name)
        temp_als_frac = suspect.base.ImageBase(temp_als_perm, als_trans)
        als_frac = als_frac + temp_als_frac

    als_map = np.nan_to_num(als_frac)

    return als_map


def create_als_field_dose(fp_field, str_field_name):
    """
    Input:      fp_main = path to full AUTOMC output, fp_field = path to field folder
                str_field_name = field name
    Output:     dose2water and dose2field = Dose(med) and Dose(water) of field
    Summary:    Creates dose to water and dose to medium for a single field
    """

    fp_dose2water = fp_field + 'total_COMBINED-DoseToWater.hdr'
    fp_dose2med = fp_field + 'total_COMBINED-Dose.hdr'
    fp_main = os.path.dirname(os.path.dirname(fp_field)) + '/'
    # my convention has always been to put '/' at end of folder this causes issues as os.path.dirname will only remove
    # this and not go up the next level. I have to apply os.path.dirname twice.

    G2Gy, _ = create_als_field_detail(fp_main, str_field_name)  # finding GATE dose factor

    dose2water = np.nan_to_num((G2Gy/100) * create_als_field_comb(fp_dose2water))
    dose2med = np.nan_to_num((G2Gy/100) * create_als_field_comb(fp_dose2med))

    return dose2water, dose2med


def create_als_field_comb(fp_als):
    """
    Input:      fp_als = path to als combined file
    Output:     als_field  = map of a field
    Summary:    Creates LET for a single field for a given let scoring method (when it's combined by AUTOMC)
    Notes:      Chance of bug if als file doesn't have coordinates required for create_als_trans
    """

    als_trans = create_als_trans(fp_als)
    als_perm = create_als_perm(fp_als)
    als_field = suspect.base.ImageBase(als_perm, als_trans)

    return als_field


###########
### ALS ###
###########
################################################
### FUNCTIONS TO LOAD COMPONENTS OF ALS FILE ###
################################################


def create_als_trans(fp_als):
    """
    Input:      fp_als = path to als file
    Output:     als_file_trans = als transform to match
    Summary:    This function creates transform file used to create ImageBase from loading als file.
    """

    # LOAD
    # loading data using custom nibabel package
    als_file = nibabel.load(fp_als)

    # loading total combined dose for the field so to copy it's location tag
    # (this sort of fixes a bug in AUTOMC where there is no header in the splits)

    if fp_als.endswith('_CT.hdr') or fp_als.endswith('_PHYSDENS.hdr') or fp_als.endswith('_SPR.hdr'):
        fp_field_hdr = fp_als
    else:
        fp_field = os.path.dirname(fp_als) + '/' # gives containing folder
        fp_field_hdr = fp_field + 'total_COMBINED-Dose.hdr'

    # finding position of last row position
    with open(fp_field_hdr, 'rb') as fin:
        fin.seek(148)
        des = struct.unpack("80s", fin.read(80))[0].decode()

    # position
    pos = np.array([float(des[29:38]), float(des[39:48]), float(des[49:58])])
    als_pos = [pos[1], -(pos[0]), -(pos[2])]

    # perm
    als_file_perm = np.squeeze(als_file.get_data()).transpose(2, 0, 1)

    # transform
    als_file_trans = als_file.affine.copy()

    # fixing neurological/radiological difference in convention
    als_file_trans[0] *= -1

    # swapping the first two axes
    als_file_trans = als_file_trans[[1, 0, 2, 3]][:, [1, 0, 2, 3]]

    # convert from last row/slice to first row/slice
    row_width = als_file_trans[1, 1]
    row_count = als_file_perm.shape[1]

    slice_width = als_file_trans[2, 2]
    slice_count = als_file_perm.shape[0]

    als_pos[1] -= row_width * (row_count - 1)
    als_pos[2] -= slice_width * (slice_count - 1)

    als_file_trans[:3, 3] = als_pos

    return als_file_trans


def create_als_perm(fp_als):
    """
    Input:      fp_als = file path of als file
    Output:     als_file_perm = corrected als file (z, x, y)
    Summary:    loading data using custom nibabel package
    """

    als_file = nibabel.load(fp_als)  # loading

    # perm
    als_file_perm = np.squeeze(als_file.get_data()).transpose(2, 0, 1)

    return als_file_perm


###########
### ALS ###
###########
####################################################
### FUNCTIONS TO LOAD DETAILS FROM AUTOMC / GATE ###
####################################################


def create_als_field_names(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     list_field_names = list of field names, list_fp_fields = list of paths of fields
    Summary:    Finds field names and paths from main AUTOMC output folder
    Notes:      This assumes that field folders start with output and field name starts at 10th char
    """
    file_list = os.listdir(fp_main)

    list_field_names = []
    list_fp_fields = []

    for name in file_list:
        if name[:6] == 'output':
            list_fp_fields.append(fp_main + name + '/')
            list_field_names.append(name[10:])

    return list_field_names, list_fp_fields


def create_als_field_detail(fp_main, str_field_name):
    """
    Input:      fp_main = path to full AUTOMC output, str_field_name = field name
    Output:     G2Gray = Factor to convert GATE output to absorbed dose per field
                num_EnergyLayer = Number of energy layers
    Summary:    Finds gate2gy and numbers of energy layers of a field from AUTOMC_data_{field_name}_partB.mat
    """

    f = open(fp_main + 'AUTOMC_data_' + str_field_name + '_partB.mat', "r")
    lines = f.readlines()

    ind_G2GY = lines.index('# name: GateToGyScaleFactor\n') + 2 # gate to gy factor
    ind_EnergyLayer = lines.index('# name: primariesPerLayer\n') + 3 # number of energy layers

    G2Gray = float(lines[ind_G2GY])
    num_EnergyLayer = int(lines[ind_EnergyLayer][11:])

    return G2Gray, num_EnergyLayer


###########
### ALS ###
###########
##############################################
### FUNCTIONS TO PRODUCE LISTS FOR LOADING ###
##############################################


def create_als_trackend_list(str_init_te, num_energylayer):
    """
    Input:      str_init_te = inital trackend str (p_track_end or track_end)
    Output:     list_te_split = a list of trackend splits
    Summary:    Creates lists of the trackend split files for loading by taking initial string and number of energy layers
    """

    list_layers = np.arange(1, num_energylayer+1) # Creates lists of numbers up to total number of energy layers

    list_te_split = [] # creates empty list to append let split strings

    # LET String Build
    # handles split string build for descriptor portion of split string

    if str_init_te == 'p_track_end':
        str_split_nam = str_init_te + '_primary'
    else:
        str_split_nam = str_init_te

    # List of trackEnd Split Strings

    for num in list_layers:

        if num < 10:  # handles numbering system of let splits
            str_split_num = '0' + str(num)
        else:
            str_split_num = str(num)

        str_stop = '-Stop.hdr'
        str_te_tag = str_split_nam + str_split_num + str_stop  # adding str components together
        list_te_split.append(str_te_tag)

    return list_te_split


def create_als_let_list(str_init_let, num_energylayer):
    """
    Input:      str_init_let = let flavour initial string, num_EnergyLayer = number of energy layers in field
    Output:     list_let_split_num & list_let_split_den = lists of let numerator and denominator splits
    Summary:    Creates lists for numerator and denominator of let files in GATE by taking the initial let str (letd,
                letdmed,letdprim etc) and number of energy layers.
    """

    list_layers = np.arange(1, num_energylayer+1)  # Creates lists of numbers up to total number of energy layers

    list_let_split = []  # creates empty list to append let split strings

    # LET String Build
    # handles GATEs naming of splits etc

    if str_init_let == 'letdmed' or str_init_let == 'letdpart' or str_init_let == 'letdprimmed':
        str_split_nam = '-doseAveraged'
    elif str_init_let == 'lett':
        str_split_nam = '-trackAveraged-letToWater'
    else:
        str_split_nam = '-doseAveraged-letToWater'

    # List of LET Split Strings

    for num in list_layers:

        if num < 10:  # handles numbering system of let splits
            str_split_num = '0' + str(num)
        else:
            str_split_num = str(num)

        str_let_tag = str_init_let + str_split_num + str_split_nam  # adding str components together
        list_let_split.append(str_let_tag)

    # Creating lists of let split numerators and denominator strings
    list_let_split_num = [s + '-numerator.hdr' for s in list_let_split]  # adds num tag to list
    list_let_split_den = [s + '-denominator.hdr' for s in list_let_split]  # adds denom tag to list

    return list_let_split_num, list_let_split_den