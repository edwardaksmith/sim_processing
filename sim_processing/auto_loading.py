from sim_processing import init_load as init
from sim_processing import rs_load as rs
import os as os

########################################################################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################################### AUTO LOADING  #####################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
########################################################################################################################


###########################################################################
########################## AUTO RS STRUCTURE LOAD  ########################
###########################################################################


def auto_outline_load(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dict_outline = dictionary of outlines held in 'outlines' with 'shape' for unravelling
                data.
    Summary:    This produces a dictionary of outlines made from the RS file in the main folder.
                The outlines are stored as flattened and sparsed arrays for compression. The
                shape for unravelling is given by the key 'shape.
    """
    dcm_ct = rs.create_dcm_ct(fp_main)
    dict_rs_pix = auto_rs_load(fp_main)

    dict_outline = {'outlines': {}, 'shape': dcm_ct.shape}

    for key in dict_rs_pix:

        dict_outline['outlines'][key] = 0

        temp_mask = rs.create_contour_mask(dcm_ct, dict_rs_pix[key])
        temp_outline = rs.create_outline(temp_mask)
        temp_outline = rs.create_flat_sparse(temp_outline)

        dict_outline['outlines'][key] = temp_outline

    return dict_outline


def auto_mask_load(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dict_mask = dictionary of masks held in 'masks' with 'shape' for unravelling
                data.
    Summary:    This produces a dictionary of masks made from the RS file in the main folder.
                The masks are stored as flattened and sparsed arrays for compression. The
                shape for unravelling is given by the key 'shape.
    """
    dcm_ct = rs.create_dcm_ct(fp_main)
    dict_rs_pix = auto_rs_load(fp_main)

    dict_mask = {'masks': {}, 'shape': dcm_ct.shape}

    for key in dict_rs_pix:

        dict_mask['masks'][key] = 0

        temp_mask = rs.create_contour_mask(dcm_ct, dict_rs_pix[key])
        temp_mask = rs.create_flat_sparse(temp_mask)

        dict_mask['masks'][key] = temp_mask

    return dict_mask


def auto_rs_load(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dict_rs_pix = a dictionary of each contour with its associated data for
                'color', 'number', 'name' and 'contours' (contour point data) and
                'contours_pix'.
    Summary:    Creates a dictionary for structure of contours point (for DICOM points ('contours') and
                pixel points('contour_pix').
    """
    dcm_ct = rs.create_dcm_ct(fp_main)
    dcm_rs = rs.create_dcm_rs(fp_main)

    dict_rs_dcm = rs.create_dict_rs_dcm(dcm_rs)

    dict_rs_pix = rs.create_dict_rs_pix(dcm_ct, dict_rs_dcm)

    return dict_rs_pix


###################################################################
########################## AUTO MAPS LOAD  ########################
###################################################################


def auto_sim_map_load(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dict_all = dictionary containing all DICOM data (except RS data) and als data for an AUTOMC simulation.
    Summary:    This functions loads all data (except RS data) for an AUTOMC simulation. DICOM data is held in 'dicom'
                while data for als is held in 'als'. All data has been interpolated to the dicom ct size and is stored
                in a flattened and sparsed format. The data can be made back into normal numpy arrays via the unravel
                function (in init_load etc) by using the shape held in 'shape'.
    """
    dict_dcm = auto_dcm_load(fp_main)
    dict_als = auto_als_load_all(fp_main)

    dict_all = {'dicom': dict_dcm, 'als': {}, 'shape': dict_dcm['CT'].shape}

    print('Processing AUTOMC Maps')

    # Resizing to CT size, flattening and sparsing
    # ALS re-sizing to DCM CT and flattening + sparsing
    for key in dict_als.keys():
        dict_all['als'][key] = init.resamp_mat_2_mat(dict_all['dicom']['CT'], dict_als[key])
        dict_all['als'][key] = init.create_flat_sparse(dict_all['als'][key])
        print('AUTOMC Processing: Completed', key)

    # DCM re-sizing to DCM CT and flattening + sparsing
    dict_all['dicom']['total_dose'] = init.resamp_mat_2_mat(dict_all['dicom']['CT'], dict_dcm['total_dose'])
    dict_all['dicom']['total_dose'] = init.create_flat_sparse(dict_all['dicom']['total_dose'])

    dict_all['dicom']['CT'] = init.create_flat_sparse(dict_all['dicom']['CT'])

    return dict_all


###################################################################
######################## AUTO DCM MAP LOAD  #######################
###################################################################


def auto_dcm_load(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dict_als_all = dictionary containing all dcm data.
    Summary:    This function loads dcm ct and dcm dose data in fp_main. Data is in original resolution and unflattened
                or sparsed.
    """
    list_dict = ['total_dose', 'CT']

    dict_dcm = dict.fromkeys(list_dict, 0)

    dict_dcm['total_dose'] = init.create_dcm_total_dose(fp_main)

    dict_dcm['CT'] = init.create_dcm_ct(fp_main)

    return dict_dcm


###################################################################
########################### AUTO ALS LOAD  ########################
###################################################################


def auto_als_load_all(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dict_als_all = dictionary containing all als data
    Summary:    This function loads all als data in fp_main. Data is in original resolution and unflattened or sparsed.
    Note:       It is assumed that all simulations have an als field for CT, physical density and stopping power ratio.
    """
    dict_als_all = auto_als_load_scores(fp_main)

    dict_als_all['CT'] = init.create_als_ct(fp_main)
    dict_als_all['physdens'] = init.create_als_physdens(fp_main)
    dict_als_all['spr'] = init.create_als_spr(fp_main)

    dict_als_all['total_dosetowater'], dict_als_all['total_dosetomed'] = init.create_als_total_dose(fp_main)

    if 'physdens' and 'letdmed' in dict_als_all:
        dict_als_all['letdmass'] = init.create_als_letd_mass(dict_als_all['letdmed'], dict_als_all['physdens'])

    return dict_als_all


def auto_als_load_scores(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     dict_als_score = dictionary of all als scoring grids
    Summary:    Present Scoring grids other than dose are automatically loaded via the file path to the main folder.
    Note:       This assumes that the simulation have finished correctly and that the first field has the same grids as
                the others. Bugs are possible.
    """
    list_present_grids = auto_als_present_scores(fp_main)

    dict_als_score = {}

    for str_present_grid in list_present_grids:
        if str_present_grid[:4] == 'letd':
            dict_als_score[str_present_grid] = init.create_als_total_letd(fp_main, str_present_grid)

        elif str_present_grid[:4] == 'lett':
            dict_als_score[str_present_grid] = init.create_als_total_lett(fp_main, str_present_grid)

        elif str_present_grid == 'track_ends_primary':
            str_te_prim = 'p_track_end'
            print('CURRENTLY NOT CALC-ING TRACK END SCORING GRIDS (PRIMARY PROTONS)')

        elif str_present_grid == 'track_ends_all':
            print('CURRENTLY NOT CALC-ING TRACK END SCORING GRIDS (ALL PROTONS)')

    return dict_als_score


def auto_als_present_scores(fp_main):
    """
    Input:      fp_main = path to full AUTOMC output
    Output:     list_present_scores = list of present score grids other than dose
    Summary:    This function finds the present scoring grids (other than dose) for a patient.
    Notes:      This assumes that the simulation have finished correctly and that the first field has the same grids as
                the others. Bugs are possible.
    """
    list_present_scores = []

    list_possible_grids = auto_als_possible_scores()

    field_names, fp_fields = init.create_als_field_names(fp_main)

    for score_grid in list_possible_grids:

        fp_file = fp_fields[0] + score_grid[1]

        if os.path.isfile(fp_file):
            list_present_scores.append(score_grid[0])

    return list_present_scores


def auto_als_possible_scores():
    """
    Input:      None
    Output:     list_possible_scores = list of possible score grids and their first energy layer split (other than dose)
                in simulations
    Summary:    Gives a list of initial tag and first energy layer split for possible scoring
                grids other than dose. It's used by auto_als_present_grids to search the main patient folder.
    Notes:      Better coding practice would have this stored as a YAML file but this is much quicker to write.
    """
    list_possible_scores = [
        ['letd', 'letd01-doseAveraged-letToWater-denominator.hdr'],
        ['letdmed', 'letdmed01-doseAveraged-denominator.hdr'],
        ['letdpart', 'letdpart01-doseAveraged-denominator.hdr'],
        ['letdprim', 'letdprim01-doseAveraged-letToWater-denominator.hdr'],
        ['letdpost', 'letdpost01-doseAveraged-letToWater-denominator.hdr'],
        ['lett', 'lett01-trackAveraged-letToWater-denominator.hdr'],
        ['letdprimmed', 'letdprimmed01-doseAveraged-denominator.hdr'],
        ['track_ends_primary', 'p_track_end_primary01-Stop.hdr'],
        ['track_ends_all', 'track_end01-Stop.hdr'],
    ]

    return list_possible_scores
