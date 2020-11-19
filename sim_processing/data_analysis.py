import numpy as np
from scipy import sparse
import pickle
import copy
import matplotlib.pyplot as plt
from sim_processing import rbe_calc as rbe


def rem_nan(np_array):

    np_array_rem_nan = np.nan_to_num(np_array, nan=0.0, neginf=0.0, posinf=0.0)

    return np_array_rem_nan


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


def view_rt_struct(dcm_ct, structure, z_slice):
    """
    Input:      dcm_ct = Numpy array / ImageBase of DICOM ct data
                structure = numpy mask / outline of structure
                z_slice = selected slice for figure
    Output:     matplotlib plot (not saved)
    Summary:    This produces a simple figure of a structure in red overlayed on a slice of a CT
    Notes:      Assumes z is the first plane ([z, x, y])
    """

    plt.imshow(dcm_ct[z_slice], cmap=plt.cm.gray)
    plt.imshow(structure[z_slice], alpha=0.5, cmap='Reds')
    plt.title(z_slice)
    plt.colorbar()
    plt.show()


def view_rt_let_crop(ct_set, oar_set, let_set, dose_set, z_slice, save_im = '', dose_window = 0.01, x_lim=[0, 511], y_lim=[0, 511]):
    """
    Input:      ct_set = 3D numpy array / ImageBase of original dicom CT (x,y = 512*512 pixels)
                oar_set = set of outlines of structures to be highlighted (3D numpy array / ImageBase)
                let_set = 3D numpy array / ImageBase of LET
                dose_set = 3D numpy array / ImageBase of dose
                z_slice = selected slice for figure
    Opt Input:  save_im = save location for created figure (no save_im defined means no saving)
                dose_window = dose window value of LET to show
                x_lim = cropping on x
                y_lim = cropping on y
    Output:     matplotlib figure and possible saving
    Summary:    This produces a matplotlib figure for LET on patient anatomy with oars at risk highlighted. Dose window
                for LET region can be defined along with cropping in x and y.
    Notes:      Assumes z is the first plane ([z, x, y])
    """


    ct = ct_set[z_slice, x_lim[0]:x_lim[1], y_lim[0]:y_lim[1]]
    let = let_set[z_slice, x_lim[0]:x_lim[1], y_lim[0]:y_lim[1]]
    oar = oar_set[z_slice, x_lim[0]:x_lim[1], y_lim[0]:y_lim[1]]
    dose = dose_set[z_slice, x_lim[0]:x_lim[1], y_lim[0]:y_lim[1]]

    let_alpha = copy.deepcopy(dose)
    let_alpha[dose > dose_window] = 1
    let_alpha[dose <= dose_window] = 0

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    im = ax2.imshow(np.multiply(let, let_alpha), alpha=1)#, cmap='jet', interpolation='')

    let_alpha[let_alpha>0.8] = 0.8

    plt.imshow(ct, cmap=plt.cm.gray)
    plt.imshow(let*let_alpha, alpha=let_alpha, interpolation ='none')#, cmap='heatmap', interpolation='antialiased')
    plt.colorbar(im).set_label('LETd, Base Settings (keV/um)')
    plt.imshow(oar, alpha=oar, interpolation='none', cmap=plt.cm.gray)

    plt.axis('off')
    plt.title('Slice: ' + str(z_slice))
    #plt.show()

    if save_im == '':
        print('LET image not saved')
    else:
        plt.savefig(save_im)
        print('LET image saved as ', save_im)

    plt.show()


def find_let_max(let_set, dose_set, dose_window=0.05):

    let = copy.deepcopy(let_set)

    let_alpha = copy.deepcopy(dose_set)
    let_alpha[dose_set > dose_window] = 1
    let_alpha[dose_set <= dose_window] = 0

    let_crop = np.multiply(let, let_alpha)
    let_max_val = np.max(let_crop[:])
    let_max_ind = np.unravel_index(np.argmax(let_crop), let_crop.shape)

    print('Max LET value = ', let_max_val)

    print('Max LET Index = ', let_max_ind)

    return let_max_val


def view_rt_dose_crop(ct_set, oar_set, dose_set, z_slice, save_im = '', dose_window = 0.05, x_lim=[0, 511], y_lim=[0, 511]):
    """
    Input:      ct_set = 3D numpy array / ImageBase of original dicom CT (x,y = 512*512 pixels)
                oar_set = set of outlines of structures to be highlighted (3D numpy array / ImageBase)
                dose_set = 3D numpy array / ImageBase of dose
                z_slice = selected slice for figure
    Opt Input:  save_im = save location for created figure (no save_im defined means no saving)
                dose_window = dose window value of dose region to show
                x_lim = cropping on x
                y_lim = cropping on y
    Output:     matplotlib figure and possible saving
    Summary:    This produces a matplotlib figure for dose on patient anatomy with oars at risk highlighted. Dose window
                can be defined along with cropping in x and y.
    Notes:      Assumes z is the first plane ([z, x, y])
    """

    ct = ct_set[z_slice, x_lim[0]:x_lim[1], y_lim[0]:y_lim[1]]
    dose = dose_set[z_slice, x_lim[0]:x_lim[1], y_lim[0]:y_lim[1]]
    oar = oar_set[z_slice, x_lim[0]:x_lim[1], y_lim[0]:y_lim[1]]

    dose_alpha = copy.deepcopy(dose)
    dose_alpha[dose > dose_window] = 0.8
    dose_alpha[dose <= dose_window] = 0

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    im = ax2.imshow(dose, alpha=1, cmap='jet') #interpolation='')

    plt.imshow(ct, cmap=plt.cm.gray)
    plt.imshow(dose, alpha=dose_alpha, cmap='jet', interpolation='antialiased')
    plt.colorbar(im).set_label('Dose, RBE = 1.1 (Gy)')
    plt.imshow(oar, alpha=oar, interpolation='none', cmap=plt.cm.gray)

    plt.axis('off')
    plt.title('Slice: ' + str(z_slice))

    if save_im == '':
        print('Dose image not saved')
    else:
        plt.savefig(save_im)
        print('Dose image saved as ', save_im)

    plt.show()


def view_rt_map(dcm_ct, map_data, z_slice, list_title):
    """
    Input:      dcm_ct = Numpy array / ImageBase of DICOM ct data
                map_data = numpy array of map
                z_slice = selected slice for figure
                list_title = list_title[0] = figure title, list_title[0] = colorbar label
    Output:     matplotlib plot (not saved)
    Summary:    This produces a simple figure of a structure in red overlayed on a slice of a CT
    Notes:      Assumes z is the first plane ([z, x, y])
    """

    plt.imshow(dcm_ct[z_slice], cmap=plt.cm.gray)
    plt.imshow(map_data[z_slice], alpha=0.5)
    title = list_title[0] + ' slice = ' + str(z_slice)
    plt.title(title)
    plt.colorbar().set_label(list_title[1])
    plt.show()


def create_dvh_graph(dose_map, rt_structure, bin_max, bin_increment=0.01):

    map_dose_struct = rt_structure*dose_map # Counts non-zeroes

    bins_array = np.arange(0, bin_max, bin_increment)
    bins_array = np.append(bins_array, bin_max)

    cum_array = np.zeros(bins_array.shape)

    count_array = np.arange(0, len(bins_array))

    for count in count_array:

        cum_temp = len(map_dose_struct[map_dose_struct > bins_array[count]])
        cum_array[count] = cum_temp

    norcum_array = 100 * (cum_array/np.count_nonzero(rt_structure))

    return bins_array, norcum_array


def create_let_w_dose(orig_data):

    dict_data = copy.deepcopy(orig_data)

    dict_rbe = {'letw': {}}

    dict_letw = {'letd': 0, 'letdprim': 0, 'letdpart': 0, 'letdmed': 0, 'letdpost': 0, 'letdmass': 0}

    dose = (unravel_data(dict_data['als']['total_dosetowater'], dict_data['shape']))/1.1
    c = 0.055

    for key in dict_letw.keys():

        letd = unravel_data(dict_data['als'][key], dict_data['shape'])
        dict_letw[key] = dose * (1 + c * letd)

    return dict_letw


def create_rbe11_dvh_graphs(orig_data, pat_struct, max_dvh_val, str_struct_name, low_xlim = 0, frac_num = 30):

    dict_data = copy.deepcopy(orig_data)

    dose_mc_w = unravel_data(dict_data['als']['total_dosetowater'], dict_data['shape'])
    dose_mc_m = unravel_data(dict_data['als']['total_dosetomed'], dict_data['shape'])
    dose_tps_w = (unravel_data(dict_data['dicom']['total_dose'], dict_data['shape'])) / frac_num

    fig, ax = plt.subplots()
    ax.set_xlabel('Dose, Gy (RBE)')
    ax.set_ylabel('Percentage of Volume, %')
    ax.set_xlim(low_xlim, max_dvh_val)
    ax.set_ylim(0, 105)
    ax.set_title(str_struct_name)
    ax.grid()

    bin, norcum = create_dvh_graph(dose_mc_w, pat_struct, max_dvh_val)
    ax.plot(bin, norcum, label='MC Water')

    bin, norcum = create_dvh_graph(dose_mc_m, pat_struct, max_dvh_val)
    ax.plot(bin, norcum, label='MC Medium')

    bin, norcum = create_dvh_graph(dose_tps_w, pat_struct, max_dvh_val)
    ax.plot(bin, norcum, label='TPS Water')

    ax.legend()

    plt.show()


def create_letvh_graphs(orig_data, pat_struct, max_lvh_val, str_struct_name, dose_window = 0.02, save_im ='', low_xlim = 0, let_maps = ['ALL']):

    dict_data = copy.deepcopy(orig_data)

    dose = unravel_data(dict_data['als']['total_dosetowater'], dict_data['shape'])

    list_let = {'letd', 'letdprim', 'letdpart', 'letdmed', 'letdpost', 'letdmass'}

    if let_maps[0] == 'ALL':
        maps = list_let
    else:
        maps = let_maps

    let_alpha = copy.deepcopy(dose)
    let_alpha[dose > dose_window] = 1
    let_alpha[dose <= dose_window] = 0

    fig, ax = plt.subplots()
    ax.set_xlabel('LETd, kev/um')
    ax.set_ylabel('Percentage of Volume, %')
    ax.set_xlim(low_xlim, max_lvh_val)
    ax.set_ylim(0, 105)
    ax.set_title(str_struct_name)

    for key in maps:

        temp_let = unravel_data(dict_data['als'][key], dict_data['shape'])
        temp_let = temp_let * let_alpha

        bin, norcum = create_dvh_graph(temp_let, pat_struct, max_lvh_val)
        ax.plot(bin, norcum, label=str(key))

    ax.grid()
    ax.legend()

    if save_im == '':
        print('LVH image not saved')
    else:
        plt.savefig(save_im)
        print('LVH image saved as ', save_im)

    plt.show()


def create_letdw_dvh_graphs(orig_data, pat_struct, max_dvh_val, str_struct_name, save_im ='', low_xlim = 0, let_maps = ['ALL']):

    dict_data = copy.deepcopy(orig_data)

    dict_letw = create_let_w_dose(dict_data)

    if let_maps =='ALL':

        maps = dict_letw.keys()

    else:

        maps = let_maps

    fig, ax = plt.subplots()
    ax.set_xlabel('Dose, Gy (RBE)')
    ax.set_ylabel('Percentage of Volume, %')
    ax.set_xlim(low_xlim, max_dvh_val)
    ax.set_ylim(0, 105)
    ax.set_title(str_struct_name)

    for key in maps:

        bin, norcum = create_dvh_graph(dict_letw[key], pat_struct, max_dvh_val)
        ax.plot(bin, norcum, label=str(key))


    dose = unravel_data(dict_data['als']['total_dosetowater'], dict_data['shape'])
    bin, norcum = create_dvh_graph(dose, pat_struct, max_dvh_val)
    ax.plot(bin, norcum, label='RBE = 1.1')
    ax.grid()
    ax.legend()
    #lines, labels = ax.get_legend_handles_labels
    #ax.legend(lines, labels, loc = 'upper left')

    if save_im == '':
        print('DVH image not saved')
    else:
        plt.savefig(save_im)
        print('DVH image saved as ', save_im)

    plt.show()


def create_let_pic(ct, oar_set, dose, let, axis, colour_label = 'LETd', str_let = 'LETd', num_title_ftsz = 40, num_ax_ftsz = 30, num_axtk_ftsz = 30):

    dose_window = 0.02
    str_cb_lab = colour_label + '(keV/um)'

    let_alpha = copy.deepcopy(dose)
    let_alpha[dose > dose_window] = 1
    let_alpha[dose <= dose_window] = 0

    im = axis.imshow(np.multiply(let, let_alpha), alpha=1)  # , cmap='jet', interpolation='')

    let_alpha[let_alpha > 0.8] = 0.6

    ax_ct = axis.imshow(ct, cmap=plt.cm.gray)
    ax_let = axis.imshow(let * let_alpha, alpha=let_alpha, interpolation='none')
    cb = plt.colorbar(im, ax=axis)
    cb.set_label(str_cb_lab, fontsize=num_ax_ftsz)
    cb.ax.tick_params(labelsize=num_axtk_ftsz)
    let_oar = axis.imshow(oar_set, alpha=oar_set, interpolation='none', cmap=plt.cm.gray)

    axis.axis('off')
    axis.set_title(str_let, fontsize=num_title_ftsz)
    #axis.set_anchor('W')


def create_dose_set(fp_init_data, cfg, ab_ratio):
    list_letd = ['letd', 'lett', 'letdmass', 'letdmed', 'letdprimmed', 'letdpart', 'letdmss', 'letdpostmss',
                 'letdmss_05cm', 'letdpostmss_05cm']

    dict_dose_letd = {el: 0 for el in list_letd}

    data_init = load_dict_from_pkl(fp_init_data)

    dose = rem_nan(unravel_data(data_init['als']['total_dosetowater'], data_init['shape']) / 1.1)

    for let in list_letd:
        print('LOADING: ' + let)

        temp_unrav_let = unravel_data(data_init['als'][let], data_init['shape'])

        temp_mcnam_dose = rbe.cal_mcnam_dose(dose, temp_unrav_let, cfg, ab_ratio)
        # NOTE: using the MC dose to water for everything

        temp_mcnam_dose = create_flat_sparse(temp_mcnam_dose)

        dict_dose_letd[let] = temp_mcnam_dose

    print('LOADING: Original Doses')

    dict_dose = {'mc_dose': dict_dose_letd,
                 'MC': {'dosetowater': data_init['als']['total_dosetowater'],
                        'dosetomed': data_init['als']['total_dosetomed']},
                 'TPS': {'dosetowater': data_init['dicom']['total_dose']},
                 'shape': data_init['shape']}

    return dict_dose
