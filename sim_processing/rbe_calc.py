import numpy as np
import scipy as sc
import scipy.sparse as scsp
import pickle as pkl
import os as os
import yaml as yaml

# Config File Load

def load_yaml_file(fp_data):
    with open(fp_data, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded

def save_yaml_file(data, fp_data):
    with open(fp_data, 'w') as outfile:
        yaml.dump(data, outfile)

# MM MODEL EQUATIONS

def cal_mm_dsb(dose, let_t, cfg):
    """ Quick Summary: Calculates simple and complex dsb yields ( x, y, z)

        This function calculates the correlations for different types of DNA
        DSBs presented in (Henthorn et al (2019)).

        Inputs:
            - dose:         dose matrix
                            (x, y, z)
            - let_t:        track weighted let matrix
                            (x, y, z)
            - cfg:          YAML config file for published parameters of MM
                            model and other RBE models.

        Outputs:
            - simp dsb:         total yield of simple dsbs (x, y, z)
            - total_comp_dsb:   total yield of complex dsbs (x, y, z)


    """

    # Calculation

    def basic_dsb(do, lt, a, b, c):
        c_ar = np.ones(lt.shape) * c

        dsb_type = do * (a * lt * (lt) + b * let_t + c_ar)
        return dsb_type

    simp_dsb = basic_dsb(dose, let_t, cfg['init_dsb']['a1'], cfg['init_dsb']['b1'], cfg['init_dsb']['c1'])
    simpbase_dsb = basic_dsb(dose, let_t, cfg['init_dsb']['a2'], cfg['init_dsb']['b2'], cfg['init_dsb']['c2'])
    comp_dsb = basic_dsb(dose, let_t, cfg['init_dsb']['a3'], cfg['init_dsb']['b3'], cfg['init_dsb']['c3'])
    compbase_dsb = basic_dsb(dose, let_t, cfg['init_dsb']['a4'], cfg['init_dsb']['b4'], cfg['init_dsb']['c4'])

    total_comp_dsb = simpbase_dsb + comp_dsb + compbase_dsb

    return simp_dsb, total_comp_dsb


def cal_mm_dsb_rbe_dose(dose, let_t, cfg):
    """ Quick Summary: Calculates RBE for endpoints of simple and complex dsb
                      yields (x, y, z).

        This function extends the correlations presented in
        (Henthorn et al (2019)) for different types of DNA DSBs presented to
        produce RBE values.

        Inputs:
            - dose:         dose matrix
                            (x, y, z)
            - let_t:        track weighted let matrix
                            (x, y, z)
            - cfg:          YAML config file for published parameters of MM
                            model and other RBE models.

        Outputs:
            - dose_rbe_simp:    RBE-weighted dose for simple dsb yield
                                (x, y, z)
            - dose_rbe_comp:    RBE-weighted dose for complex dsb yield
                                (x, y, z)

    """

    # Calculation

    [simp_dsb, total_comp_dsb] = cal_mm_dsb(dose, let_t, cfg)

    dose_rbe_simp = simp_dsb / cfg['init_dsb']['pho_simp']
    dose_rbe_comp = total_comp_dsb / cfg['init_dsb']['pho_comp']

    return dose_rbe_simp, dose_rbe_comp


def cal_mm_resmis(dose, let_t, cfg):
    ''' Quick Summary: Calculates residual and misrepair yields (x, y, z)

        This function calculates the correlations for residual and misrepair
        yields presented in (Henthorn et al (2018)).

        Inputs:
            - dose:         dose matrix
                            (x, y, z)
            - let_t:        track weighted let matrix
                            (x, y, z)
            - cfg:          YAML config file for published parameters of MM
                            model and other RBE models.

        Outputs:
            - res:  yield of residual (x, y, z)
            - mis:  yield of misrepair (x, y, z)

    '''

    # Obtaining sparese arrays for constants which are added

    l_ones = np.ones(let_t.shape)

    # Calculation

    # Number of DSBs
    n_dsb = dose * ((cfg['dam_yield']['d1'] * let_t) + (l_ones * cfg['dam_yield']['e1']))
    # Cluster Density
    clu_den = (cfg['dam_yield']['f1'] * let_t * (let_t)) + (cfg['dam_yield']['g1'] * let_t) + (
                l_ones * cfg['dam_yield']['h1'])

    # Residual Yield
    res = n_dsb * cfg['dam_yield']['c1']
    # Misrepair Yield
    mis = n_dsb * (
                (cfg['dam_yield']['a1'] * clu_den + (l_ones * cfg['dam_yield']['b1'])) * (1 - cfg['dam_yield']['c1']))

    return res, mis


def cal_mm_resmis_rbe_dose(dose, let_t, cfg):
    ''' Quick Summary: Calculates dose-weighted RBE for residual and misrepair.
                       (spot, x, y, z)

        This function extends the correlations presented in
        (Henthorn et al (2018)) for residual and misrepair yields to produce
        RBE values.

        Inputs:
            - dose:         dose matrix
                            (x, y, z)
            - let_t:        track weighted let matrix
                            (x, y, z)
            - cfg:          YAML config file for published parameters of MM
                            model and other RBE models.

        Outputs:
            - dose_rbe_res:  RBE-weighted dose for residual (x, y, z)
            - dose_rbe_mis:  RBE-weighted dose for misrepair (x, y, z)

    '''

    # Residual and Misrepair Yields
    [res, mis] = cal_mm_resmis(dose, let_t, cfg)

    # Ones

    # Calculation

    dose_rbe_res = res / cfg['dam_yield']['pho_res']
    dose_rbe_mis = mis / cfg['dam_yield']['pho_mis']

    return dose_rbe_res, dose_rbe_mis


# PHENO BIO MODELS

def cal_letwt_dose(dose, let_d, cfg):
    ''' Quick Summary: Calculates let-weighted dose (x, y, z)
        Inputs:
        - dose:         dose matrix
                        (x, y, z)
        - let_d:        Dose Weighted let matrix
                        (x, y, z)
        - cfg:          YAML config file for published parameters of MM
                        model and other RBE models.
    '''

    # ones array for LET
    l_ones = np.ones(let_d.shape)

    # Calculation

    let_wt_dose = dose * (l_ones + let_d * cfg['let_wt']['mcmahon_c'])

    return let_wt_dose


def cal_mcnam_dose(dose, let_d, cfg, ab_ratio):
    ''' Quick Summary: Calculates RBE-weighted dose using the McNamara Model
        Inputs:
        - dose:         dose matrix
                        (x, y, z)
        - let_d:        Dose Weighted let matrix
                        (x, y, z)
        - cfg:          YAML config file for published parameters of MM
                        model and other RBE models.
        - ab_ratio:     ab ratio matrix
                        (x, y, z)
    '''

    l_ones = np.ones(let_d.shape)

    rbe_max = (l_ones * cfg['mcnamara']['p0']) + cfg['mcnamara']['p1'] * (let_d / ab_ratio)
    rbe_min = (l_ones * cfg['mcnamara']['p2']) + cfg['mcnamara']['p3'] * let_d * (np.sqrt(ab_ratio))

    eq_a = 0.5 * 1 / dose
    eq_b = ab_ratio * ab_ratio
    eq_c = 4 * (dose * ab_ratio)
    eq_d = 4 * (dose * dose)

    eq_sqrt = eq_b + eq_c * rbe_max + eq_d * (rbe_min * rbe_min)

    rbe_mcnam = eq_a * (np.sqrt(eq_sqrt) - ab_ratio)

    dose_rbe_mcnam = dose * rbe_mcnam

    dose_rbe_mcnam[np.isnan(dose_rbe_mcnam)] = 0
    dose_rbe_mcnam[np.isinf(dose_rbe_mcnam)] = 0

    return dose_rbe_mcnam