from sim_processing import rbe_calc as rbe
import os as os


##################################
###### Dict Structure Creation ###
##################################

dict_rbe_cfg = {
    'mm_model' : {'EAKS19' : {},
                  'YQ20'   : {},
                  'NH19'   : {}
                  },
    'let_wt'   : {'McMa19' : {}
                  },
    'mcnamara' : {'McNa15' : {}
                  }
                }


###################
###### MM Model ###
###################

##########
# EAKS19 #
##########
################
# Damage Yield #
################

dict_rbe_cfg['mm_model']['EAKS19']['dam_yield'] = \
    {'a1': float(0.1966),
     'b1': float(0.008),
     'c1': float(0.0736),
     'd1': float(1.149),
     'e1': float(24.1),
     'f1': float(0.0004879),
     'g1': float(0.00284),
     'h1': float(0.0513),
     'pho_mis': float(0.042770764),
     'pho_res': float(1.72601618)}

dict_rbe_cfg['mm_model']['EAKS19']['summary'] = 'NEED TO FILL THESE IN AT SOME POINT'

########
# YQ19 #
########
################
# Damage Yield #
################

dict_rbe_cfg['mm_model']['YQ20']['dam_yield'] = \
    {'a1': float(),
     'b1': float(0.008),
     'c1': float(0.0736),
     'd1': float(1.149),
     'e1': float(24.1),
     'f1': float(0.0004879),
     'g1': float(0.00284),
     'h1': float(0.0513),
     'pho_mis': float(0.042770764),
     'pho_res': float(1.72601618)}

dict_rbe_cfg['mm_model']['YQ20']['summary'] = 'NEED TO FILL THESE IN AT SOME POINT'

########
# NH19 #
########
################
# Initial DSB #
################

dict_rbe_cfg['mm_model']['NH19']['init_dam'] = \
    {'a1': float(-0.00243873),
     'a2': float(-0.000676722),
     'a3': float(0.00129442),
     'a4': float(0.00347432),
     'b1': float(0.398052),
     'b2': float(0.208652),
     'b3': float(0.316023),
     'b4': float(0.141311),
     'c1': float(16.4117),
     'c2': float(2.37577),
     'c3': float(4.85635),
     'c4': float(1.56121),
     'pho_comp': float(9.6328),
     'pho_simp': float(15.3328)
     }

dict_rbe_cfg['mm_model']['NH19']['summary'] = 'NEED TO FILL THESE IN AT SOME POINT'

#################
###### LET_wt ###
#################
##########
# McMa19 #
##########

dict_rbe_cfg['let_wt']['McMa19'] = {'c': float(0.055)}

dict_rbe_cfg['let_wt']['McMa19']['summary'] = 'NEED TO FILL THESE IN AT SOME POINT'


#########################
###### McNamara Model ###
#########################

dict_rbe_cfg['mcnamara']['McNa15'] = \
     {'p0': float(0.999064),
      'p1': float(0.35605),
      'p2': float(1.1012),
      'p4': float(0.0038703)}

dict_rbe_cfg['mcnamara']['McNa15']['summary'] = 'NEED TO FILL THESE IN AT SOME POINT'

#SAVING

fp_save_dict = 'RBE_config.yaml'

rbe.save_yaml_file(dict_rbe_cfg, fp_save_dict)


