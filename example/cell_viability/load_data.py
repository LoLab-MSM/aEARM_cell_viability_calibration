# Michael W. Irvin
# 21 July 2021
# Load Cell Viability Dataset
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from opt2q.noise import NoiseModel
from opt2q_examples.apoptosis_model import model
import numpy as np

# Load Data
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'Cell_Viability_Data.xlsx')
data = pd.read_excel(file_path)

# Important Columns
observables = ['Percent Cell Viability']
std_observables = ['CV_Std_dev']
dropped_columns = ['Notes']
param_columns = ['param', 'value']
merge_cols = list(data.iloc[:, ~data.columns.isin(observables+std_observables+dropped_columns+param_columns)].columns)

# Dataset
data_df = data.iloc[:, ~data.columns.isin(param_columns + dropped_columns)].drop_duplicates(ignore_index=True)


# Background Parameters
def add_ec_columns(p):
    pb_ec_pre = data.iloc[:, data.columns.isin(merge_cols)].drop_duplicates(ignore_index=True)
    pb_ec = pb_ec_pre.iloc[np.repeat(range(len(pb_ec_pre)), len(p))]
    pb_ec = pb_ec.reset_index(drop=True)

    params_ = p.iloc[np.tile(range(len(p)), len(pb_ec_pre))]
    params_ = params_.reset_index(drop=True)
    param_bg = pd.concat([pb_ec, params_], axis=1)
    return param_bg


all_params_names = [p.name for p in model.parameters_all()]
rate_params_names = [p.name for p in model.parameters_rules()]
rate_params_values = [p.value for p in model.parameters_rules()]
remaining_params_names = list(set(data.param.unique())-set(rate_params_names))
remaining_params_values = [model.parameters_all()[all_params_names.index(p)].value for p in remaining_params_names]

params = pd.DataFrame({'param': rate_params_names + remaining_params_names,
                       'value': rate_params_values + remaining_params_values})
param_background = add_ec_columns(params)

# Experimental Conditions
experimental_conditions = NoiseModel(model=model, param_mean=param_background)
experimental_conditions_raw = data.iloc[:, ~data.columns.isin(observables+std_observables+dropped_columns)]
experimental_conditions.update_values(param_mean=experimental_conditions_raw)

experimental_conditions_df = experimental_conditions.run()


def update_params(x_new):
    # Parameters 0-33 are values in x are the log of the model parameters -- these take normal prior with +/-1.5 std
    # Parameters 34-37 are sensitivity parameters (as modeled as log-values of R_0).
    #   They take a normal prior mu=2.3, sig=0.113. This results in a log-normal expression at 200+/-60.
    # Parameter 38 is the Luteolin effect on R_0 parameter. This takes a normal prior mu = 0, sig = 1.5
    # Parameters 39-40 are the caspase 8 and caspase 3 respectively. This takes a half-normal prior mu = 0, sig = 1.5
    # Parameters 41-42 are the effects of FADD knockdowns (low and high respectively).
    #   This takes a truncated normal prior, domain of [0, inf), with mu=1, 3 and sig=1.5
    # Parameter 43 is the effect of Bortezomib This takes a half-normal prior with mu = 0, sig = 1.5

    # log10(p) = x
    ec_cols = merge_cols + ['param', 'value']
    param_len = len(model.parameters_rules())
    new_bg_params_ = pd.DataFrame({'param': [p.name for p in model.parameters_rules()] + remaining_params_names,
                                   'value': [10 ** p for p in x_new[:param_len]] + remaining_params_values})

    new_bg_params = add_ec_columns(new_bg_params_)

    # fixed params from data
    data_params = data[merge_cols+param_columns]

    # Publication dependent apoptosis sensitivity effect
    publications = ['Horinaka_M_2005', 'Wajant_H_1998', 'Orzechowska_E_2014', 'Roux_J_2015']
    pub_params = x_new[param_len:param_len+len(publications)]

    for i, pb in enumerate(publications):
        new_bg_params.loc[
            new_bg_params[(new_bg_params['Publication'].str.contains(pb)) & (new_bg_params['param'] == 'R_0')].index,
                          'value'] = 10**pub_params[i]

    # luteolin effect on R_0
    luteolin_effect_param = 10**x_new[param_len+len(publications)]
    luteolin_effect = new_bg_params[(new_bg_params['Genotype'].str.contains('Luteolin')) &
                                    (new_bg_params['param'] == 'R_0')][ec_cols]\
        .rename(columns={'value': 'old_value'})

    luteolin_effect['value'] = luteolin_effect.apply(
        lambda row: float(row[['Genotype']].str.extract('(\d+)').iloc[0, 0]) * luteolin_effect_param
        + row['old_value'], axis=1)  # R_0 = R_0wt + l*lut_concentration
    luteolin_effect.drop('old_value', axis=1, inplace=True)
    luteolin_effect.reset_index(drop=True, inplace=True)

    # caspase 8 inhibition effects
    caspase_8_inh_param = 10**-x_new[param_len+len(publications)+1]
    caspase_8_inh_effect = new_bg_params[
        ((new_bg_params['Genotype'].str.contains('C8_inh')) | (new_bg_params['Genotype'].str.contains('zVAD'))) &
        ((new_bg_params['param'] == 'kc2') | (new_bg_params['param'] == 'kc4'))][ec_cols]\
        .rename(columns={'value': 'old_value'})
    caspase_8_inh_effect['value'] = caspase_8_inh_effect.apply(lambda row: row['old_value'] * caspase_8_inh_param, axis=1)
    caspase_8_inh_effect.drop('old_value', axis=1, inplace=True)
    caspase_8_inh_effect.reset_index(drop=True, inplace=True)

    # caspase 3 inhibition effects
    caspase_3_inh_param = 10 ** -x_new[param_len + len(publications) + 2]
    caspase_3_inh_effect = new_bg_params[
        ((new_bg_params['Genotype'].str.contains('C3_inh')) | (new_bg_params['Genotype'].str.contains('zVAD'))) &
        ((new_bg_params['param'] == 'kc3') | (new_bg_params['param'] == 'kc8'))][ec_cols] \
        .rename(columns={'value': 'old_value'})
    caspase_3_inh_effect['value'] = caspase_3_inh_effect.apply(lambda row: row['old_value'] * caspase_3_inh_param,
                                                               axis=1)
    caspase_3_inh_effect.drop('old_value', axis=1, inplace=True)
    caspase_3_inh_effect.reset_index(drop=True, inplace=True)

    # High and Low Delta FADD effects
    low_delta_FADD_effect_parameter = 10 ** -x_new[param_len + len(publications) + 3]
    low_delta_FADD_effect = new_bg_params[(new_bg_params['Genotype'].str.contains('Low_Delta_FADD')) &
                                          (new_bg_params['param'] == 'kc0')][ec_cols]\
        .rename(columns={'value': 'old_value'})
    low_delta_FADD_effect['value'] = low_delta_FADD_effect.apply(lambda row: row['old_value'] *
                                                                             low_delta_FADD_effect_parameter, axis=1)
    low_delta_FADD_effect.drop('old_value', axis=1, inplace=True)

    high_delta_FADD_effect_parameter = 10 ** -x_new[param_len + len(publications) + 4]
    high_delta_FADD_effect = new_bg_params[(new_bg_params['Genotype'].str.contains('High_Delta_FADD')) &
                                           (new_bg_params['param'] == 'kc0')][ec_cols] \
        .rename(columns={'value': 'old_value'})
    high_delta_FADD_effect['value'] = high_delta_FADD_effect.apply(lambda row: row['old_value'] *
                                                                               high_delta_FADD_effect_parameter, axis=1)
    high_delta_FADD_effect.drop('old_value', axis=1, inplace=True)
    high_delta_FADD_effect.reset_index(drop=True, inplace=True)

    # Bortezomib effect
    bortezomib_effect_parameter = 10 ** x_new[param_len + len(publications) + 5]
    bortezomib_effect = new_bg_params[(new_bg_params['Genotype'].str.contains('Bortezomib')) &
                                      (new_bg_params['param'] == 'IC_0')][ec_cols] \
        .rename(columns={'value': 'old_value'})
    bortezomib_effect['value'] = bortezomib_effect.apply(lambda row: row['old_value'] *
                                                                     bortezomib_effect_parameter, axis=1)
    bortezomib_effect.drop('old_value', axis=1, inplace=True)
    bortezomib_effect.reset_index(drop=True, inplace=True)

    experimental_conditions.update_values(new_bg_params)
    experimental_conditions.update_values(data_params)
    experimental_conditions.update_values(luteolin_effect)
    experimental_conditions.update_values(caspase_8_inh_effect)
    experimental_conditions.update_values(caspase_3_inh_effect)
    experimental_conditions.update_values(low_delta_FADD_effect)
    experimental_conditions.update_values(high_delta_FADD_effect)
    experimental_conditions.update_values(bortezomib_effect)
    # return data_params
    return experimental_conditions.run()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec

    def plot_data(data_df_):
        cm = plt.get_cmap('tab20')
        fig = plt.figure(1, figsize=(9, 4))
        gs = gridspec.GridSpec(1, 7, hspace=0.5)
        ax = fig.add_subplot(gs[0, :4])

        i = 0
        for name, group in data_df_.groupby(['Cells', 'Genotype', 'Figure', 'Publication']):
            x = [float(x_) for x_ in group['TRAIL_Conc'].str.extract('(\d*\.*\d*)').iloc[:, 0]]
            y = group['Percent Cell Viability']
            y_err = group['CV_Std_dev']
            ax.errorbar(x, y, y_err, fmt='o', capsize=5, color=cm.colors[i % 20])
            ax.errorbar([], [], [], fmt='o', capsize=5, color=cm.colors[i % 20],
                        label=(group.Publication.iloc[0], group.Genotype.iloc[0]))
            i += 1
            plt.xscale('log')

        plt.xlabel('TRAIL Conc [ng/mL]')
        plt.ylabel('Cell Viability')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        plt.title('Cell Viability in Hela Cells')
        plt.show()


    plot_data(data_df)
    plot_data(data_df[data_df.Genotype == 'WT'])
