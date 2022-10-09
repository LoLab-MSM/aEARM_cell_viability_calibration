# Michael W. Irvin
# 21 July 2021
# Setup Cell Viability Calibration
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from example.cell_viability.load_data import experimental_conditions_df, merge_cols, update_params
from opt2q.simulator import Simulator
from opt2q.measurement.base.transforms import ScaleGroups, Standardize
from proportion_cell_fate_measurement_model.core.preprocessing import critical_points as cp
from proportion_cell_fate_measurement_model.core.preprocessing import max_derivative as md
from opt2q_examples.apoptosis_model import model
from example.cell_viability import sample_params
import numpy as np

len_mechanism_params = 44
sim = Simulator(model, tspan=np.linspace(0, 21600, 100), param_values=experimental_conditions_df, solver='scipyode',
                solver_options={'integrator': 'lsoda'}, integrator_options={'mxstep': 2**20})
sim.param_values = update_params(sample_params.x + [2.3, 2.3, 2.3, 2.3, -1, 1, 1, 1, 1, 1])
new_results = sim.run().opt2q_dataframe.reset_index()

# Set up preprocessing steps
critical_points = ScaleGroups(columns=['time', 'tBID_obs'], drop_columns=True, groupby='simulation',
                              scale_fn=cp, **{'wrt': 'time', 'atol': 1e-6})  # Step 1A: Critical Points
cps_results = critical_points.transform(new_results[['time', 'tBID_obs', 'simulation']+merge_cols])

max_derivative_points = ScaleGroups(columns=['time', 'tBID_obs'], drop_columns=True, groupby='simulation',
                                    scale_fn=md, **{'wrt': 'time', 'atol': 1e-6})  # Step 1B: Half-Max Points
mdp_results = max_derivative_points.transform(new_results[['time', 'tBID_obs', 'simulation'] + merge_cols])

cps_results = cps_results.merge(mdp_results, on=merge_cols + ['simulation'], how='outer')

standardize = Standardize()  # Step 2: Standardize
preprocessed_results = standardize.transform(cps_results)
preprocessed_results.fillna(0.0, inplace=True)  # Step 3: Fill NaNs

# Feature List
starting_feature_list = list(preprocessed_results.loc[:, preprocessed_results.columns.str.contains('tBID')].columns
                             .sort_values())


def plot_dynamics(obs, new_res):
    i = 0
    for name_, group_ in new_res.groupby(['Genotype']):
        j = 0
        ntc = len(group_.groupby(['TRAIL_Conc']))
        for trail_c, group_tc in group_.groupby(['TRAIL_Conc']):
            plt.plot(group_tc['time'], group_tc[obs], color=cm.colors[i % 20], alpha=0.1 + j * 0.9 / ntc, label=trail_c)
            j += 1
        i += 1
        plt.legend()
        plt.title(f'{obs.split("_obs")[0]} Dynamics in {name_} HeLa Cells')
        plt.show()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import load_data
    from sample_params import x

    new_params = load_data.update_params(x + [2.3, 2.3, 2.3, 2.3, -1, 1, 1, 1, 1, 1])
    sim.param_values = new_params
    new_results = sim.run().opt2q_dataframe.reset_index()

    cm = plt.get_cmap('tab10')

    plot_dynamics('tBID_obs', new_results)
    plot_dynamics('C3_active_obs', new_results)

    for name, group in preprocessed_results.groupby('Genotype'):
        plt.scatter(group.tBID_obs__p_poi__time__0, group.tBID_obs__p_poi__0, alpha=0.4, label=name)
    plt.xlabel('time at tBID point of inflection')
    plt.ylabel('tBID point of inflection')
    plt.legend()
    plt.show()

    for name, group in preprocessed_results.groupby('Genotype'):
        plt.scatter(group.tBID_obs__p_poi__time__0, group.dtBID_obs__p_poi__0, alpha=0.4, label=name)
    plt.xlabel('time at tBID point of inflection')
    plt.ylabel('BID truncation rate at point of inflection')
    plt.legend()
    plt.show()
