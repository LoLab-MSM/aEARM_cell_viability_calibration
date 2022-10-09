# MW Irvin

# 8 Nov 2021
# Plot Results of Calibration to Cell Viability Data
import sys, os, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from scipy.stats import norm
from pydream.parameters import SampledParam
from opt2q_examples.apoptosis_model import model
from opt2q_examples.plot_tools import calc
from opt2q.simulator import Simulator
from opt2q.measurement.base import ScaleToMinMax
from example.cell_viability.calibrate_model import make_priors
from example.cell_viability import sample_params
from example.cell_viability.load_data import update_params, merge_cols
from example.cell_viability.setup_calibration import critical_points, max_derivative_points, standardize
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# Plot Settings
# =============
cm = plt.get_cmap('tab10')
line_width = 2
tick_labels_x = 16
tick_labels_y = 17

# Load Calibration Results
# ========================
calibration_folder = 'calibration_results'
calibration_date = '20211021'
calibration_measurement_model = 'gaussian_process_model'
calibration_dataset = 'wajant_dataset' 

# files
script_dir = os.path.dirname(__file__)
file_paths_ = sorted([
    os.path.join(script_dir, calibration_folder, f) for f in
    os.listdir(os.path.join(script_dir, calibration_folder))
    if f'{calibration_dataset}_{calibration_measurement_model}_{calibration_date}' in f])

# parameter files
param_file_paths_ = [f for f in file_paths_ if 'parameters' in f]
file_order = sorted(list(set(int(re.findall(r'\d+', file_name)[-1]) for file_name in param_file_paths_)))
traces = sorted(list(set(int(re.findall(r'\d+', file_name)[-2]) for file_name in param_file_paths_)))

parameter_file_paths = []
for file_num in file_order:
    parameter_file_paths += [g for g in param_file_paths_ if f'_{file_num}_' in g]

parameter_samples = []
for trace_num in traces:
    parameter_sample = np.concatenate([np.load(os.path.join(script_dir, calibration_folder, pp))
                                       for pp in parameter_file_paths if f'_{trace_num}_' in pp])
    parameter_samples.append(parameter_sample)

# Plot Training Dataset
# =====================
file_path = os.path.join(script_dir, 'Cell_Viability_Data.xlsx')
data = pd.read_excel(file_path)

dropped_columns = ['Notes']
param_columns = ['param', 'value']
data_df = data.iloc[:, ~data.columns.isin(param_columns + dropped_columns)].drop_duplicates(ignore_index=True)

# plot function
def plot_training_data(training_data_df, ax):
    i = 0
    genotypes = {'WT': (0, 'Wildtype'),
                 'Low_Delta_FADD': (1, 'Low Delta-FADD'),
                 'High_Delta_FADD': (2, 'High Delta-FADD')}
    for name, group in training_data_df.groupby(['Cells', 'Genotype', 'Figure', 'Publication']):
        x = [float(x_) for x_ in group['TRAIL_Conc'].str.extract('(\d*\.*\d*)').iloc[:, 0]]
        y = group['Percent Cell Viability']
        y_err = group['CV_Std_dev']
        ax.errorbar(x, y, y_err, fmt='o', capsize=5, color=cm.colors[genotypes[group.Genotype.iloc[0]][0]])
        ax.errorbar([], [], [], fmt='o', capsize=5, color=cm.colors[genotypes[group.Genotype.iloc[0]][0]],
                    label=genotypes[group.Genotype.iloc[0]][1])
        i += 1


fig = plt.figure(1, figsize=(9, 4))
gs = gridspec.GridSpec(1, 7, hspace=0.5)
ax0 = fig.add_subplot(gs[0, :4])
plot_training_data(data_df[data_df['Publication'].str.contains('Wajant')], ax0)  # plot training data
plt.xscale('log')
plt.xlabel('TRAIL Conc [ng/mL]')
plt.ylabel('Cell Viability')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.title('Cell Viability in Hela Cells')
plt.show()

# Plot Supplemental Figures
# Parameter KDE (and Priors)
# -- note you only need the parameters that are actually tuned by the data.
# Parameters 0-33 are values in x are the log of the model parameters -- these take normal prior with +/-1.5 std
# Parameters 34-37 are sensitivity parameters (as modeled as log-values of R_0).
# Parameters 41-42 are the effects of FADD knockdowns (low and high respectively).
#   This takes a truncated normal prior, domain of [0, inf), with mu=1, 3 and sig=1.5
parameter_names = [p.name for p in model.parameters_rules()]
parameter_names += ['Horinaka_M_2005', 'Wajant_H_1998', 'Orzechowska_E_2014', 'Roux_J_2015']
parameter_names += ['Luteolin Effect']
parameter_names += ['Caspase 8 Inh.', 'Caspase 3 Inh.']
parameter_names += ['low delta FADD', 'high delta FADD']
parameter_names += ['Bortexomib Effect']
# parameter_names += ['dtBID_obs__max', 'dtBID_obs__max__t', 'dtBID_obs__p_poi__0', 'tBID_obs__min__0',
#                     'tBID_obs__min__not_nan__0', 'tBID_obs__min__time__0', 'tBID_obs__p_poi__0',
#                     'tBID_obs__p_poi__not_nan__0', 'tBID_obs__p_poi__time__0', 'unused feature', 'gp_constant']
parameter_names += ["max Bid truncation rate", "time at which Bid truncation rate maximizes", "d/dt tBID conc. at point of inflection (poi)", "relative min tBID conc.",
                    "relative min tBID exists", "time at relative min tBID conc.", "tBID conc. at poi", 
                    "tBID poi exists", "time at tBID poi", "unused feature", "gp constant"]                    
parameter_priors = [SampledParam(norm, loc=p, scale=1.5) for p in sample_params.x]\
                   + make_priors(calibration_measurement_model)[1:]
parameter_posteriors = np.concatenate([p[200000:, :] for p in parameter_samples])

# select parameters that depend on the data
selected_names = parameter_names[:34] + parameter_names[35:36] + parameter_names[41:43] + parameter_names[44:53] \
                      + parameter_names[54:55]

selected_priors = parameter_priors[:34] + parameter_priors[35:36] + parameter_priors[41:43] + parameter_priors[44:53] \
                  + parameter_priors[54:55]

selected_posteriors = np.concatenate(
    [parameter_posteriors[:, :34], parameter_posteriors[:, 35:36], parameter_posteriors[:, 41:43],
     parameter_posteriors[:, 44:53], parameter_posteriors[:, 54:55]], axis=1) \
[np.random.randint(0, len(parameter_posteriors), 100)]

# Plot KDEs of Posteriors and Priors
n = 24
gs_columns = 3
n_params_subset = 12

for ps in range(int(np.ceil(len(selected_names)/n_params_subset))):
    param_subset = selected_names[n * ps:min(n * (ps + 1), len(selected_names))]
    priors_subset = selected_priors[n * ps:min(n * (ps + 1), len(selected_names))]
    posteriors_subset = selected_posteriors[:, n * ps:min(n * (ps + 1), len(selected_names))]

    gs_rows = int(np.ceil(len(param_subset)/gs_columns))

    fig = plt.figure(1, figsize=(9, 11 * gs_rows / 8.0))
    if gs_rows == 0:
        continue

    gs = gridspec.GridSpec(gs_rows, gs_columns, hspace=0.1)
    for i, param in enumerate(param_subset):
        r = int(i / gs_columns)
        c = i % gs_columns
        ax = fig.add_subplot(gs[r, c])
        ax.set_yticks([])
        ax.set_title(param)
        # Prior
        x = np.linspace(*priors_subset[i].dist.interval(0.99), 100)
        plt.plot(x, priors_subset[i].dist.pdf(x))

        # Posterior
        sns.kdeplot(posteriors_subset[:, i], ax=ax, color=cm.colors[1], alpha=0.6)

    gs.tight_layout(fig)
    # plt.savefig(f'Supplemental__KDE_Parameter_Priors_Posteriors_Cell_Viability_Dataset_Plot{ps}.pdf')
    plt.show()

# Model predictions of tBID Dynamics for comparison to synthetic dataset
sim_parameters = pd.DataFrame(10**selected_posteriors[:, :34], columns=selected_names[:34])
sim_parameters['L_0'] = 3000  # Simulate Model (for 50ng/mL TRAIL)
sim_parameters['simulation'] = range(len(sim_parameters))
sim_parameters.reset_index(inplace=True, drop=True)

sim = Simulator(model, tspan=np.linspace(0, 21600, 100), param_values=sim_parameters, solver='scipyode',
                solver_options={'integrator': 'lsoda'}, integrator_options={'mxstep': 2**20})
sim_res = sim.run()
sim_results = sim_res.opt2q_dataframe.reset_index().rename(columns={'index': 'time'})
sim_results_normed = ScaleToMinMax(columns=['tBID_obs'], groupby='simulation').\
    transform(sim_results[['time', 'tBID_obs', 'simulation']])

sim.param_values = pd.DataFrame([10**np.array(sample_params.x[:34])], columns=selected_names[:34])
sim_results_true = sim.run().opt2q_dataframe.reset_index().rename(columns={'index': 'time'})
sim_results_true_normed = ScaleToMinMax(columns=['tBID_obs']).\
    transform(sim_results_true[['time', 'tBID_obs']])

# ---- Figure 6D (or part of previous figures) ----
# Plot Posterior predictions for aEARM w/ 50ng/mL TRAIL
sim_results_lower_quantile, sim_results_median, sim_results_upper_quantile = calc.simulation_results_quantiles_list(
    sim_results_normed, (0.025, 0.500, 0.975)
)
y_lower = sim_results_lower_quantile['tBID_obs']
y_upper = sim_results_upper_quantile['tBID_obs']

area = sum(y_upper - y_lower)
print('Area of the 95% CI is ', area)

fig, ax1 = plt.subplots()
y1 = sim_results_lower_quantile['tBID_obs'].values
y2 = sim_results_upper_quantile['tBID_obs'].values
x = sim_results_upper_quantile['time'].values
ax1.fill_between(x, y1, y2, color='k', alpha=0.2, label='posterior', linewidth=line_width)
ax1.plot(x, sim_results_median['tBID_obs'].values, color='k', alpha=0.2)
ax1.plot(np.linspace(0, 21600, 100), sim_results_true_normed['tBID_obs'].values,
         color='k', alpha=0.2, label='true', linewidth=line_width, linestyle=':')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('Normalized tBID Concentration')
ax1.legend()
ax1.tick_params(axis='x', which='major', labelsize=tick_labels_x)
ax1.tick_params(axis='y', which='major', labelsize=tick_labels_y)
plt.show()

# Simulate Model of (all KOs and concentrations of TRAIL)
full_exp_cond_parameters = pd.concat([update_params(p[:44]) for p in parameter_posteriors[
    np.random.randint(0, len(parameter_posteriors), 100)]])
exp_cond_parameters = full_exp_cond_parameters[full_exp_cond_parameters['Publication'].str.contains('Wajant')]
exp_cond_parameters.drop(columns=['simulation'], inplace=True)
exp_cond_parameters.reset_index(inplace=True, drop=True)

sim.param_values = exp_cond_parameters[exp_cond_parameters['TRAIL_Conc'].str.contains('100 ng/mL')]\
    .reset_index(drop=True).reset_index().rename(columns={'index': 'simulation'})
exp_cond_100ng_mL_results = sim.run(tspan=np.linspace(0, 36000, 100))\
    .opt2q_dataframe.reset_index().rename(columns={'index': 'time'})
exp_cond_100ng_mL_results['time'] = exp_cond_100ng_mL_results['time'].apply(lambda row: row/60.0)


# plot 100ng/mL TRAIL dependent tBID vs KOs
def plot_dynamics(sim_results_to_plot, groupby, label='posterior'):
    fig00, ax00 = plt.subplots()
    ii = 0
    for genotype, group in sim_results_to_plot.groupby(groupby):
        # get the upper, lower and median of the posterior predictions of the Genotype dependent trajectories
        ec_results_lower_quantile, ec_results_median, ec_results_upper_quantile = calc.simulation_results_quantiles_list(
            group, (0.025, 0.500, 0.975)
        )
        y01 = ec_results_lower_quantile['tBID_obs'].values
        y02 = ec_results_upper_quantile['tBID_obs'].values
        x0 = ec_results_median['time'].values
        ax00.fill_between(x0, y01, y02, color=cm.colors[ii], alpha=0.2, label=f'{genotype} {label}',
                          linewidth=line_width)
        ax00.plot(x0, ec_results_median['tBID_obs'].values, color=cm.colors[ii], alpha=0.2)
        ii += 1
    ax00.set_xlabel('time [s]')
    ax00.set_ylabel('tBID Concentration [Copies per cell]')
    ax00.legend()
    ax00.tick_params(axis='x', which='major', labelsize=tick_labels_x)
    ax00.tick_params(axis='y', which='major')
    return ax00

ax = plot_dynamics(exp_cond_100ng_mL_results, 'Genotype')
ax.set_title('Predicted BID truncation dynamics in response to 100ng/mL TRAIL')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

# plot WT tBID dynamics as TRAIL concentration dependent
exp_wt_params = exp_cond_parameters[exp_cond_parameters['Genotype'].str.contains('WT')]\
    .reset_index(drop=True).reset_index().rename(columns={'index': 'simulation'})
# add numeric values for TRAIL concentration as a separate column
exp_wt_params['TRAIL_Conc_numeric'] = exp_wt_params['TRAIL_Conc'].\
    apply(lambda row: float(re.findall('(\d*\.*\d*)', row)[0]))
sim.param_values = exp_wt_params

exp_cond_wt_results = sim.run(tspan=np.linspace(0, 36000, 100))\
    .opt2q_dataframe.reset_index().rename(columns={'index': 'time'})
exp_cond_wt_results['time'] = exp_cond_wt_results['time'].apply(lambda row: row/60.0)
ax = plot_dynamics(exp_cond_wt_results, 'TRAIL_Conc_numeric', label='ng/mL')
ax.set_title('Predicted BID truncation dynamics in WT cells and varying TRAIL concentrations')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

# Plot WT and tBID dynamics for 25ng/mL (For comparing accuracy to other sources).
exp_cond_wt_25ngml_results = exp_cond_wt_results[exp_cond_wt_results['TRAIL_Conc'].str.contains('25 ng/mL') &
                                                 ~exp_cond_wt_results['TRAIL_Conc'].str.contains('.25 ng/mL')]

ax = plot_dynamics(exp_cond_wt_25ngml_results, 'TRAIL_Conc_numeric', label='posterior')
ax.set_title('Predicted BID truncation dynamics in response to 25ng/mL TRAIL')
ax.set_xlabel('time [min]')
plt.show()

# ---- Figure 6C (Probit Model Only) ----
# Plot Phenotype Phase Diagram (Probit Model of Cell Viability only)
# plot posterior prediction of the 50% cell death line
# plot posterior prediction (KDE) of each TRAIL concentration value in the scatter plot.

# Plot population parameter vs. concentration (Probit Model of Cell Viability only)

# ---- Figure 6D ----
# Plot posterior prediction of feature importance
fig11, ax11 = plt.subplots()
for pid in range(44, 53):
    sns.kdeplot(parameter_posteriors[:, pid][np.random.randint(0, len(parameter_posteriors), 1000)],
                ax=ax11, color=cm.colors[pid % len(cm.colors)], alpha=0.6, label=parameter_names[pid])
ax11.set_title('Posterior prediction of Feature Importance')
ax11.set_xlabel('Feature Coefficient')
ax11.set_ylabel('Density')
plt.legend(bbox_to_anchor=(1.5, 0., 0.5, 0.5))
plt.show()

# ---- Figure 6B ----
# Plot Training Dataset and Predictions
preprocessed_sims = []
cell_viability_predictions = []
test_set_cell_viability_predictions = []

for param_row in parameter_posteriors[np.random.randint(0, len(parameter_posteriors), 100), :]:
    new_param_values = update_params(param_row[:44])
    # some how the TAT_Bid parameters are getting overwritten.
    # Manually changing TAT_Bid Bid_0 from 40000.0 default to 48000.0 (i.e., 20% increased Bid as reported by Orzechowska_E_2014)
    new_param_values.loc[(new_param_values['Publication']=='Orzechowska_E_2014') & (new_param_values['Genotype']=='TAT_Bid'), 'Bid_0'] = 48000.0

    sim.param_values = new_param_values
    sr = sim.run().opt2q_dataframe.reset_index()  # simulate dynamics (we need the whole dataset)

    # preprocess (including standardize the data)
    cps_res = critical_points.transform(
            sr[['time', 'tBID_obs', 'simulation'] + merge_cols])
    hmp_res = max_derivative_points.transform(sr[['time', 'tBID_obs', 'simulation'] + merge_cols])

    # combine critical points and half max points
    cps_res = cps_res.merge(hmp_res, on=merge_cols, how='outer')

    preprocessed_res = standardize.transform(cps_res.loc[:, ~cps_res.columns.str.contains('simulation')])
    preprocessed_res.fillna(0.0, inplace=True)  # The fillna needed to be after the cps_res line

    preprocessed_sims.append(preprocessed_res)  # save the preprocessed data into a concatenated dataframe

    # Observations
    y_pred_trimmed = preprocessed_res[preprocessed_res['Publication'].str.contains('Wajant')]
    data_trimmed = data_df[data_df['Publication'].str.contains('Wajant')]

    y = data_trimmed[['Percent Cell Viability']].values
    y_sig = data_trimmed['CV_Std_dev'].values / np.std(y)

    # Observations - Test set
    y_pred_trimmed_test = preprocessed_res[preprocessed_res['Publication'].str.contains('Orzechowska')]
    data_trimmed_test = data_df[data_df['Publication'].str.contains('Orzechowska')]

    # Measurement Model
    # match extracted features to feature coefficient parameters
    current_features = list(preprocessed_res.loc[:, preprocessed_res.columns.str.contains('tBID')].columns)
    features_as_named_in_calibration = ['dtBID_obs__max', 'dtBID_obs__max__t', 'dtBID_obs__p_poi__0', 'tBID_obs__min__0',
                    'tBID_obs__min__not_nan__0', 'tBID_obs__min__time__0', 'tBID_obs__p_poi__0',
                    'tBID_obs__p_poi__not_nan__0', 'tBID_obs__p_poi__time__0', 'unused feature', 'gp_constant']
    if set(current_features) - set(features_as_named_in_calibration) != set():
        continue
    # feature_params = [param_row[44:53][parameter_names[44:53].index(feature)] for feature in current_features]
    feature_params = [param_row[44:53][features_as_named_in_calibration.index(feature)] for feature in current_features]

    # cell viability predictions
    if calibration_measurement_model == 'gaussian_process_model':
        from proportion_cell_fate_measurement_model.gaussian_process_model import GaussianProcessModel
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Constant

        m_diag = [p ** -2 for p in feature_params]
        kernel_ = Constant(50.0, (1e-10, 1e10)) * RBF(m_diag, (1e-10, 1e10))

        mm_ = GaussianProcessModel(current_features, kernel=kernel_, normalize_y=True, alpha=y_sig ** 2,
                                   optimizer=None, y_variable='Percent Cell Viability')

        x_y_combined = y_pred_trimmed.merge(data_trimmed, on=merge_cols, how='outer')
        x_y_combined_test = y_pred_trimmed_test.merge(data_trimmed_test, on=merge_cols, how='outer')

        mm_.gp.kernel.theta = [param_row[54]] + m_diag

        # use measurement model to predict cell viability
        cv_prediction = mm_.transform(x_y_combined[current_features + merge_cols],
                                      y=x_y_combined[[mm_.y_var] + merge_cols])
        cell_viability_predictions.append(cv_prediction)  # training set

        cv_prediction_test = mm_.predict(x_y_combined_test[current_features + merge_cols])
        cv_prediction_test = pd.concat([pd.DataFrame(cv_prediction_test[0], columns=['Percent Cell Viability']),
                                        x_y_combined_test[merge_cols]], axis=1)
        test_set_cell_viability_predictions.append(cv_prediction_test)  # test set

cell_viability_predictions = pd.concat(cell_viability_predictions)
cell_viability_predictions['TRAIL_Conc_numeric'] = cell_viability_predictions['TRAIL_Conc'].\
    apply(lambda row: float(re.findall('(\d*\.*\d*)', row)[0]))

fig22, ax22 = plt.subplots()
sns.violinplot(x="TRAIL_Conc_numeric", y="Percent Cell Viability", hue='Genotype', ax=ax22, inner=None, palette="muted",
               data=cell_viability_predictions, linewidth=0)
ax22.set_xlabel('TRAIL Concentration [ng/mL]')
ax22.set_title('Posterior Predictions of Cell Viability')
ax22.xaxis.set_tick_params(labelsize=8)
ax22.set_ylim((-20., 150.))
plt.show()

# Plot Test Dataset and Predictions
test_set_cell_viability_predictions = pd.concat(test_set_cell_viability_predictions)

fig3, ax3 = plt.subplots()
sns.violinplot(x="TRAIL_Conc", y="Percent Cell Viability", hue='Genotype', ax=ax3, inner=None,
               palette="muted", data=test_set_cell_viability_predictions, linewidth=0)
ax3.set_xlabel('TRAIL Concentration [ng/mL]')
ax3.set_title('Posterior Predictions of Cell Viability')
ax3.legend(loc='lower right')
plt.show()
