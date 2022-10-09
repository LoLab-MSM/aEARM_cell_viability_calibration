# Michael W. Irvin
# 22 July 2021
# Calibrate Apoptosis Reaction Model to Cell Viability Data
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import datetime as dt
from scipy.stats import norm, truncnorm, laplace, halfnorm
from pydream.parameters import SampledParam
from pydream.core import run_dream
from pydream import Dream_shared_vars
from pydream.convergence import Gelman_Rubin
from opt2q.calibrator import objective_function
from example.cell_viability import load_data
from example.cell_viability.load_data import data_df
from example.cell_viability.setup_calibration import sim, critical_points, max_derivative_points, standardize, len_mechanism_params, \
    starting_feature_list
from example.cell_viability import sample_params


measurement_model = 'gaussian_process_model'
max_len_features = 10

# Use smaller dataset
only_wajant_dataset = True
only_roux_dataset = False
both_datasets = True if only_wajant_dataset and only_roux_dataset else False

# Use larger burn-in
larger_burn_in = True


# Priors
def make_priors(m_model):
    prior_samples = [SampledParam(norm, loc=sample_params.x, scale=1.5)]      # log(rate background params) float
    prior_samples += [SampledParam(norm, loc=2.3, scale=0.113) for i in range(4)]  # sensitivity params
    prior_samples += [SampledParam(norm, loc=0, scale=1.5)]  # Luteolin effect
    prior_samples += [SampledParam(halfnorm, loc=0, scale=1.5) for i in range(2)]  # caspase-inh effects
    prior_samples += [SampledParam(truncnorm, a=(0-i)/1.5, b=np.inf, loc=i, scale=1.5) for i in range(2)] # FADD KD effects
    prior_samples += [SampledParam(halfnorm, loc=0, scale=1.5)]  # Bortezomib effect
    # neg (log exp. con. params) float

    if m_model == 'gaussian_process_model':
        prior_samples += [SampledParam(laplace, loc=0.0, scale=0.1) for i in range(max_len_features)]  # coefficients
        prior_samples += [SampledParam(laplace, loc=0.0, scale=50)]  # constant

    return prior_samples


sampled_params_0 = make_priors(measurement_model)

if measurement_model == 'gaussian_process_model':
    from proportion_cell_fate_measurement_model.gaussian_process_model import GaussianProcessModel
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Constant

# Calibration Settings
n_chains = 6
n_iterations = 100000    # iterations per file-save
burn_in_len = 1000000 if larger_burn_in else 200000     # number of iterations during burn-in
max_iterations = 400000
now = dt.datetime.now()
model_name = f'cell_viability_calibration_{measurement_model}_{now.year}{now.month}{now.day}'


Dream_shared_vars.feature_list = starting_feature_list


# Likelihood Function
@objective_function(feature_list=starting_feature_list, return_results=False, evals=0)
def likelihood(x):
    # update mechanistic model parameters
    new_params = load_data.update_params(x[:len_mechanism_params])

    # run mechanistic model
    sim.param_values = new_params
    sim_res = sim.run(num_processors=1).opt2q_dataframe.reset_index()

    # preprocess results
    try:
        cps_res = critical_points.transform(
            sim_res[['time', 'tBID_obs', 'simulation'] + load_data.merge_cols])
        hmp_res = max_derivative_points.transform(sim_res[['time', 'tBID_obs', 'simulation'] + load_data.merge_cols])

        # combine critical points and half max points
        cps_res = cps_res.merge(hmp_res, on=load_data.merge_cols, how='outer')

        preprocessed_res = standardize.transform(cps_res.loc[:, ~cps_res.columns.str.contains('simulation')])
        preprocessed_res.fillna(0.0, inplace=True)

    except (ValueError, ZeroDivisionError, TypeError, KeyError):
        return -1e10

    # which critical points features appear in `preprocess_res` depend on the mechanistic model parameters.
    # this section makes sure the measurement model parameters match the features that appear in the `preprocess_res`.
    # update feature list
    current_features = list(preprocessed_res.loc[:, preprocessed_res.columns.str.contains('tBID')].columns)
    new_features = sorted(list(set(current_features)-set(likelihood.feature_list)))
    updated_features = likelihood.feature_list + new_features
    if len(updated_features) > max_len_features:
        return -1e10

    likelihood.feature_list = updated_features
    Dream_shared_vars.feature_list = likelihood.feature_list

    # the features are indexed in the same order as likelihood.feature_list
    feature_params = [x[len_mechanism_params + likelihood.feature_list.index(feature)] for feature in current_features]

    # measurement model
    if measurement_model == 'gaussian_process_model':
        m_diag = [p**-2 for p in feature_params]  # The kernel uses M: the diagonal of l**-2  (see notes)
        const = x[len_mechanism_params + max_len_features]

        # Because the structure of the kernel changes with the dimensionality of the features, GP model has to be
        # re-instantiated
        kernel_ = Constant(50.0, (1e-10, 1e10)) * RBF(m_diag, (1e-10, 1e10))

        # Exclude data from likelihood function
        # Trim data_df and y_predicted to use only certain data
        if both_datasets:
            y_pred_trimmed = preprocessed_res[preprocessed_res['Publication'].str.contains('Wajant') &
                                              preprocessed_res['Publication'].str.contains('Roux')]
            data_trimmed = data_df[data_df['Publication'].str.contains('Wajant') &
                                   data_df['Publication'].str.contains('Roux')]

        elif only_wajant_dataset:
            y_pred_trimmed = preprocessed_res[preprocessed_res['Publication'].str.contains('Wajant')]
            data_trimmed = data_df[data_df['Publication'].str.contains('Wajant')]

        elif only_roux_dataset:
            y_pred_trimmed = preprocessed_res[preprocessed_res['Publication'].str.contains('Roux')]
            data_trimmed = data_df[data_df['Publication'].str.contains('Roux')]

        else:
            y_pred_trimmed = preprocessed_res
            data_trimmed = data_df

        # Observations
        y = data_trimmed[['Percent Cell Viability']].values
        y_sig = data_trimmed['CV_Std_dev'].values / np.std(y)

        mm_ = GaussianProcessModel(current_features, kernel=kernel_, normalize_y=True, alpha=y_sig ** 2,
                                   optimizer=None, y_variable='Percent Cell Viability')

        x_y_combined = y_pred_trimmed.merge(data_trimmed, on=load_data.merge_cols, how='outer')
        likelihood.y_data_predicted_combined = x_y_combined

        mm_.gp.kernel.theta = [const] + m_diag

        try:
            log_likelihood = mm_.likelihood(x_y_combined[current_features + load_data.merge_cols],
                                            y=x_y_combined[[mm_.y_var] + load_data.merge_cols])
        except np.linalg.LinAlgError:
            return -1e10

    else:
        raise ValueError('Supports only "gaussian_process_model" measurement models')

    print(likelihood.evals)
    print(x)
    print(log_likelihood)
    likelihood.evals += 1
    return log_likelihood


if __name__ == '__main__':

    # -------- Calibration -------
    # Model Inference via PyDREAM
    ncr = 50
    gamma_levels = 8
    p_gamma_unity = 0.2
    lamb = 0.01
    de_pairs = 1
    zeta = 1e-15

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = n_iterations
    sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                       likelihood=likelihood,
                                       niterations=n_iterations,
                                       nchains=n_chains,
                                       multitry=False,
                                       nCR=ncr,
                                       gamma_levels=gamma_levels,
                                       adapt_gamma=True,
                                       p_gamma_unity=p_gamma_unity,
                                       history_thin=1,
                                       model_name=model_name,
                                       verbose=True,
                                       crossover_burnin=min(n_iterations, burn_in_len),
                                       DEpairs=de_pairs,
                                       lamb=lamb,
                                       zeta=zeta
                                       )

    # Save sampling output (sampled parameter values and their corresponding logps).
    def save_samples(sampled_params_, log_ps_):
        for chain in range(len(sampled_params_)):
            np.save(f'{model_name}_{str(chain)}_{str(total_iterations)}_parameters', sampled_params_[chain])
            np.save(f'{model_name}_{str(chain)}_{str(total_iterations)}_log_p', log_ps_[chain])
        # features = likelihood.feature_list
        # np.save(f'{model_name}_{str(total_iterations)}_features_list', features)
        np.savetxt(f'{model_name}_{str(total_iterations)}_features_list.txt',
                   Dream_shared_vars.feature_list, delimiter=", ", fmt="%s")
        gr_ = Gelman_Rubin(sampled_params_)
        np.savetxt(f'{model_name}_{str(total_iterations)}_GR.txt', gr_)
        burn_in_len_ = max(burn_in_len - n_iterations, 0)

        print('At iteration: ', total_iterations, ' GR = ', gr_)
        print(f'At iteration: {total_iterations}, {burn_in_len} steps of burn-in remain.')
        return gr_, burn_in_len_

    gr, burn_in_len = save_samples(sampled_params, log_ps)

    old_samples = sampled_params
    if np.isnan(gr).any() or np.any(gr > 1.2):
        while not converged or (total_iterations < max_iterations):
            starts = [sampled_params[chain][-1, :] for chain in range(n_chains)]

            total_iterations += n_iterations
            sampled_params, log_ps = run_dream(parameters=sampled_params_0,
                                               likelihood=likelihood,
                                               niterations=n_iterations,
                                               nchains=n_chains,
                                               multitry=False,
                                               nCR=ncr,
                                               gamma_levels=gamma_levels,
                                               adapt_gamma=True,
                                               p_gamma_unity=p_gamma_unity,
                                               history_thin=1,
                                               model_name=model_name,
                                               verbose=True,
                                               restart=True,  # restart at the last sampled position
                                               start=starts,
                                               crossover_burnin=min(n_iterations, burn_in_len),
                                               DEpairs=de_pairs,
                                               lamb=lamb,
                                               zeta=zeta
                                               )

            gr, burn_in_len = save_samples(sampled_params, log_ps)

            if np.all(gr < 1.2):
                converged = True
