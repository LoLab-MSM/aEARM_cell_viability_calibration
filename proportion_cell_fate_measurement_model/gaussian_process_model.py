# Michael W. Irvin
# 19 July 2021
# Gaussian Process Model of Proportional Cell Fate Data

# Equation 15.19 of Murphy, K.P. (2012) Machine Learning: A probabilistic perspective
# The radial basis kernel with function change /sigma_f and measurement noise /sigma_y is optimized along with
# the parameters of the mechanistic model that gives X.

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Constant
from opt2q.measurement.base import Transform


class GaussianProcessModel(Transform):
    """
    Parameters
    ----------
    feature_list:ist of strings
        columns in `x` that are regard as features (the order must match that of the coefficients).
        the columns of `x` named in feature_list must be numeric columns.

    do_fit_transform: bool, optional
        When True, Simply fit the Gaussian process to values of ``x`` and data ``y``. When False, use a previous fit for
        predictions; useful for prior predictions. Defaults to True.

    gp_kwargs: arguments provided to the scikit-learn `GaussianProcessRegressor` class.
        Defaults to `GaussianProcessRegressor` class defaults -- Except for optimizer!
        The default optimizer is set to None. This allows estimation of kernel parameters using external optimizers (e.g.
        PyDREAM).
    """

    def __init__(self, feature_list, do_fit_transform=True, y_variable='percent cell fate', **gp_kwargs):
        super(GaussianProcessModel).__init__()
        self.feature_list = feature_list
        self.do_fit_transform = do_fit_transform is True or not isinstance(do_fit_transform, bool)  # non-bool -> True
        _optimizer = gp_kwargs.pop('optimizer', None)
        gp_kwargs.update(optimizer=_optimizer)
        self.gp = self.build_gp(gp_kwargs)
        self.set_params_fn = {'feature_list': self._set_feature_list}
        self.y_var = y_variable
        self.y_train = None
        self.x_train = None

    def _set_feature_list(self, feature_list):
        self.feature_list = feature_list

    @staticmethod
    def build_gp(kw):
        return GaussianProcessRegressor(**kw)

    def fit(self, x, y):
        self.y_train = y[[self.y_var]].values
        self.x_train = x[self.feature_list].values
        self.gp.fit(self.x_train, self.y_train)

    def predict(self, x, **kwargs):
        kw = dict()
        if 'return_cov' in kwargs:
            kw.update(return_cov=kwargs.get('return_cov'))
        else:
            kw.update(return_std=kwargs.get('return_std', True))
        return self.gp.predict(x[self.feature_list].values, **kw)

    def transform(self, x, **kwargs):
        """
        Predicts target variable as modeled by the Gaussian Process Regressor.

        :param x: pd.DataFrame with columns mentioned in the `feature_list`
        :param kwargs:
            y: pd.DataFrame (optional if not doing fit-transform). Single column data rows match that in `x`.
            update_training_set: bool, optional defaults to False
                If True, uses `x` and `y` to as training data in the fit-transform.
        :return: predictions
        """
        x_extra_columns = list(set(x.columns) - set(self.feature_list))
        y_var = kwargs.get('y_variable', self.y_var)
        update_training_set = kwargs.get('update_training_set', False)

        if self.do_fit_transform:
            # Fit using self.x_train and self.y_train
            if not update_training_set and (self.x_train is not None and self.y_train is not None):
                self.gp.fit(self.x_train, self.y_train)

            elif 'y' in kwargs:
                y = kwargs.get('y')
                self.y_train = y[[self.y_var]].values
                self.x_train = x[self.feature_list].values
                self.gp.fit(self.x_train, self.y_train)

            else:
                raise ValueError('Must supply y if do_fit_transform is True')

        y_res = self.predict(x, **kwargs)

        if len(y_res) == 2:
            y_ = pd.concat([pd.DataFrame(y_res[0], columns=[y_var]),
                            pd.DataFrame(y_res[1], columns=[f'{y_var}_std']),
                            x[x_extra_columns]], axis=1)
        else:
            y_ = pd.concat([pd.DataFrame(y_res, columns=[y_var]),
                            x[x_extra_columns]], axis=1)
        return y_

    def likelihood(self, x, y):
        """Return Log-Marginal Likelihood of the Gaussian Process Model"""

        y_train = y[[self.y_var]].values
        x_train = x[self.feature_list].values
        self.gp.fit(x_train, y_train)
        return self.gp.log_marginal_likelihood(self.gp.kernel.theta)


if __name__ == '__main__':
    from example.cell_viability.load_data import data_df
    from example.cell_viability.setup_calibration import starting_feature_list, standardize
    from example.cell_viability.setup_calibration import preprocessed_results as preprocess_start
    from opt2q.simulator import Simulator
    from opt2q_examples.apoptosis_model import model
    from example.cell_viability import sample_params
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from opt2q.measurement.base.transforms import ScaleGroups
    from proportion_cell_fate_measurement_model.core.preprocessing import critical_points as cp
    from proportion_cell_fate_measurement_model.core.preprocessing import half_maximal_point as hmp

    np.random.seed(1)

    # ----------------------------------------------------------------------
    # Observations
    y = data_df[data_df['Genotype'] == 'WT'][['Percent Cell Viability']].values
    y_sig = data_df[data_df['Genotype'] == 'WT']['CV_Std_dev'].values/np.std(y)

    # Measurement Model
    kernel = Constant(50.0, (1e-5, 1e5)) * RBF(np.ones(len(starting_feature_list)), (1e-3, 1e3))

    gaussian_process = GaussianProcessModel(starting_feature_list, kernel=kernel, normalize_y=True, alpha=y_sig**2,
                                            optimizer=None, y_variable='Percent Cell Viability')
    gaussian_process.transform(preprocess_start[data_df['Genotype'] == 'WT'],
                               y=data_df[data_df['Genotype'] == 'WT'][['Percent Cell Viability']])
    gaussian_process.likelihood(preprocess_start[data_df['Genotype'] == 'WT'],
                                y=data_df[data_df['Genotype'] == 'WT'][['Percent Cell Viability']])

    # Make the prediction on the meshed x-axis
    sim = Simulator(model, tspan=np.linspace(0, 21600, 100), solver='scipyode',
                    solver_options={'integrator': 'lsoda'}, integrator_options={'mxstep': 2 ** 20})
    n = 25

    sim.param_values = pd.concat([pd.DataFrame(np.logspace(1, np.log10(12000), n), columns=['L_0']),
                                  pd.DataFrame(np.repeat(
                                      np.array([10 ** p for p in sample_params.x[:-2]]).reshape(1, -1), n, axis=0),
                                      columns=[p.name for p in model.parameters_rules()])
                                  ], axis=1)
    new_results = sim.run().opt2q_dataframe.reset_index()
    critical_points = ScaleGroups(columns=['time', 'tBID_obs'], drop_columns=True, groupby='simulation',
                                  scale_fn=cp, **{'wrt': 'time', 'atol': 1e-6})  # Step 1A: Critical Points
    cps_results = critical_points.transform(new_results[['time', 'tBID_obs', 'simulation']])

    half_max_points = ScaleGroups(columns=['time', 'tBID_obs'], drop_columns=True, groupby='simulation',
                                  scale_fn=hmp, **{'wrt': 'time', 'atol': 1e-6})  # Step 1B: Half-Max Points
    hmp_results = half_max_points.transform(new_results[['time', 'tBID_obs', 'simulation']])

    cps_results = cps_results.merge(hmp_results, on='simulation', how='outer')

    standardize.set_params(do_fit_transform=False)
    preprocessed_results = standardize.transform(cps_results)
    preprocessed_results.fillna(0.0, inplace=True)

    # Predict Cell Viability
    th = np.array([3.73697885e-01, 6.90775528e+00, 1.19351513e-07, 6.90775528e+00,
                   9.02593913e-01, 0.00000000e+00, 1.19422373e-01])
    gaussian_process.gp.kernel.theta = th
    y_res1 = gaussian_process.transform(preprocessed_results)
    y_pred1 = y_res1[['Percent Cell Viability']].values
    sigma = y_res1['Percent Cell Viability_std'].values

    th2 = th * 1e-10
    gaussian_process.gp.kernel.theta = th2
    y_res2 = gaussian_process.transform(preprocessed_results)
    y_pred2 = y_res2[['Percent Cell Viability']].values
    sigma2 = y_res2['Percent Cell Viability_std'].values

    # Plot Results
    plt.figure()
    x_data = [float(x_) for x_ in data_df[data_df['Genotype'] == 'WT']['TRAIL_Conc'].str.extract('(\d*\.*\d*)').iloc[:, 0]]

    x_data[0] = 0.1
    cm = plt.get_cmap('tab10')
    plt.errorbar(x_data, y, y_sig*np.std(y), fmt='o', capsize=5, color=cm.colors[1], label='WT')
    plt.plot(np.logspace(-1, np.log10(200), n), y_pred1, color=cm.colors[1], alpha=0.7, label='Prediction')
    plt.fill(np.concatenate([np.logspace(-1, np.log10(200), n), np.logspace(-1, np.log10(200), n)[::-1]]),
             np.concatenate([y_pred1[:, 0] - 1.9600 * sigma,
                             (y_pred1[:, 0] + 1.9600 * sigma)[::-1]]),
             alpha=.4, fc=cm.colors[1], ec='None', label='95% confidence interval')
    plt.fill(np.concatenate([np.logspace(-1, np.log10(200), n), np.logspace(-1, np.log10(200), n)[::-1]]),
             np.concatenate([y_pred2[:, 0] - 1.9600 * sigma2,
                             (y_pred2[:, 0] + 1.9600 * sigma2)[::-1]]),
             alpha=.4, fc=cm.colors[0], ec='None', label='95% confidence interval')
    plt.xlabel('TRAIL Conc [ng/mL]')
    plt.ylabel('Cell Viability')
    plt.xscale('log')
    plt.title('Gaussian Process Model of Cell Viability in Hela Cells')
    plt.legend(loc='upper right')
    plt.show()
