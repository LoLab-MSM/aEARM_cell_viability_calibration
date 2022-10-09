# Michael W. Irvin
# 11 July 2021
# Suite of preprocessing functions
# See example.setup_calibration for an example of how to use the preprocessing functions.

import pandas as pd
import numpy as np
from opt2q.measurement.base.functions import fast_linear_interpolate_fillna


def max_derivative(x, wrt=0, atol=1e-8):
    dx1 = derivative(x, wrt)

    ll = []
    for col in dx1.loc[:, ~dx1.columns.isin([wrt, 'simulation'])].columns:
        # Get row where col is max
        _max_data = list(dx1.loc[dx1[col].argmax()][[col, wrt]])

        # convert to 1-row dataframe
        _max_cols = [f'd{col}__max', f'd{col}__max__t'] # for n, m in enumerate(list_wrt_at_max_zeros)]
        ll.append(pd.DataFrame([_max_data], columns=_max_cols))

    return pd.concat(ll, axis=1)


def critical_points(x, wrt=0, atol=1e-8):
    """
    Returns local max, min and points of inflection for every column in a array.

    :param x: array (e.g. pd.Dataframe)
    :param wrt: str column name that is designated the independent variable. This column is not scaled.
    :param atol: tolerance of linear interpolator
        (low values can cause divide by zero errors and misinterpreting flat lines as critical points)
    :return: pd.Dataframe
    """
    x_ = x.reset_index(drop=True)
    cp = [relative_max_min_points(x_, wrt, atol=atol), inflection_points(x_, wrt, atol=atol)]

    return pd.concat(cp, axis=1)


def relative_max_min_points(x, wrt=0, atol=1e-8):
    dx1 = derivative(x, wrt)

    rmm = []
    for col in x.loc[:, x.columns != wrt].columns:
        lmm = local_pts_per_column(x, dx1, col, wrt, atol=atol)
        rmm.append(lmm)

    return pd.concat(rmm, axis=1)


def inflection_points(x, wrt=0, atol=1e-8):
    dx1 = derivative(x, wrt)
    dx2 = derivative(dx1, wrt)

    dx1_rn = dx1.rename(columns={c: f'd{c}' for c in x.loc[:, x.columns != wrt].columns})
    dx2_rn = dx2.rename(columns={c: f'd{c}' for c in x.loc[:, x.columns != wrt].columns})

    rmm = []
    for col in x.loc[:, x.columns != wrt].columns:
        poi = local_pts_per_column(x, dx2, col, wrt, pts_type='dx2', atol=atol)
        dx_at_poi = local_pts_per_column(dx1_rn, dx2_rn, f'd{col}', wrt, pts_type='dx2', atol=atol)

        rmm.append(poi)
        rmm.append(dx_at_poi.loc[:, ~dx_at_poi.columns.str.contains(f'__{wrt}__', case=False) &
                                    ~dx_at_poi.columns.str.contains('__not_nan__', case=False)])

    return pd.concat(rmm, axis=1)


def derivative(x, wrt=0):
    dif = pd.DataFrame(np.gradient(x, axis=0, edge_order=2), columns=x.columns)
    dx1 = dif.loc[:, dif.columns != wrt].div(dif[wrt], axis=0)
    dx1[wrt] = x[wrt].values
    return dx1


def find_crosses_indices(s, atol=1e-8):
    idx = np.roll(s, 1) * s < 0 - atol  # avoids divide by zero errors
    idx[0], idx[-1] = False, False
    idx = idx | np.roll(idx, -1)
    return idx[:-1]


def local_pts_per_column(x, dx, col, wrt, pts_type='dx', atol=1e-8):
    _x = x.reset_index(drop=True)
    _dx = dx.reset_index(drop=True)
    idx = find_crosses_indices(_dx[col], atol=atol)
    # find t when dx1 = 0
    # create a array with Nan for wrt where col is 0.
    dx1_i = _dx[[col, wrt]].loc[idx].values
    maxes = dx1_i[::2, 0] > 0
    dx1_nans = np.insert(dx1_i, range(len(dx1_i))[1::2], [0, np.nan], 0)
    interp_idx = np.argwhere(np.isnan(dx1_nans))

    if len(dx1_nans) > 1:
        t_at_dx1_zeros = fast_linear_interpolate_fillna(dx1_nans, interp_idx)[1::3, 1]
        lmm_i = _x[[col, wrt]].loc[idx].values

        lmm_nans = np.insert(lmm_i, range(len(lmm_i))[1::2], [np.nan, np.nan], 0)
        lmm_nans[1::3, 1] = t_at_dx1_zeros
        lmm_nans = lmm_nans[:, ::-1]  # reverse columns

        interp_idx_lmm = np.argwhere(np.isnan(lmm_nans))
        local_max_min = fast_linear_interpolate_fillna(lmm_nans, interp_idx_lmm)
        lmm_df = pd.DataFrame(local_max_min[1::3, :], columns=[wrt, col])
        lmm_df['nn'] = np.full(len(lmm_df), 1.0)

        # Flatten Df and Name columns
        lmm_flattened = lmm_df.values.flatten()
        lmm_columns = name_max_min_columns(lmm_flattened, col, wrt, maxes, pts_type)
        return pd.DataFrame([lmm_flattened], columns=lmm_columns)
    else:
        return pd.DataFrame()  # No local max or min in this column


def name_max_min_columns(flattened_array, col, wrt, dx_signs, points_type='dx'):
    tags = [f'__{wrt}', '', '__not_nan']
    if points_type == 'dx':
        lmm_columns = [f'{col}__{"max" if dx_signs[i // 3] else "min"}{tags[i % 3]}__{i // 3}'
                       for i in range(len(flattened_array))]
    elif points_type == 'dx2':
        lmm_columns = [f'{col}__{"p_poi" if dx_signs[i // 3] else "n_poi"}{tags[i % 3]}__{i // 3}'
                       for i in range(len(flattened_array))]
    else:
        lmm_columns = [f'{col}__{"lmm"}{tags[i % 3]}__{i // 3}'
                       for i in range(len(flattened_array))]
    return lmm_columns
