import unittest
from proportion_cell_fate_measurement_model.core import preprocessing as pp
import pandas as pd
import numpy as np


class PreprocessingTestCase(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, np.pi, 10)
        self.xin = pd.DataFrame({'t': self.x, 'x': np.cos(self.x), 'y': np.sin(self.x)})
        self.dx1 = pp.derivative(self.xin, wrt='t')

    def tearDown(self):
        self.x = None
        self.xin = None

    def test_derivative(self):
        test = pp.derivative(self.xin, wrt='t')
        target = pd.DataFrame([
             [-0.010419,  1.038906, 0.000000],
             [-0.335117,  0.920725, 0.349066],
             [-0.629813,  0.750582, 0.698132],
             [-0.848545,  0.489908, 1.047198],
             [-0.964930,  0.170143, 1.396263],
             [-0.964930, - 0.170143,  1.745329],
             [-0.848545, - 0.489908,  2.094395],
             [-0.629813, - 0.750582,  2.443461],
             [-0.335117, - 0.920725,  2.792527],
             [-0.010419, - 1.038906,  3.141593]], columns=['x', 'y', 't'])
        pd.testing.assert_frame_equal(test[['t', 'x', 'y']], target[['t', 'x', 'y']], atol=1e-6)

    def test_local_min_per_column_case_0(self):
        test = pp.local_pts_per_column(self.xin, self.dx1, 'y', 't')
        target = pd.DataFrame([[1.57080, 0.98481, 1.00000]],
                              columns=['y__max__t__0', 'y__max__0', 'y__max__not_nan__0'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], atol=1e-6)

    def test_local_min_per_column_case_1(self):  # No local max or min
        test = pp.local_pts_per_column(self.xin, self.dx1, 'x', 't')
        target = pd.DataFrame()
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], atol=1e-6)

    def test_relative_max_min_points(self):
        test = pp.relative_max_min_points(self.xin, 't')
        target = pd.DataFrame([[1.57080, 0.98481, 1.00000]],
                              columns=['y__max__t__0', 'y__max__0', 'y__max__not_nan__0'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], atol=1e-6)

    def test_inflection_points(self):
        test = pp.inflection_points(self.xin, 't')
        target = pd.DataFrame([[1.57080, 0.00000, 1.00000, -0.96493]],
                              columns=['x__n_poi__t__0', 'x__n_poi__0', 'x__n_poi__not_nan__0', 'dx__n_poi__0'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], atol=1e-6)

    def test_critical_points(self):
        test = pp.critical_points(self.xin, 't')
        target = pd.DataFrame([[1.57080, 0.98481, 1.00000, 1.57080, 0.00000, 1.00000, -0.96493]],
                              columns=['y__max__t__0', 'y__max__0', 'y__max__not_nan__0', 'x__n_poi__t__0',
                                       'x__n_poi__0', 'x__n_poi__not_nan__0', 'dx__n_poi__0'])
        pd.testing.assert_frame_equal(test[test.columns], target[test.columns], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
