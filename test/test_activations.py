import sys
import numpy as np

class TestActivations:
    def test_tanh(self):
        from darwin.activations import tanh_activation
        input_array = np.array(
            [[-0.39916268, -1.74727587], [-0.18221106, 0.16319185], [-1.9630198, -0.22021899]]
        )
        expected_results = np.array(
            [[-0.37923229, -0.9410647], [-0.18022096, 0.16175843], [-0.96131959, -0.21672678]]
        )
        assert np.allclose(tanh_activation(input_array), expected_results)

    def test_sigmoid(self):
        from darwin.activations import sigmoid_activation
        input_array = np.array(
            [[-0.39916268, -1.74727587], [-0.18221106, 0.16319185], [-1.9630198, -0.22021899]]
        )
        expected_results = np.array(
            [[0.40151353, 0.14839112], [0.45457285, 0.54070766], [0.12314061, 0.44516668]]
        )
        assert np.allclose(sigmoid_activation(input_array), expected_results)

    def test_sigmoid_tanh(self):
        from darwin.activations import sigmoid_tanh_activation
        input_array = np.array(
            [[-0.39916268, -1.74727587], [-0.18221106, 0.16319185], [-1.9630198, -0.22021899]]
        )
        expected_results = np.array(
            [[0.40151353, 0.14839112], [0.45457285, 0.54070766], [0.12314061, 0.44516668]]
        )
        assert np.allclose(sigmoid_tanh_activation(input_array), expected_results)

    def test_fast_sigmoid(self):
        from darwin.activations import fast_sigmoid_activation
        input_array = np.array(
            [[-0.39916268, -1.74727587], [-0.18221106, 0.16319185], [-1.9630198, -0.22021899]]
        )
        expected_results = np.array(
            [[-0.28528683, -0.63600306], [-0.15412735, 0.14029659], [-0.66250647, -0.18047497]]
        )
        assert np.allclose(fast_sigmoid_activation(input_array), expected_results)

    def test_relu(self):
        from darwin.activations import relu_activation
        input_array = np.array(
            [[-0.39916268, -1.74727587], [-0.18221106, 0.16319185], [-1.9630198, -0.22021899]]
        )
        expected_results = np.array([[-0., -0.], [-0., 0.16319185], [-0., -0.]])
        assert np.allclose(relu_activation(input_array), expected_results)

    def test_approximate_sigmoid(self):
        from darwin.activations import approximate_sigmoid_activation
        input_array = np.array(
            [[-0.39916268, -1.74727587], [-0.18221106, 0.16319185], [-1.9630198, -0.22021899]]
        )
        expected_results = np.array(
            [[0.4998005, 0.49912789], [0.49990891, 0.50008158], [0.49902041, 0.49988991]]
        )
        assert np.allclose(approximate_sigmoid_activation(input_array), expected_results)

    def test_ramp(self):
        from darwin.activations import ramp_activation
        input_array = np.array(
            [[-0.39916268, -1.74727587], [-0.18221106, 0.16319185], [-1.9630198, -0.22021899]]
        )
        expected_results = np.array(
            [[-0.39916268, -1.], [-0.18221106, 0.16319185], [-1., -0.22021899]]
        )
        assert np.allclose(ramp_activation(input_array), expected_results)