import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from picaso import information_content as ic

def test_jacobian_perturbation_logic():
    # Mock driver_dict
    driver_dict = {
        'observation_type': 'thermal',
        'param1': 100.0,
        'param2': 1e-12,
        'param3': np.array([10.0, 20.0, 30.0]),
        'param4': np.array([1e-10, 1e-11, 1e-12])
    }
    
    # Mock run function to return a spectrum-like dict
    # We'll make it return different values based on the input to help us identify what happened
    def mock_run(driver_dict=None):
        # Return a simple array as 'thermal' spectrum
        # We can use the sum of parameters as a "spectrum" to track changes
        val1 = np.mean(driver_dict.get('param1', 0))
        val2 = np.mean(driver_dict.get('param2', 0))
        # Just return something that depends on the parameters
        return {'thermal': np.array([val1, val2])}

    with patch('picaso.information_content.run', side_effect=mock_run):
        # Test 1: Linear scalar (param1=100.0, d_param=0.01 -> actual_dp = 1.0)
        # deriv = ( (101.0 + 1e-12) - (100.0 + 1e-12) ) / 1.0 = 1.0
        jac = ic.jacobian(driver_dict=driver_dict, params=['param1'], d_param=0.01, is_log=False)
        assert np.isclose(jac[0, 0], 1.0) 
        
        # Test 2: Log scalar (param2=1e-12, d_param=0.01)
        # log10(val) = -12. actual_dp = |0.01 * -12| = 0.12
        # new_val = 10**(-12 + 0.12) = 10**-11.88
        # deriv = ( (100.0 + 10**-11.88) - (100.0 + 1e-12) ) / 0.12
        jac = ic.jacobian(driver_dict=driver_dict, params=['param2'], d_param=0.01, is_log=True)
        expected_actual_dp = 0.12
        expected_new_val = 10**(-12 + expected_actual_dp)
        expected_deriv = (expected_new_val - 1e-12) / expected_actual_dp
        assert np.isclose(jac[1, 0], expected_deriv)

        # Test 3: Linear array (param3=[10, 20, 30], d_param=0.01)
        # mean(val) = 20. actual_dp = 0.01 * 20 = 0.2
        # new_val = [10.2, 20.2, 30.2]
        # mean(new_val) = 20.2
        # deriv = ( (20.2 + mean(param2)) - (20.0 + mean(param2)) ) / 0.2 = 0.2 / 0.2 = 1.0
        # Wait, mock_run uses np.mean(driver_dict['param1']) which is param3 in this case if we pass it as params=['param3']
        # actually ic.jacobian will use 'param3' if we pass it.
        # But wait, my mock_run uses driver_dict.get('param1') etc.
        # If I want to test param3, I should make mock_run more generic.
        
    def mock_run_generic(driver_dict=None):
        # Sum of means of all numeric params
        s = 0
        for k, v in driver_dict.items():
            if isinstance(v, (int, float, np.ndarray)):
                s += np.mean(v)
        return {'thermal': np.array([s])}

    with patch('picaso.information_content.run', side_effect=mock_run_generic):
        # Test 3 again: Linear array
        # param3 = [10, 20, 30], mean=20, d_param=0.01 -> actual_dp = 0.2
        # new_val = [10.2, 20.2, 30.2], mean=20.2
        # spec_plus = 20.2 + others, base_spec = 20.0 + others
        # deriv = 0.2 / 0.2 = 1.0
        jac = ic.jacobian(driver_dict=driver_dict, params=['param3'], d_param=0.01, is_log=False)
        assert np.isclose(jac[0, 0], 1.0)

        # Test 4: Log array
        # param4 = [1e-10, 1e-11, 1e-12], log10 = [-10, -11, -12], mean_log = -11
        # actual_dp = |0.01 * -11| = 0.11
        # new_val = 10**([-10, -11, -12] + 0.11)
        # mean(new_val) = mean(10**([-9.89, -10.89, -11.89]))
        # base_spec_mean_param4 = mean([1e-10, 1e-11, 1e-12])
        # deriv = (mean(new_val) - base_spec_mean_param4) / 0.11
        jac = ic.jacobian(driver_dict=driver_dict, params=['param4'], d_param=0.01, is_log=True)
        val4 = driver_dict['param4']
        actual_dp = 0.11
        new_val4 = 10**(np.log10(val4) + actual_dp)
        expected_deriv = (np.mean(new_val4) - np.mean(val4)) / actual_dp
        assert np.isclose(jac[0, 0], expected_deriv)

if __name__ == "__main__":
    test_jacobian_perturbation_logic()
    print("Test passed!")
