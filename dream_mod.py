import pydream.core
import traceback
import numpy as np


def _sample_dream(args):
    try:
        dream_instance = args[0]
        iterations = args[1]
        start = args[2]
        verbose = args[3]
        nverbose = args[4]
        chain_idx = args[5]
        naccepts_iterations_total = args[6]
        step_fxn = getattr(dream_instance, 'astep')
        sampled_params = np.empty((iterations, dream_instance.total_var_dimension))
        log_ps = np.empty((iterations, 1))
        acceptance_rates_size = int(np.floor(iterations / nverbose))
        if acceptance_rates_size == 0:
            acceptance_rates_size = 1
        acceptance_rates = np.zeros(acceptance_rates_size)
        q0 = start
        iterations_total = np.sum(naccepts_iterations_total[1])
        naccepts = naccepts_iterations_total[0][-1]
        naccepts100win = 0
        acceptance_counter = 0
        for iteration_idx, iteration in enumerate(range(iterations_total, iterations_total + iterations)):
            if iteration%nverbose == 0:
                acceptance_rate = float(naccepts)/(iteration+1)
                acceptance_rates[acceptance_counter] = acceptance_rate
                acceptance_counter += 1
                if verbose:
                    print('Iteration: ',iteration,' acceptance rate: ',acceptance_rate)
                if iteration%100 == 0:
                    acceptance_rate_100win = float(naccepts100win)/100
                    if verbose:
                        print('Iteration: ',iteration,' acceptance rate over last 100 iterations: ',acceptance_rate_100win)
                    naccepts100win = 0
            old_params = q0
            sampled_params[iteration_idx], log_prior, log_like = step_fxn(q0)
            log_ps[iteration_idx] = log_like + log_prior
            q0 = sampled_params[iteration_idx]
            if old_params is None:
                old_params = q0

            if np.any(q0 != old_params):
                naccepts += 1
                naccepts100win += 1

        naccepts_iterations_total = np.append(naccepts_iterations_total, np.array([[naccepts], [iterations]]), axis=1)
        np.save(f'{dream_instance.model_name}naccepts_chain{chain_idx}.npy', naccepts_iterations_total)
        print("Saving the feature")
        print(f'{dream_instance.model_name}_feature_list_{chain_idx}.txt')
        np.savetxt(f'{dream_instance.model_name}_feature_list_{chain_idx}.txt',
                   dream_instance.model.likelihood.feature_list, delimiter=", ", fmt="%s")

    except Exception as e:
        traceback.print_exc()
        print()
        raise e

    return sampled_params, log_ps, acceptance_rates


pydream.core._sample_dream = _sample_dream
run_dream = pydream.core.run_dream
