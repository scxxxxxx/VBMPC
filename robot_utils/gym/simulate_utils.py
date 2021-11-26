import numpy as np


def discount_sum(x, gamma, discounted_terminal=0.0):
    """
    discount sum a sequence with terminal value
    """
    y = []
    run_sum = discounted_terminal
    for t in range( len(x)-1, -1, -1):
        run_sum = x[t] + gamma*run_sum
        y.append(run_sum)

    return np.array(y[::-1])


def gather_paths_parallel(env_name, start_state, base_act, filter_coefs, base_seed, goal, reward_type, reference, alpha, paths_per_cpu, num_cpu=None):
    num_cpu = mp.cpu_count() if num_cpu is None else num_cpu
    args_list = []
    for i in range(num_cpu):
        cpu_seed = base_seed + i*paths_per_cpu
        args_list_cpu = [env_name, start_state, paths_per_cpu, base_act, filter_coefs, cpu_seed, goal, reward_type, reference, alpha]
        args_list.append(args_list_cpu)

    # do multiprocessing
    results = _try_multiprocess(args_list, num_cpu, max_process_time=300, max_timeouts=4)
    paths = []
    for result in results:
        for path in result:
            paths.append(path)

    return paths


def _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts):
    # Base case
    if max_timeouts == 0:
        return None

    if num_cpu == 1:
        results = [generate_paths_star(args_list[0])]  # dont invoke multiprocessing unnecessarily
    else:
        pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
        parallel_runs = [pool.apply_async(generate_paths_star, args=(args_list[i],)) for i in range(num_cpu)]
        try:
            results = [p.get(timeout=max_process_time) for p in parallel_runs]
        except Exception as e:
            print(str(e))
            print("Timeout Error raised... Trying again")
            pool.close()
            pool.terminate()
            pool.join()
            return _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts - 1)

        pool.close()
        pool.terminate()
        pool.join()

    return results


def generate_paths_star(args_list):
    return generate_paths(*args_list)


def generate_paths(env_name, start_state, N, base_act, filter_coefs, base_seed, goal, reward_type, reference, alpha):
    """
    first generate enough perturbed actions
    then do rollouts with generated actions
    set seed inside this function for multiprocessing
    """
    np.random.seed(base_seed)
    act_list = []
    for i in range(N):
        act = generate_perturbed_actions(base_act, filter_coefs)
        act_list.append(act)
    paths = rollout(env_name, start_state, act_list, goal, reward_type, reference, alpha)
    return paths


def generate_perturbed_actions(base_act, filter_coefs):
    """
    Generate perturbed actions around a base action sequence
    """
    sigma, beta_0, beta_1, beta_2 = filter_coefs
    eps = np.random.normal(loc=0, scale=1.0, size=base_act.shape) * sigma
    for i in range(2, eps.shape[0]):
        eps[i] = beta_0*eps[i] + beta_1*eps[i-1] + beta_2*eps[i-2]
    return base_act + eps


def rollout(env, start_state, act_list, goal, alpha):
    paths = []
    horizon = act_list[0].shape[0]
    n_rollouts = len(act_list)
    for i in range(n_rollouts):
        if isinstance(goal, int):  # humanoid pybullet save state
            env.set_env_state(goal)
        else:
            env.set_env_state(start_state)

        obs = []
        act = []
        rewards = []
        env_infos = []
        states = []
        next_obs = []

        for t in range(horizon):
            obs.append(e._get_obs())
            act.append(act_list[i][t])
            env_infos.append(e.get_env_infos())
            states.append(e.get_env_state())
            s, r, d, ifo = env.step(act[-1])
            rewards.append(r)
            next_obs.append(s)

        path = dict(observations=np.array(obs),
                    actions=np.array(act),
                    rewards=np.array(rewards),
                    env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
                    states=states,
                    next_observations=next_obs)
        paths.append(path)

    return paths
