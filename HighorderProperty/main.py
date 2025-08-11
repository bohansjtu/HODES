import time
import random
import HODES as ho

import statistics
import csv
import gc


# Parameters
state_sizes = [5, 10, 15, 20, 30, 50, 80]
event_sizes = [3, 4]
uo_sizes = [1, 2]
ua_sizes = [1, 2]
num_trials = 100

results = []

for x in state_sizes:            # number of states
    trial_times = {"obs_double": [], "twin_obs": [], "state_pair_obs": []}
    trial_sizes = {"obs_double": [], "twin_obs": [], "state_pair_obs": []}

    for trial in range(num_trials):

        y = random.choice(event_sizes)
        z = random.choice(uo_sizes)
        w = random.choice(uo_sizes)

        g = ho.random_automata.generate(x, y, num_uo=z, num_ua=w)

        # Structure 1: obs_double
        try:
            start = time.time()
            obs = ho.composition.observer(g)
            kw_rec = ho.composition.parallel(g, obs)
            obs_double = ho.composition.observer_ua(kw_rec)
            elapsed = time.time() - start
            trial_times["obs_double"].append(elapsed)
            trial_sizes["obs_double"].append(len(obs_double.vs))
        except Exception as e:
            print(f"Exception in obs_double: {e}", flush=True)

        # Structure 2: twin_obs
        try:
            start = time.time()
            obs = ho.composition.observer(g)
            kw_rec = ho.composition.parallel(g, obs)
            twin_obs = ho.composition.twin_ua(kw_rec)
            elapsed = time.time() - start
            trial_times["twin_obs"].append(elapsed)
            trial_sizes["twin_obs"].append(len(twin_obs.vs))
        except Exception as e:
            print(f"Exception in twin_obs: {e}", flush=True)

        # Structure 3: sta_pair_obs
        try:
            start = time.time()
            state_pair = ho.composition.sta_pair_observer(g)
            elapsed = time.time() - start
            trial_times["state_pair_obs"].append(elapsed)
            trial_sizes["state_pair_obs"].append(len(state_pair.vs))
        except Exception as e:
            print(f"Exception in sta_pair_obs: {e}", flush=True)


        for var in ['g', 'obs', 'kw_rec', 'obs_double', 'twin_obs', 'state_pair']:
            if var in locals():
                del globals()[var]
                gc.collect()



    result_entry = {
        "params": (x),
        "obs_double_time_avg": statistics.mean(trial_times["obs_double"]) if trial_times["obs_double"] else None,
        "obs_double_time_max": max(trial_times["obs_double"]) if trial_times["obs_double"] else None,
        "obs_double_size_avg": statistics.mean(trial_sizes["obs_double"]) if trial_sizes["obs_double"] else None,
        "obs_double_size_max": max(trial_sizes["obs_double"]) if trial_sizes["obs_double"] else None,

        "twin_obs_time_avg": statistics.mean(trial_times["twin_obs"]) if trial_times["twin_obs"] else None,
        "twin_obs_time_max": max(trial_times["twin_obs"]) if trial_times["twin_obs"] else None,
        "twin_obs_size_avg": statistics.mean(trial_sizes["twin_obs"]) if trial_sizes["twin_obs"] else None,
        "twin_obs_size_max": max(trial_sizes["twin_obs"]) if trial_sizes["twin_obs"] else None,

        "state_pair_time_avg": statistics.mean(trial_times["state_pair_obs"]) if trial_times["state_pair_obs"] else None,
        "state_pair_time_max": max(trial_times["state_pair_obs"]) if trial_times["state_pair_obs"] else None,
        "state_pair_size_avg": statistics.mean(trial_sizes["state_pair_obs"]) if trial_sizes["state_pair_obs"] else None,
        "state_pair_size_max": max(trial_sizes["state_pair_obs"]) if trial_sizes["state_pair_obs"] else None,
    }
    results.append(result_entry)
    print(f"Finished (x={x})")



with open("structure_benchmark_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
