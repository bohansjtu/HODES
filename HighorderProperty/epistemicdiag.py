import HODES as ho
from typing import Set, Union
from HODES.automata.DFA import DFA
from HODES.automata.event import Event
from HODES.automata.NFA import NFA
import random

EventSet = Set[Event]
Automata_t = Union[DFA, NFA]


def random_state_set(G: DFA, n: int) -> set:
    """
    Randomly sample a set of `n` state names from the DFA `G`.

    Parameters
    ----------
    G : DFA
        The automaton to sample from.
    n : int
        The number of states to include in the random set.

    Returns
    -------
    Set
        A set of state names (from G.vs["name"]).
    """
    all_states = G.vs["name"]
    if n > len(all_states):
        raise ValueError(f"Requested {n} states, but G only has {len(all_states)} states.")

    return set(random.sample(all_states, n))

def augment_with_secret_flag(G: DFA, secret_states: set) -> DFA:
    """
    Returns an augmented version of G where each state is labeled with a flag
    indicating whether the current state is a secret state (i.e., x in X_S).

    The new state names are (original_state_name, is_secret: bool).

    Parameters
    ----------
    G: DFA
        Original automaton
    secret_states: set
        Set of state names in G considered as secret

    Returns
    -------
    DFA
        Augmented DFA
    """
    from collections import deque

    G_aug = DFA()
    G_x0 = G.vs[0]
    is_secret = G_x0["name"] in secret_states
    init_name = (G_x0["name"], is_secret)

    state_map = {init_name: 0}
    G_aug_vertices = [{
        "name": init_name,
        "marked": G_x0["marked"],
        "flag": is_secret
    }]
    G_aug_edges = []
    queue = deque([G_x0])

    while queue:
        x = queue.popleft()
        for edge in x["out"]:
            e = edge[1]
            x_dst = G.vs[edge[0]]
            dst_is_secret = x_dst["name"] in secret_states
            dst_name = (x_dst["name"], dst_is_secret)

            if dst_name not in state_map:
                state_map[dst_name] = len(G_aug_vertices)
                G_aug_vertices.append({
                    "name": dst_name,
                    "marked": x_dst["marked"],
                    "flag": dst_is_secret
                })
                queue.append(x_dst)

            src_index = state_map[(x["name"], x["name"] in secret_states)]
            dst_index = state_map[dst_name]
            G_aug_edges.append({"pair": (src_index, dst_index), "label": e})

    # Add vertices and edges to the automaton
    G_aug.add_vertices(
        len(G_aug_vertices),
        [v["name"] for v in G_aug_vertices],
        [v["marked"] for v in G_aug_vertices],
    )
    for i, v in enumerate(G_aug_vertices):
        G_aug.vs[i]["flag"] = v["flag"]

    G_aug.add_edges(
        [e["pair"] for e in G_aug_edges],
        [e["label"] for e in G_aug_edges],
        check_DFA=False,
        fill_out=True,
    )

    # Copy event sets
    G_aug.events = G.events.copy()
    G_aug.Eua = G.Eua.copy()
    G_aug.Euo = G.Euo.copy()

    return G_aug

def augment_observer_with_secret_flag(G_aug: DFA) -> DFA:
    """
    Given an augmented DFA `G_aug` whose state name is a tuple (original_name, flag),
    compute the observer and label each observer state with a boolean flag:
    - True if all underlying G_aug states have flag=True
    - False otherwise

    The flag is stored in `Obs.vs["kw_flag"]`.

    Parameters
    ----------
    G_aug : DFA
        Augmented system with state names of the form (name, flag)

    Returns
    -------
    DFA
        Observer automaton with an additional "flag" attribute on each state
    """
    Obs = ho.composition.observer(G_aug)

    def extract_flag(state_name):
        return state_name[1]  # from (original_name, flag)

    flags = []
    for obs_state in Obs.vs["name"]:
        flags.append(all(extract_flag(name) for name in obs_state))

    Obs.vs["kw_flag"] = flags
    return Obs

def check_epis_diag(G_aug: DFA) -> bool:

    obs_aug = augment_observer_with_secret_flag(G_aug)
    parallel = ho.composition.parallel(G_aug, obs_aug)

    for v in parallel.vs:
        _, q = v["name"]
        v["kw_flag"] = obs_aug.vs.find(name=q)["kw_flag"]

    Obs_ua = ho.composition.observer_ua(parallel)

    for obs_state in Obs_ua.vs["name"]:
        flags = set()

        for (x, q) in obs_state:
            try:
                flag = parallel.vs.find(name=(x, q))["kw_flag"]
                flags.add(flag)
                if len(flags) > 1:
                    return False
            except:
                continue

    return True

def check_epis_diag_single_exp(G_aug: DFA) -> bool:

    obs_aug = augment_observer_with_secret_flag(G_aug)
    parallel = ho.composition.parallel(G_aug, obs_aug)

    for v in parallel.vs:
        _, q = v["name"]
        v["kw_flag"] = obs_aug.vs.find(name=q)["kw_flag"]

    Twin_ua = ho.composition.twin_ua(parallel)

    for (x1, x2) in Twin_ua.vs["name"]:
        try:
            flag1 = parallel.vs.find(name=x1)["kw_flag"]
            flag2 = parallel.vs.find(name=x2)["kw_flag"]
            if flag1 != flag2:
                return False
        except:
            continue

    return True