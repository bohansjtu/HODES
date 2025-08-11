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
    indicating whether any secret state has ever been visited along the path.

    The new state names are (original_state_name, has_visited_secret: bool).

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
    init_flag = G_x0["name"] in secret_states
    init_name = (G_x0["name"], init_flag)

    state_map = {init_name: 0}
    G_aug_vertices = [{"name": init_name, "marked": G_x0["marked"]}]
    G_aug_edges = []
    queue = deque([(G_x0, init_flag)])

    while queue:
        x, flag = queue.popleft()
        cur_name = (x["name"], flag)
        src_index = state_map[cur_name]

        for edge in x["out"]:
            e = edge[1]
            x_dst = G.vs[edge[0]]
            next_flag = flag or (x_dst["name"] in secret_states)
            dst_name = (x_dst["name"], next_flag)

            if dst_name not in state_map:
                state_map[dst_name] = len(G_aug_vertices)
                G_aug_vertices.append({
                    "name": dst_name,
                    "marked": x_dst["marked"]
                })
                queue.append((x_dst, next_flag))

            dst_index = state_map[dst_name]
            G_aug_edges.append({"pair": (src_index, dst_index), "label": e})

    # Add vertices and edges to the automaton
    G_aug.add_vertices(
        len(G_aug_vertices),
        [v["name"] for v in G_aug_vertices],
        [v["marked"] for v in G_aug_vertices],
    )
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

def check_intru_detect(G_aug: DFA) -> bool:

    obs_aug = augment_observer_with_secret_flag(G_aug)
    parallel = ho.composition.parallel(G_aug, obs_aug)
    Obs_ua = ho.composition.observer_ua(parallel)

    for obs_state in Obs_ua.vs["name"]:
        all_flags_true = True

        for (x, q) in obs_state:
            q_set = obs_aug.vs.find(name=q)["name"]

            for y in q_set:
                if not G_aug.vs.find(name=y)["name"][1]:
                    all_flags_true = False
                    break

        if all_flags_true:
            return False

    return True


def check_intru_detect_single_exp(G_aug: DFA) -> bool:

    obs = ho.composition.sta_pair_observer(G_aug)

    for state_set in obs.vs["name"]:
        all_yz_flags_true = True
        for triple in state_set:
            _, (y, z) = triple
            y_flag = y[1]
            z_flag = z[1]
            if not (y_flag and z_flag):
                all_yz_flags_true = False
                break

        if all_yz_flags_true:
            return False

    return True