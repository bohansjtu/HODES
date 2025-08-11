import HODES as ho
from collections import deque
from typing import Set, Union
from HODES.automata.DFA import DFA
from HODES.automata.event import Event
from HODES.automata.NFA import NFA

EventSet = Set[Event]
Automata_t = Union[DFA, NFA]

def parallel_with_detect_flag(G: DFA) -> DFA:
    """

    Parameters
    ----------
    G : DFA
        Original DFA

    Returns
    -------
    DFA
        The composed DFA with boolean detect_flag in each state
    """
    D = ho.composition.observer(G)

    G_out = DFA()

    G_x0 = G.vs[0]
    D_x0 = D.vs[0]

    init_name = (G_x0["name"], D_x0["name"])
    detect_flag = isinstance(D_x0["name"], tuple) and len(D_x0["name"]) == 1
    G_out_vertices = [{"name": init_name, "marked": G_x0["marked"] and D_x0["marked"], "detect_flag": detect_flag}]
    G_out_names = {init_name: 0}
    G_out_edges = []

    queue = deque([(G_x0, D_x0)])

    private_G = G.events - D.events
    private_D = D.events - G.events

    while queue:
        xg, xd = queue.popleft()
        outg = {e[1]: e[0] for e in xg["out"]}
        outd = {e[1]: e[0] for e in xd["out"]}
        events_g = set(outg.keys())
        events_d = set(outd.keys())
        events_all = events_g | events_d
        events_both = events_g & events_d

        current_name = (xg["name"], xd["name"])
        src_index = G_out_names[current_name]

        for e in events_all:
            if e in events_both:
                xg_dst = G.vs[outg[e]]
                xd_dst = D.vs[outd[e]]
            elif e in private_G:
                xg_dst = G.vs[outg[e]]
                xd_dst = xd
            elif e in private_D:
                xg_dst = xg
                xd_dst = D.vs[outd[e]]
            else:
                continue

            dst_name = (xg_dst["name"], xd_dst["name"])
            if dst_name not in G_out_names:
                detect_flag = isinstance(xd_dst["name"], tuple) and len(xd_dst["name"]) == 1
                G_out_names[dst_name] = len(G_out_vertices)
                G_out_vertices.append({
                    "name": dst_name,
                    "marked": xg_dst["marked"] and xd_dst["marked"],
                    "detect_flag": detect_flag
                })
                queue.append((xg_dst, xd_dst))

            dst_index = G_out_names[dst_name]
            G_out_edges.append({"pair": (src_index, dst_index), "label": e})

    # Build the DFA
    G_out.add_vertices(
        len(G_out_vertices),
        [v["name"] for v in G_out_vertices],
        [v["marked"] for v in G_out_vertices],
    )
    G_out.add_edges(
        [e["pair"] for e in G_out_edges],
        [e["label"] for e in G_out_edges],
        check_DFA=False,
        fill_out=True,
    )
    G_out.events = G.events | D.events
    G_out.Eua.update(G.Eua | D.Eua)
    G_out.Euo.update(G.Euo | D.Euo)

    G_out.vs["detect_flag"] = [v["detect_flag"] for v in G_out_vertices]

    return G_out

def check_highorder_detect(G_aug: DFA) -> bool:
    """
    Returns False if there exists an observer state with mixed detect_flag values
    among its constituent G_aug states.

    Parameters
    ----------
    G_aug : DFA
        A DFA with 'detect_flag' attribute (boolean) on each state

    Returns
    -------
    bool
        True if every observer state is internally consistent in detect_flag (all True or all False),
        False if any observer state contains both detectable and non-detectable G_aug states.
    """
    Obs = ho.composition.observer_ua(G_aug)

    for v in Obs.vs:
        composed_states = v["name"]  # tuple of G_aug state names
        flags = []

        for state_name in composed_states:
            try:
                idx = G_aug.vs.find(name=state_name).index
                flags.append(G_aug.vs[idx]["detect_flag"])
            except ValueError:
                raise ValueError(f"State {state_name} not found in G_aug.")

        if len(set(flags)) > 1:
            # Mixed detect_flag values in this observer state
            return False

    return True  # All observer states are internally consistent

def check_highorder_detect_single_exp(G_aug: DFA) -> bool:

    Twin_ua = ho.composition.twin_ua(G_aug)

    for (x1, x2) in Twin_ua.vs["name"]:
        try:
            flag1 = G_aug.vs.find(name=x1)["detect_flag"]
            flag2 = G_aug.vs.find(name=x2)["detect_flag"]
            if flag1 != flag2:
                return False
        except:
            continue

    return True