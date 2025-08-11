
from collections import deque
from typing import Any, Dict, List,  Set,  Union

from itertools import product
from collections import deque
from typing import Tuple, Set


from HODES.automata.DFA import DFA
from HODES.automata.event import Event
from HODES.automata.NFA import NFA
from HODES.error import MissingAttributeError
from collections import defaultdict

EventSet = Set[Event]
Automata_t = Union[DFA, NFA]
SHOW_PROGRESS = False




def product(*automata: DFA) -> DFA:
    """
    Computes the synchronous product composition of 2 (or more) Automata in a BFS manner,
    Returns the resulting composition as a new automatan.

    Parameters
    ----------
    automata : list[DFA]

    Returns
    -------
    DFA
        The product of the automata
    """
    if len(automata) < 2:
        raise MissingAttributeError(
            "Product composition needs more than one automaton."
        )

    if any(g.vcount() == 0 for g in automata):
        return DFA()

    G1 = automata[0]
    input_list = automata[1:]

    for G2 in input_list:
        G_out = DFA()
        G1_x0 = G1.vs[0]
        G2_x0 = G2.vs[0]
        G_out_vertices = [
            {
                "name": (G1_x0["name"], G2_x0["name"]),
                "marked": G1_x0["marked"] and G2_x0["marked"],
            }
        ]
        G_out_names = {G_out_vertices[0]["name"]: 0}
        G_out_edges = []  # type: List[Dict[str, Any]]

        queue = deque([(G1_x0, G2_x0)])

        while len(queue) > 0:
            x1, x2 = queue.popleft()
            active_x1 = {e[1]: e[0] for e in x1["out"]}
            active_x2 = {e[1]: e[0] for e in x2["out"]}
            active_both = set(active_x1.keys()) & set(active_x2.keys())
            cur_name = (x1["name"], x2["name"])
            src_index = G_out_names[cur_name]

            for e in active_both:
                x1_dst = G1.vs[active_x1[e]]
                x2_dst = G2.vs[active_x2[e]]
                dst_name = (x1_dst["name"], x2_dst["name"])
                dst_index = G_out_names.get(dst_name)

                if dst_index is None:
                    G_out_vertices.append(
                        {
                            "name": dst_name,
                            "marked": x1_dst["marked"] and x2_dst["marked"],
                        }
                    )
                    dst_index = len(G_out_vertices) - 1
                    G_out_names[dst_name] = dst_index
                    queue.append((x1_dst, x2_dst))

                G_out_edges.append({"pair": (src_index, dst_index), "label": e})

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
        G_out.events = G1.events | G2.events
        G_out.Eua.update(G1.Eua | G2.Eua)
        G_out.Euo.update(G1.Euo | G2.Euo)

        G1 = G_out

    return G_out



def parallel(*automata: DFA) -> DFA:
    """
    Computes the parallel composition of 2 (or more) Automata in a BFS manner, and returns the resulting composition as a new Automata.

    Parameters
    ----------
    automata : list[DFA]

    Returns
    -------
    DFA
        The parallel composition
    """
    if len(automata) < 2:
        raise MissingAttributeError("More than one automaton are needed.")

    G1 = automata[0]
    input_list = automata[1:]

    if any(i.vcount() == 0 for i in automata):
        # if any inputs are empty, return empty automata
        return DFA()

    for G2 in input_list:
        G_out = DFA()

        G1_x0 = G1.vs[0]
        G2_x0 = G2.vs[0]
        G_out_vertices = [
            {
                "name": (G1_x0["name"], G2_x0["name"]),
                "marked": G1_x0["marked"] and G2_x0["marked"],
            }
        ]
        G_out_names = {G_out_vertices[0]["name"]: 0}
        G_out_edges = []  # type: List[Dict[str, Any]]

        queue = deque([(G1_x0, G2_x0)])

        private_G1 = G1.events - G2.events
        private_G2 = G2.events - G1.events

        while len(queue) > 0:
            x1, x2 = queue.popleft()
            active_x1 = {e[1]: e[0] for e in x1["out"]}
            active_x2 = {e[1]: e[0] for e in x2["out"]}
            active_both = set(active_x1.keys()) & set(active_x2.keys())
            cur_name = (x1["name"], x2["name"])
            src_index = G_out_names[cur_name]

            for e in set(active_x1.keys()) | set(active_x2.keys()):
                if e in active_both:
                    x1_dst = G1.vs[active_x1[e]]
                    x2_dst = G2.vs[active_x2[e]]
                elif e in private_G1:
                    x1_dst = G1.vs[active_x1[e]]
                    x2_dst = x2
                elif e in private_G2:
                    x1_dst = x1
                    x2_dst = G2.vs[active_x2[e]]
                else:
                    continue

                dst_name = (x1_dst["name"], x2_dst["name"])
                dst_index = G_out_names.get(dst_name)

                if dst_index is None:
                    G_out_vertices.append(
                        {
                            "name": dst_name,
                            "marked": x1_dst["marked"] and x2_dst["marked"],
                        }
                    )
                    dst_index = len(G_out_vertices) - 1
                    G_out_names[dst_name] = dst_index
                    queue.append((x1_dst, x2_dst))

                G_out_edges.append({"pair": (src_index, dst_index), "label": e})

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
        G_out.events = G1.events | G2.events
        G_out.Eua.update(G1.Eua | G2.Eua)
        G_out.Euo.update(G1.Euo | G2.Euo)

        G1 = G_out

    return G_out


def observer(G: Automata_t) -> Automata_t:
    """
    Compute the observer automata of the input `G` with respect to its unobservable events
    `G` should be a `DFA`, `NFA` or `PFA`

    Parameters
    ----------
    G : NFA or DFA or PFA
        The automaton

    Returns
    -------
    DFA
        The observer automaton
    """
    observer = DFA()
    if not G.vcount() or G is None:
        return observer

    vertice_names = list()  # list of vertex names for igraph construction
    vertice_number = dict()  # dictionary vertex_names -> vertex_id
    outgoing_list = list()  # list of outgoing lists for each vertex
    marked_list = list()  # list with vertices marking
    transition_list = list()  # list of transitions for igraph construction
    transition_label = list()  # list os transitions label for igraph construction

    # BFS queue that holds states that must be visited
    queue = list()

    # index tracks the current number of vertices in the graph
    index = 0

    if isinstance(G, NFA):
        init_states = frozenset(v.index for v in G.vs if v["init"])
    else:
        init_states = frozenset({0})

    # Makes Euo hashable for URo dict key:
    Euo = frozenset(G.Euo)

    # Find URo from initial state(s):
    v0 = G.URo.from_set(init_states, Euo, freeze_result=True)

    name_v0 = frozenset([G.vs["name"][v] for v in v0])
    marking = any([G.vs["marked"][v] for v in v0])
    vertice_names.insert(index, name_v0)
    vertice_number[v0] = index
    marked_list.insert(index, marking)

    index = index + 1
    queue.append(v0)
    while queue:
        v = queue.pop(0)

        # finding observable adjacent from v
        adj_states = dict()
        for vert in v:
            for target, event in G.vs["out"][vert]:
                if event in adj_states and event not in G.Euo:
                    adj_states[event].add(target)
                elif event not in adj_states and event not in G.Euo:
                    s = set()
                    s.add(target)
                    adj_states[event] = s

        # print(adj_states)
        outgoing_v1v2 = list()
        for ev in adj_states.keys():
            next_state = frozenset(adj_states[ev])

            next_state = G.URo.from_set(next_state, Euo, freeze_result=True)
            # updating lists for igraph construction
            if next_state in vertice_number.keys():
                transition_list.append((vertice_number[v], vertice_number[next_state]))
                transition_label.append(ev)
            else:
                name_next_state = frozenset([G.vs["name"][v] for v in next_state])
                transition_list.append((vertice_number[v], index))
                transition_label.append(ev)
                vertice_number[next_state] = index
                marking = any([G.vs["marked"][v] for v in next_state])
                marked_list.insert(index, marking)
                vertice_names.insert(index, name_next_state)
                queue.append(next_state)
                index = index + 1
            outgoing_v1v2.append(observer.Out(vertice_number[next_state], ev))
        outgoing_list.insert(vertice_number[v], outgoing_v1v2)

    # constructing DFA: igraph and events sets
    observer.add_vertices(index, vertice_names)
    observer.events = G.events - G.Euo
    observer.Eua.update(G.Eua - G.Euo)
    observer.Euo.clear()
    observer.vs["marked"] = marked_list

    observer.vs[0]["init"] = True
    observer.add_edges(
        transition_list, transition_label, check_DFA=False, fill_out=False
    )
    observer.vs["out"] = outgoing_list

    observer.vs["name"] = [tuple(v) for v in observer.vs["name"]]

    return observer

def observer_ua(G: Automata_t) -> Automata_t:
    """
    Compute the observer automata of the input `G` with respect to its unobservable events
    `G` should be a `DFA`, `NFA` or `PFA`

    Parameters
    ----------
    G : NFA or DFA or PFA
        The automaton

    Returns
    -------
    DFA
        The observer automaton
    """
    observer = DFA()
    if not G.vcount() or G is None:
        return observer

    vertice_names = list()  # list of vertex names for igraph construction
    vertice_number = dict()  # dictionary vertex_names -> vertex_id
    outgoing_list = list()  # list of outgoing lists for each vertex
    marked_list = list()  # list with vertices marking
    transition_list = list()  # list of transitions for igraph construction
    transition_label = list()  # list os transitions label for igraph construction

    # BFS queue that holds states that must be visited
    queue = list()

    # index tracks the current number of vertices in the graph
    index = 0

    if isinstance(G, NFA):
        init_states = frozenset(v.index for v in G.vs if v["init"])
    else:
        init_states = frozenset({0})

    # Makes Euo hashable for URo dict key:
    Eua = frozenset(G.Eua)

    # Find URa from initial state(s):
    v0 = G.URa.from_set(init_states, Eua, freeze_result=True)

    name_v0 = frozenset([G.vs["name"][v] for v in v0])
    marking = any([G.vs["marked"][v] for v in v0])
    vertice_names.insert(index, name_v0)
    vertice_number[v0] = index
    marked_list.insert(index, marking)

    index = index + 1
    queue.append(v0)
    while queue:
        v = queue.pop(0)

        # finding observable adjacent from v
        adj_states = dict()
        for vert in v:
            for target, event in G.vs["out"][vert]:
                if event in adj_states and event not in G.Eua:
                    adj_states[event].add(target)
                elif event not in adj_states and event not in G.Eua:
                    s = set()
                    s.add(target)
                    adj_states[event] = s

        # print(adj_states)
        outgoing_v1v2 = list()
        for ev in adj_states.keys():
            next_state = frozenset(adj_states[ev])

            next_state = G.URa.from_set(next_state, Eua, freeze_result=True)
            # updating lists for igraph construction
            if next_state in vertice_number.keys():
                transition_list.append((vertice_number[v], vertice_number[next_state]))
                transition_label.append(ev)
            else:
                name_next_state = frozenset([G.vs["name"][v] for v in next_state])
                transition_list.append((vertice_number[v], index))
                transition_label.append(ev)
                vertice_number[next_state] = index
                marking = any([G.vs["marked"][v] for v in next_state])
                marked_list.insert(index, marking)
                vertice_names.insert(index, name_next_state)
                queue.append(next_state)
                index = index + 1
            outgoing_v1v2.append(observer.Out(vertice_number[next_state], ev))
        outgoing_list.insert(vertice_number[v], outgoing_v1v2)

    # constructing DFA: igraph and events sets
    observer.add_vertices(index, vertice_names)
    observer.events = G.events - G.Eua
    observer.Euo.update(G.Euo - G.Eua)
    observer.Eua.clear()
    observer.vs["marked"] = marked_list

    observer.vs[0]["init"] = True
    observer.add_edges(
        transition_list, transition_label, check_DFA=False, fill_out=False
    )
    observer.vs["out"] = outgoing_list

    observer.vs["name"] = [tuple(v) for v in observer.vs["name"]]

    return observer

def twin(G: Automata_t) -> Automata_t:
    """
    Computes the verifier automata of the input G based on Euo
    """

    unobservable = list(G.Euo)
    Ver = NFA()


    GN_x0 = (G.vs[0])
    Gf_x0 = (G.vs[0])
    Ver_vertices = [
        {
            "name": (GN_x0["name"], Gf_x0["name"]),
            "marked": GN_x0["marked"] and Gf_x0["marked"],
        }
    ]
    Ver_names = {Ver_vertices[0]["name"]: 0}
    Ver_edges = []
    queue = deque([(GN_x0, Gf_x0)])
    while len(queue) > 0:
        x1, x2 = queue.popleft()
        active_x1 = {e[1]: e[0] for e in x1["out"]} #GN
        active_x2 = {e[1]: e[0] for e in x2["out"]} #Gf
        active_both = set(active_x1.keys()) & set(active_x2.keys())
        cur_name = (x1["name"], x2["name"])
        src_index = Ver_names[cur_name]
        for e in set(active_x1.keys()) | set(active_x2.keys()):
            x1_dst = list()
            x2_dst = list()
            evt = list()
            if e not in unobservable and e in active_both:
                x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                x2_dst.append(G.vs[active_x2[e]]) # f(x2,e)
                evt.append(Event("("+ e.label+","+ e.label+")"))
            elif e in unobservable:
                if e in active_both:
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(x2)
                    evt.append(Event("("+e.label+",eps)"))
                    x1_dst.append(x1)
                    x2_dst.append(G.vs[active_x2[e]]) # f(x2,e)
                    evt.append(Event("(eps," +e.label+")"))
                elif e in active_x1:
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(x2)
                    evt.append(Event("("+e.label+",eps)"))
                elif e in active_x2:
                    x1_dst.append(x1)
                    x2_dst.append(G.vs[active_x2[e]]) # f(x2,e)
                    evt.append(Event("(eps," +e.label+")"))

            else:
                continue

            for i in range(0,len(x1_dst)):

                dst_name = (x1_dst[i]["name"], x2_dst[i]["name"])
                dst_index = Ver_names.get(dst_name)

                if dst_index is None:
                    Ver_vertices.append(
                        {
                            "name": dst_name,
                            "marked": x1_dst[i]["marked"] and x2_dst[i]["marked"],
                        }
                    )
                    dst_index = len(Ver_vertices)-1
                    Ver_names[dst_name] = dst_index
                    queue.append((x1_dst[i],x2_dst[i]))

                Ver_edges.append({"pair": (src_index, dst_index), "label": evt[i]})

    Ver.add_vertices(
        len(Ver_vertices),
        [v["name"] for v in Ver_vertices],
        [v["marked"] for v in Ver_vertices],
    )
    Ver.add_edges(
        [e["pair"] for e in Ver_edges], [e["label"] for e in Ver_edges]
    )
    Ver.events = G.events
    Ver.Eua.update(G.Eua)
    Ver.Euo.update(G.Euo)

    return Ver

def twin_ua(G: Automata_t) -> Automata_t:
    """
    Computes the verifier automata of the input G based on Euo
    """

    unobservable = list(G.Eua)
    Ver = NFA()


    GN_x0 = (G.vs[0])
    Gf_x0 = (G.vs[0])
    Ver_vertices = [
        {
            "name": (GN_x0["name"], Gf_x0["name"]),
            "marked": GN_x0["marked"] and Gf_x0["marked"],
        }
    ]
    Ver_names = {Ver_vertices[0]["name"]: 0}
    Ver_edges = []
    queue = deque([(GN_x0, Gf_x0)])
    while len(queue) > 0:
        x1, x2 = queue.popleft()
        active_x1 = {e[1]: e[0] for e in x1["out"]} #GN
        active_x2 = {e[1]: e[0] for e in x2["out"]} #Gf
        active_both = set(active_x1.keys()) & set(active_x2.keys())
        cur_name = (x1["name"], x2["name"])
        src_index = Ver_names[cur_name]
        for e in set(active_x1.keys()) | set(active_x2.keys()):
            marked = False
            x1_dst = list()
            x2_dst = list()
            evt = list()
            if e not in unobservable and e in active_both:
                x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                x2_dst.append(G.vs[active_x2[e]]) # f(x2,e)
                evt.append(Event("("+ e.label+","+ e.label+")"))
            elif e in unobservable:
                if e in active_both:
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(x2)
                    evt.append(Event("("+e.label+",eps)"))
                    x1_dst.append(x1)
                    x2_dst.append(G.vs[active_x2[e]]) # f(x2,e)
                    evt.append(Event("(eps," +e.label+")"))
                elif e in active_x1:
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(x2)
                    evt.append(Event("("+e.label+",eps)"))
                elif e in active_x2:
                    x1_dst.append(x1)
                    x2_dst.append(G.vs[active_x2[e]]) # f(x2,e)
                    evt.append(Event("(eps," +e.label+")"))

            else:
                continue

            for i in range(0,len(x1_dst)):

                dst_name = (x1_dst[i]["name"], x2_dst[i]["name"])
                dst_index = Ver_names.get(dst_name)

                if dst_index is None:
                    Ver_vertices.append(
                        {
                            "name": dst_name,
                            "marked": x1_dst[i]["marked"] and x2_dst[i]["marked"],
                        }
                    )
                    dst_index = len(Ver_vertices)-1
                    Ver_names[dst_name] = dst_index
                    queue.append((x1_dst[i],x2_dst[i]))

                Ver_edges.append({"pair": (src_index, dst_index), "label": evt[i]})

    Ver.add_vertices(
        len(Ver_vertices),
        [v["name"] for v in Ver_vertices],
        [v["marked"] for v in Ver_vertices],
    )
    Ver.add_edges(
        [e["pair"] for e in Ver_edges], [e["label"] for e in Ver_edges]
    )
    Ver.events = G.events
    Ver.Eua.update(G.Eua)
    Ver.Euo.update(G.Euo)

    return Ver

def sta_pair_observer(G: Automata_t) -> Automata_t:

    pair_observer = DFA()
    if not G.vcount() or G is None:
        return pair_observer

    unobservable = list(G.Euo)

    Ver = NFA()

    G1_x0 = (G.vs[0])
    G2_x0 = (G.vs[0])
    G3_x0 = (G.vs[0])
    Ver_vertices = [
        {
            "name": (G1_x0["name"], (G2_x0["name"], G3_x0["name"])),
            "marked": G1_x0["marked"] and G2_x0["marked"] and G3_x0["marked"],
        }
    ]
    Ver_names = {Ver_vertices[0]["name"]: 0}
    Ver_edges = []
    verqueue = deque([(G1_x0, G2_x0, G3_x0)])
    while len(verqueue) > 0:
        x1, x2, x3 = verqueue.popleft()
        active_x1 = {e[1]: e[0] for e in x1["out"]} #GN
        active_x2 = {e[1]: e[0] for e in x2["out"]} #Gf
        active_x3 = {e[1]: e[0] for e in x3["out"]}
        active_all = set(active_x1.keys()) & set(active_x2.keys()) & set(active_x3.keys())
        active_12 = set(active_x1.keys()) & set(active_x2.keys())
        active_13 = set(active_x1.keys()) & set(active_x3.keys())
        active_23 = set(active_x2.keys()) & set(active_x3.keys())
        cur_name = (x1["name"], (x2["name"], x3["name"]))
        src_index = Ver_names[cur_name]
        for e in set(active_x1.keys()) | set(active_x2.keys()) | set(active_x3.keys()):
            x1_dst = list()
            x2_dst = list()
            x3_dst = list()
            evt = list()
            if e not in unobservable and e in active_all:
                x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                x2_dst.append(G.vs[active_x2[e]]) # f(x2,e)
                x3_dst.append(G.vs[active_x3[e]]) # f(x2,e)
                evt.append(Event(e.label))
            elif e in unobservable:
                if e in active_all:
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(x2)
                    x3_dst.append(x3)
                    evt.append(Event(e.label))
                    x1_dst.append(x1)
                    x2_dst.append(G.vs[active_x2[e]])
                    x3_dst.append(x3)
                    evt.append(Event("eps"))
                    x1_dst.append(x1)
                    x2_dst.append(x2)
                    x3_dst.append(G.vs[active_x3[e]])
                    evt.append(Event("eps"))
                    x1_dst.append(x1)
                    x2_dst.append(G.vs[active_x2[e]])
                    x3_dst.append(G.vs[active_x3[e]])
                    evt.append(Event("eps"))
                    x1_dst.append(G.vs[active_x1[e]])  # fN(x1,e)
                    x2_dst.append(G.vs[active_x2[e]])
                    x3_dst.append(x3)
                    evt.append(Event(e.label))
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(x2)
                    x3_dst.append(G.vs[active_x3[e]])
                    evt.append(Event(e.label))
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(G.vs[active_x2[e]])
                    x3_dst.append(G.vs[active_x3[e]])
                    evt.append(Event(e.label))
                elif e in active_12:
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(x2)
                    x3_dst.append(x3)
                    evt.append(Event(e.label))
                    x1_dst.append(x1)
                    x2_dst.append(G.vs[active_x2[e]])
                    x3_dst.append(x3)
                    evt.append(Event("eps"))
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(G.vs[active_x2[e]])
                    x3_dst.append(x3)
                    evt.append(Event(e.label))
                elif e in active_13:
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(x2)
                    x3_dst.append(x3)
                    evt.append(Event(e.label))
                    x1_dst.append(x1)
                    x2_dst.append(x2)
                    x3_dst.append(G.vs[active_x3[e]])
                    evt.append(Event("eps"))
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(x2)
                    x3_dst.append(G.vs[active_x3[e]])
                    evt.append(Event(e.label))
                elif e in active_23:
                    x1_dst.append(x1) # fN(x1,e)
                    x2_dst.append(G.vs[active_x2[e]])
                    x3_dst.append(x3)
                    evt.append(Event("eps"))
                    x1_dst.append(x1)
                    x2_dst.append(x2)
                    x3_dst.append(G.vs[active_x3[e]])
                    evt.append(Event("eps"))
                    x1_dst.append(x1) # fN(x1,e)
                    x2_dst.append(G.vs[active_x2[e]])
                    x3_dst.append(G.vs[active_x3[e]])
                    evt.append(Event("eps"))
                elif e in active_x1:
                    x1_dst.append(G.vs[active_x1[e]]) # fN(x1,e)
                    x2_dst.append(x2)
                    x3_dst.append(x3)
                    evt.append(Event(e.label))
                elif e in active_x2:
                    x1_dst.append(x1)
                    x2_dst.append(G.vs[active_x2[e]])
                    x3_dst.append(x3)# f(x2,e)
                    evt.append(Event("eps"))
                elif e in active_x3:
                    x1_dst.append(x1)
                    x2_dst.append(x2)
                    x3_dst.append(G.vs[active_x3[e]])# f(x2,e)
                    evt.append(Event("eps"))

            else:
                continue

            for i in range(0,len(x1_dst)):

                dst_name = (x1_dst[i]["name"], (x2_dst[i]["name"], x3_dst[i]["name"]))
                dst_index = Ver_names.get(dst_name)

                if dst_index is None:
                    Ver_vertices.append(
                        {
                            "name": dst_name,
                            "marked": x1_dst[i]["marked"] and x2_dst[i]["marked"] and x3_dst[i]["marked"],
                        }
                    )
                    dst_index = len(Ver_vertices)-1
                    Ver_names[dst_name] = dst_index
                    verqueue.append((x1_dst[i],x2_dst[i],x3_dst[i]))

                Ver_edges.append({"pair": (src_index, dst_index), "label": evt[i]})

    Ver.add_vertices(
        len(Ver_vertices),
        [v["name"] for v in Ver_vertices],
        [v["marked"] for v in Ver_vertices],
    )
    Ver.vs[0]["init"] = True
    Ver.add_edges(
        [e["pair"] for e in Ver_edges], [e["label"] for e in Ver_edges]
    )
    Ver.events = G.events
    Ver.Eua.update(G.Eua)
    Ver.Euo.update(G.Euo)
    Ver.events.add(Event("eps"))
    Ver.Eua.add(Event("eps"))
    Ver.Euo.add(Event("eps"))

    pair_observer = observer_ua(Ver)

    return pair_observer


