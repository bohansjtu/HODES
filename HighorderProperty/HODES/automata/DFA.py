import igraph as ig

from HODES import error
from HODES.automata.automata import Automata
from HODES.automata.event import Event
# from HODES.basic_operations.language_equivalence import compare_language

# TODO see below
# MUST HAVE A DEFINITION NFA TO DFA
# CHECKS IF THERE IS NONDETERMINISM
# IF NO NONDETERMINISM THEN OUTPUTS A COPY OF THE NFA
# IF THERE IS NONDETERMINISM THEN OUTPUTS THE DETERMINIZING OF NFA

# TODO how do we incorporate documentation from the superclass Automata into the one for DFA?
# TODO as we ultimately represent the DFA as a directed graph, should we view DFA as a type of NFA? This would simplify some implementations

class DFA(Automata):
    """docstring for """

    def __init__(
        self, init=None, Eua=set(), Euo=set(), E=set(), check_DFA=True, **args
    ):
        super(DFA, self).__init__(init, Eua, Euo, E)
        if isinstance(init, ig.Graph) and check_DFA:
            all_out = self.check_DFA()
            if not all(all_out):
                raise error.DeterminismError(
                    "ERROR:\nTRIED TO CREATE A DFA BUT IT IS A NFA\n State %s is nondeterministic"
                    % self._graph.vs["name"][all_out.index(False)]
                )
            elif "prob" in self._graph.es.attributes():
                raise error.InvalidAutomataTypeError(
                    "ERROR:\nTRIED TO CREATE A DFA BUT IT IS A PFA"
                )
        # if symbolic arguments
        self.symbolic = dict()
        if args:
            for key, value in args.items():
                if key == "bdd":
                    self.symbolic[key] = value
                elif key == "transitions":
                    self.symbolic[key] = value
                elif key == "uobsa":
                    self.symbolic[key] = value
                elif key == "uobs":
                    self.symbolic[key] = value
                elif key == "states":
                    self.symbolic[key] = value[1]
                    self.symbolic["states_dict"] = value[0]
                elif key == "events":
                    self.symbolic[key] = value[1]
                    self.symbolic["events_dict"] = value[0]
                else:
                    raise error.InvalidAttributeError(
                        "ERROR:\nTRIED TO CREATE SYMBOLIC DFA ARG ERROR\nARG KEYS ARE:bdd,transitions,uobsa,uobs,states,events"
                    )

        self.type = "DFA"

        # ADD SOME CONSTRAINTS ON CREATING THE OBJECT
        # LIKE NOT HAVING ATTRIBUTES PROB
        # CHECK IF IT IS DETERMINISTIC
        # AVOID MULTIPLE TESTS. IF IT IS A DFA COPY, DEFINED BASED ON OPERATIONS ON DFAS THEN NO NEED TO CHECK
        # ONLY CHECK IF init IS A FRESH IGRAPH INSTANCE

    def copy(self):
        """
        Copy from self to other, as in::

            other = self.copy()

        """
        A = DFA(self)
        return A

    def add_edge(self, source, target, label, check_DFA=True, fill_out=True, **kwargs):
        if not isinstance(label, Event):
            # convert labels from str to Event
            # label = Event(label)
            pass
        if check_DFA:
            out_events = [e[1] for e in self.vs["out"][source]]
            if label in out_events:
                # Passive check: if adding this edge would create nondeterminism,
                # do nothing. No exit (although possibly issue a warning?)
                raise error.DeterminismError(
                    "Tried to create a DFA but it is a NFA. Repeated at source {}".format(
                        source
                    )
                )

        self.events.add(label)

        self._graph.add_edge(source, target)
        self.es[self.ecount() - 1].update_attributes({"label": label})

        if fill_out:
            out = self.vs[source]["out"]
            if out is not None:
                out.append(self.Out(target, label))
            else:
                out = [self.Out(target, label)]

            self.vs[source].update_attributes({"out": out})

    def add_edges(self, pair_list, labels, check_DFA=True, fill_out=True, **kwargs):
        # TODO WE SHOULD ADD A WARNING IF IT CHECK_DFA IS DISABLE FOR UNKNOWN FUNCTIONS
        # IF THE CALLER IS PARALLEL COMP, OBSERVER, ETC, THEN NOT WARNING SHOULD BE PRINTED
        # THIS CAN BE DONE BY CHECKING THE FUNCTION CALLER

        if len(pair_list) != len(labels):
            raise IncongruencyError("Length of pairs != length of labels")

        if not pair_list:
            # no transitions provided
            return

        # labels = [Event(l) if not isinstance(l, Event) else l for l in labels]

        if check_DFA:
            modified_sources = dict()
            for i, pair in enumerate(pair_list):
                source = pair[0]
                if source not in modified_sources:
                    modified_sources[source] = []
                modified_sources[source].append(labels[i])

            for source in modified_sources.keys():
                out_events = set(el[1] for el in self.vs["out"][source])
                out_events.update(modified_sources[source])

                if len(out_events) != len(self.vs["out"][source]) + len(
                    modified_sources[source]
                ):
                    raise error.DeterminismError(
                        "Tried to create a DFA but it is a NFA. Repeated at source {}".format(
                            source
                        )
                    )

        new_labels = list(self._graph.es["label"])
        new_labels.extend(labels)
        self.events.update(labels)
        self._graph.add_edges(pair_list)
        self._graph.es["label"] = new_labels

        if kwargs:
            for key, value in kwargs.items():
                self.es[key] = value

        if fill_out:
            out_list = self.vs["out"]
            for label, pair in zip(labels, pair_list):
                out = out_list[pair[0]]
                if out is not None:
                    out.append(self.Out(pair[1], label))
                else:
                    out = [self.Out(pair[1], label)]
                out_list[pair[0]] = out
            self.vs["out"] = out_list

    def check_DFA(self):
        """
        Check if the automaton is deterministic.

        An instance of `DFA` can become nondeterministic if the `check_DFA` option in `add_edges` is disabled, or
        if the underlying graph was edited through other means.

        Returns
        -------
        True, if the automaton is deterministic
        """
        out_event = lambda v: {el[1] for el in v}
        return [len(out_event(v)) == len(v) for v in self._graph.vs["out"]]

    # def __eq__(self, other):
    #     return isinstance(other, DFA) and compare_language(self, other)
