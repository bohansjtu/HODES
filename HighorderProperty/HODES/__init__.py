# flake8: noqa
# from HODES.automata.automata import _Automata

from HODES import error, random_automata
from HODES.automata.DFA import DFA
from HODES.automata.event import Event
from HODES.automata.NFA import NFA
from HODES.basic_operations import composition
from HODES.file.fsm_to_bdd import read_fsm_to_bdd
from HODES.file.fsm_to_igraph import read_fsm
from HODES.file.igraph_pickle import *
from HODES.file.igraph_to_fsm import write_fsm
from HODES.visualization.plot import plot
from HODES.visualization.write_svg import write_svg

__version__ = "20.9.2"
