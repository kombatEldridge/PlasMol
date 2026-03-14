# utils/input/__init__.py
# from plasmol.utils.input.parser_new_but_old import parseInputFile
from plasmol.utils.input.params import PARAMS
from plasmol.utils.input.cli import parse_arguments

__all__ = ['PARAMS', 'parse_arguments']