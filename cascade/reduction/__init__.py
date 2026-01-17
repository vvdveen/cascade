"""
Program reduction module for Cascade.

Minimizes bug-triggering programs to find the essential
instructions that cause the bug.
"""

from .reducer import Reducer, ReductionResult
from .tail_finder import TailFinder
from .head_finder import HeadFinder
