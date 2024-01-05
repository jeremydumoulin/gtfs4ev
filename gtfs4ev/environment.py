# coding: utf-8

from pathlib import Path

PARENT_PATH = Path.cwd()
ROOT_PATH = PARENT_PATH / 'gtfs4ev'

TEST_PATH = ROOT_PATH / 'tests'

OUTPUT_PATH = ROOT_PATH.parents[0] / 'output'
INPUT_PATH = ROOT_PATH.parents[0] / 'input'
