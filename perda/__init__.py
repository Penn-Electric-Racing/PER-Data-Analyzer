# from .analyzer import analyzer
from .newanalyzer import newanalyzer

__all__ = ["create, create_new"]


# def create():
#     return analyzer()


def create_new():
    return newanalyzer()
