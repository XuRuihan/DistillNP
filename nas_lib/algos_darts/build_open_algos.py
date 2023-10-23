import sys
from .gin_uncertainty_predictor_open_search import gin_uncertainty_predictor_search_open
from .gin_predictor_open_search import gin_predictor_search_open
from .gin_predictor_seg_open_search import gin_predictor_seg_search_open
from .narformer_open_search import narformer_search_open
from .narformer_open_search_distill import narformer_distill_search_open


def build_open_algos(agent):
    return getattr(sys.modules[__name__], agent+'_search_open')
