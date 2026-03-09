"""Downloaders for individual PPI databases."""
from .biogrid import BioGRIDDownloader
from .intact import IntActDownloader
from .string_db import STRINGDownloader
from .mint import MINTDownloader
from .hippie import HIPPIEDownloader
from .reactome import ReactomeDownloader
from .signor import SIGNORDownloader
from .bioplex import BioPlexDownloader
from .innatedb import InnateDBDownloader
from .phosphositeplus import PhosphoSitePlusDownloader
from .omnipath import OmniPathDownloader
from .complex_portal import ComplexPortalDownloader
from .hint import HINTDownloader
from .corum import CORUMDownloader
from .negatome import NegatomeDownloader
from .virhostnet import VirHostNetDownloader
from .huri import HuRIDownloader
from .dip import DIPDownloader

__all__ = [
    "BioGRIDDownloader",
    "IntActDownloader",
    "STRINGDownloader",
    "MINTDownloader",
    "HIPPIEDownloader",
    "ReactomeDownloader",
    "SIGNORDownloader",
    "BioPlexDownloader",
    "InnateDBDownloader",
    "PhosphoSitePlusDownloader",
    "OmniPathDownloader",
    "ComplexPortalDownloader",
    "HINTDownloader",
    "CORUMDownloader",
    "NegatomeDownloader",
    "VirHostNetDownloader",
    "HuRIDownloader",
    "DIPDownloader",
]
