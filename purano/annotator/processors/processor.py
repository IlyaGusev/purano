from typing import List

from purano.util.registrable import Registrable
from purano.proto.info_pb2 import Info as InfoPb
from purano.models import Document


class Processor(Registrable):
    def __call__(self, docs: List[Document], infos: List[InfoPb]):
        raise NotImplementedError("Base Processor call")
