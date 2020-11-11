from typing import List

from navec import Navec
from slovnet import NER

from purano.annotator.processors import Processor
from purano.models import Document
from purano.proto.info_pb2 import Info as InfoPb, EntitySpan as EntitySpanPb


@Processor.register("ner_slovnet")
class NerSlovnetProcessor(Processor):
    def __init__(self, model_path, vector_model_path):
        navec = Navec.load(vector_model_path)
        self.model = NER.load(model_path)
        self.model.navec(navec)

    def __call__(self,
        docs: List[Document],
        infos: List[InfoPb],
        input_fields: List[str],
        output_field: str
    ):
        for doc_num, (doc, info) in enumerate(zip(docs, infos)):
            sample = " ".join([getattr(doc, input_field) for input_field in input_fields])
            markup = self.model(sample)
            spans = []
            for s in markup.spans:
                entity_span = EntitySpanPb()
                entity_span.begin = s.start
                entity_span.end = s.stop
                entity_span.tag = EntitySpanPb.Tag.Value(s.type)
                spans.append(entity_span)
            getattr(info, output_field).extend(spans)
