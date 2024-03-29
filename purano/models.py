from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from purano.proto.info_pb2 import Info as InfoPb

Base = declarative_base()


class Document(Base):
    __tablename__ = "document"

    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, nullable=False)
    host = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    language = Column(Text)
    category = Column(Text)
    text = Column(Text)
    title = Column(Text)
    patched_text = Column(Text)
    patched_title = Column(Text)
    out_links = Column(Text, nullable=True)
    topics = Column(Text, nullable=True)
    authors = Column(Text, nullable=True)
    info = relationship("Info", uselist=False, back_populates="document")


class Info(Base):
    __tablename__ = "info"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('document.id'), unique=True)
    document = relationship("Document", back_populates="info")
    info = Column(String)

    def __init__(self, document_id: int, info: InfoPb):
        self.document_id = document_id
        self.info = info.SerializeToString()

    def get_info(self):
        info = InfoPb()
        info.ParseFromString(self.info)
        return info

    def __getitem__(self, field):
        return getattr(self.get_info(), field)
