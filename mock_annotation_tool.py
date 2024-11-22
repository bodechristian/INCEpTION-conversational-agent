from collections import defaultdict
import cassis
import os
import time
import logging

from os import getcwd, listdir
from os.path import join
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


class MockAnnotationTool():
    # id: cas_json
    documents = {}
    vector_stores = {}
    next_id = 0
    current_document_id = -1
    folderpath = join(getcwd(), "documents")

    def __init__(self):
        t1 = time.time()
        self.COHERE_API_KEY = os.environ['COHERE_API_KEY']
        self.OPEN_AI_API_KEY = os.environ['OPEN_AI_API_KEY']
        self.create_vectorstore()
        self.load_docs()
        if len(self.documents) > 0:
            # 0: political document, 1: cheetah document
            self.current_document_id = 1
        l = logging.getLogger('output')
        l.debug(f"Vector store initialized in {time.time() - t1:.2f} seconds")

    def load_docs(self):
        """loads all documents in the given documents folder"""
        for filename in listdir(self.folderpath):
            with open(join(self.folderpath, filename), 'rb') as f:
                cas = cassis.load_cas_from_json(f)

                # add 'highlights' layer if it doesn't exist yet
                if "highlights" not in [_type.name for _type in cas.typesystem.get_types()]:
                    typesystem = cas.typesystem
                    typesystem.create_type(name='highlights')
                    cas.to_json(join(self.folderpath, filename))
                self.documents[self.next_id] = cas

                # add to vector store
                self.add_cas_to_vectorstore(cas, self.next_id)

                self.next_id += 1

    def create_vectorstore(self, embeddings_model="ollama"):
        # set embeddings
        if embeddings_model == "cohere":
            embeddings = CohereEmbeddings(
                cohere_api_key=self.COHERE_API_KEY, model="embed-english-v3.0")
        elif embeddings_model == "openai":
            embeddings = OpenAIEmbeddings(api_key=self.OPEN_AI_API_KEY, model='text-embedding-3-large')
        elif embeddings_model == 'ollama':
            embeddings = OllamaEmbeddings(model="mistral:latest")
        # create initial vector store
        self.vector_store = Chroma(
            collection_name="INCEpTION",
            embedding_function=embeddings,
        )

    def add_cas_to_vectorstore(self, cas: cassis.Cas, doc_id):
        # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250, chunk_overlap=100, add_start_index=True
        )
        documents = text_splitter.create_documents([cas.sofa_string], metadatas=[
                                                   {"doc_id": doc_id, "isAnnotation": False}])

        # add annotations to vectorstore
        for layer in cas.typesystem.get_types():
            # get custom layers
            if layer.name.startswith("webanno"):
                # iterate through annotations on this layer
                for anno in cas.select(layer.name):
                    # get surrounding text as context (+/- 50 chars)
                    anno_text = cas.sofa_string[max(anno.begin-50, 0):min(anno.end+50, len(cas.sofa_string))]
                    anno_active_features = {feat.name: anno[feat.name]
                                            for feat in layer.features if anno[feat.name] is not None}
                    doc = Document(page_content=anno_text, metadata={
                        'start_index': anno.begin, 'text': anno.get_covered_text(), 'isAnnotation': True, 'doc_id': doc_id, 'layer': layer.name,
                        **anno_active_features})
                    documents.append(doc)

        self.vector_store.add_documents(documents)

    def get_documenttext_by_id(self, id: int):
        return self.documents[id].sofa_string

    def get_document_by_id(self, id: int):
        return self.documents[id]

    def get_documents(self):
        return self.documents

    def set_current_document(self, id: int):
        if id < 0 or id >= self.next_id:
            # logger.debug("set_current_document id parameter was illegal")
            return
        self.current_document_id = id

    def get_current_document(self):
        return self.documents[self.current_document_id]

    def get_current_documenttext(self):
        return self.documents[self.current_document_id].sofa_string

    def get_vectorstore(self):
        return self.vector_store

    def update_current_document(self, cas):
        self.documents[self.current_document_id] = cas

    def update_document_by_id(self, id, cas):
        if id < 0 or id >= self.next_id:
            # logger.debug("set_current_document id parameter was illegal")
            return
        self.documents[id] = cas

    def get_layers_and_features(self):
        """Returns layers and their features in a dictionary
        format: {layer1: [feature1, feature2], layer2: [feature3], ...}"""
        landfs = defaultdict(list)
        cas = self.get_current_document()
        for f in cas.select('de.tudarmstadt.ukp.clarin.webanno.api.type.FeatureDefinition'):
            if f.layer.name.startswith('webanno'):
                landfs[f.layer.name].append(f.name)

        return landfs


if __name__ == "__main__":
    a = MockAnnotationTool()
    print(a.get_layers_and_features())
