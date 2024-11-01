from collections import defaultdict
import cassis

from os import getcwd, listdir
from os.path import join


class Software_environment():
    # id: cas_json
    documents = {}
    next_id = 0
    current_document_id = -1
    folderpath = join(getcwd(), "documents")

    def __init__(self):
        self.load_docs()
        if len(self.documents) > 0:
            self.current_document_id = 0

    def load_docs(self):
        """loads all documents in the given documents folder"""
        for filename in listdir(self.folderpath):
            with open(join(self.folderpath, filename), 'rb') as f:
                self.documents[self.next_id] = cassis.load_cas_from_json(f)
                self.next_id += 1

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
        # for l in landfs.keys():
        #     try:
        #         parentype = cas.typesystem.get_type(l)
        #     except:
        #         parent_type = cas.typesystem.create_type(name=l)
        #         for f in landfs[l]:
        #             cas.typesystem.create_feature(
        #                 domainType=parent_type, name=f, rangeType=cassis.TYPE_NAME_STRING)

        return landfs


if __name__ == "__main__":
    a = Software_environment()
    print(a.get_layers_and_features())
