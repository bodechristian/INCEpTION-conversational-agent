import re
import logging
import os
import math
import sys
import json

from typing import Callable

from prompts import *
from groq import Groq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from software_environment import Software_environment

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')


class void_INCEpTION_UI:
    """empty class that indicates that the function has sideeffects in the INCEpTION UI
    otherwise void"""
    pass


class AgentfunctionsToolcalling():
    def __init__(self, softwareenv: Software_environment, callback_llm, state) -> None:
        self.softwareenv = softwareenv
        self.callback_llm = callback_llm
        self.state = state

        self.valid_functions = {
            "search_context": self.search_context,
            "check_annotations": self.check_annotations,
            "summarize_document": self.summarize_document,
            "classify_span": self.classify_span,
            "get_scope": self.get_scope,
            "annotate": self.annotate,
            "highlight": self.highlight,
            "get_layer_and_feature": self.get_layer_and_feature,
            "respond": self.respond
        }

    def search_context(self) -> str:
        """Search for relevent chunks in the text based on the given criteria"""
        """Takes criteria and returns top-k chunks from Vector Store (RAG)"""
        logger = logging.getLogger("functions")
        logger.debug(
            "inside search\tparameters:\tcriteria_query=%s", self.state["user_query"])
        # TODO: ask llm for better criteria query from self.state["user_query"]
        # get relevant chunks from vector store
        contxt = self.softwareenv.get_vectorstore().similarity_search(
            self.state["user_query"])  # , filter={"doc_id": 0}
        [logger.debug(f"{i}: doc {d.metadata}\n{d.page_content}\n") for i, d in enumerate(contxt)]

        # highlight best context
        self.state['annotation_positions'] = contxt[0]
        self.get_scope()
        self.get_layer_and_feature()
        self.highlight()

        # create return string
        contxt_string = "\n\n".join([f"{i+1}: {el.page_content}" for i, el in enumerate(contxt)])

        return contxt_string

    def check_annotations(self) -> str:
        """Iterate over annotations either solely annotations or with sliding context-window"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside check_annotations\nparameters:")
        logger.debug(f"{self.state["layer"]=}\n{self.state["feature"]=}\n{self.state["user_query"]=}")

        # why does where not accept multiple k:v's inside the dict ?!?!?!?!??!
        annos_docs = self.softwareenv.get_vectorstore().get(where={'isAnnotation': True})

        # get relevant layer+feature
        # should this always be specific to a layer + feature?
        annos = []
        for text, metadata in zip(annos_docs['documents'], annos_docs['metadatas']):
            if metadata['layer'] == self.state["layer"]:
                annos.append((text, metadata['text']))

        # TODO: cap the length of this and do multiple LLM calls
        annos_string = "\n\n".join([f"{i+1}: {text}\ncontext: {context}" for i, (context, text) in enumerate(annos)])
        logger.debug(get_system_prompt_verify_annos(annos_string))
        return_result = self.callback_llm(get_system_prompt_verify_annos(annos_string), self.state["user_query"])
        return return_result

    def summarize_document(self) -> str:
        """Classifies text based on the scope"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside summarize\nparameters:")
        logger.debug(f"{self.state["scope"]=}\n")

        txt = ""
        if self.state["scope"] == "current document":
            txt = self.softwareenv.get_current_documenttext()
        return_result = self.callback_llm(SYSTEM_PROMPT_SUMMARIZE, txt)

        return return_result

    def classify_span(self) -> list[tuple[str, tuple[int, int]]]:
        """Iterates over text determined by the scope and classifies text based on the criteria
            First tuple element is the categorization, second is start and end index of the classified text"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside classify_span\nparameters:")
        logger.debug(f"{self.state["user_query"]=}\n{self.state["scope"]=}\n")
        print(f"\nin classify:")
        print(self.state)

        # TODO: ask llm for better criteria query from self.state["user_query"]
        # get text from doc/cas
        if self.state["scope"] == "current document":
            documenttext = self.softwareenv.get_current_documenttext()
        elif self.state["scope"] == "all documents":
            documenttext = self.softwareenv.get_current_documenttext()
        else:
            return 'No valid scope yet.'

        # chunk texts
        # long texts may go out of context window and make llm ignore the prompt 'only respond with embedded text'
        # long texts also make llm embelish (e.g. change 'Clinton' to 'Hillary Clinton')
        # changing the length of text and making indexes inaccurate
        # too short texts make llm add additional text (e.g. 'candidates' -> the llm answers)
        # PROBLEM: LLM loves turning \r\n\r\n into \n\n
        # or sometimes \r\n into \r\n\r\n
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, separators=["\r\n\r\n", "\n"], keep_separator="end", strip_whitespace=False)
        text_chunks = text_splitter.split_text(documenttext)

        # the found spans in the following format
        # [(categorization, (start, end)), ..]
        found_spans = []
        cnt_docs = 0

        for text in text_chunks:
            return_result = self.callback_llm(
                get_system_prompt_classify(self.state["user_query"]), text)

            # hacky fix for \r\n\r\n -> \n\n problem
            # maybe will break other documents that dont use \r\n
            return_result = re.sub(
                r'(^|[^\r])\n\n', r'\g<1>\r\n\r\n', return_result)
            return_result = re.sub(r'([^\r])\n', r'\g<1>\r\n', return_result)

            # logger.debug(f"\n{SYSTEM_PROMPT_CLASSIFY}\n")
            logger.debug("input:\n%s\n", [text])
            logger.debug(f"\noutput:\n{[return_result]}")
            logger.debug("\n--------------------------------\n")

            # extract tags
            found_tags = re.finditer(
                r'\<([a-z A-Z]+)\>([^<]*?)<\/([a-z A-Z]+)\>', return_result)

            # count of tag-characters to subtract to get correct index position in original text
            # could alternatively be done in regex with lookarounds
            cnt = 0
            # iterate over the found tags and count their index position
            for m in found_tags:
                true_start = m.start()-cnt + cnt_docs
                # add the number of tag-symbols and letters in tags
                cnt += 5 + len(m.group(1)) + len(m.group(3))
                if (documenttext[true_start:true_start + len(m.group(2))] == m.group(2)):
                    found_spans.append(
                        (m.group(1), (true_start, true_start + len(m.group(2)))))
                else:  # index is incorrect
                    # find all text mentions
                    matches = re.finditer(m.group(2), documenttext)
                    # check if match was found
                    if matches:
                        # use match that has closest index
                        best_match = sorted(
                            matches, key=lambda el: abs(el.start()-true_start))[0]
                        true_start = best_match.start()
                        found_spans.append(
                            (m.group(1), (true_start, true_start + len(m.group(2)))))
            cnt_docs += len(text)

        # safety test, applying start:end onto the initial text
        found_words = [(documenttext[s:e], s, e) for _, (s, e) in found_spans]
        self.state = found_words
        logger.debug(
            "Found these words in the text: %s\n--------------------------------\n", found_words)
        stringify_spans = [f"({cat}, ({s}, {e}))" for (cat, (s, e)) in found_spans]
        return f"[{', '.join(stringify_spans)}]"

    def highlight(self) -> void_INCEpTION_UI:
        """Highlights the given spans from the text"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside highlight\nparameters:")
        logger.debug(f"{self.state["layer"]=}\n{self.state["feature"]=}\n{self.state["scope"]=}\n{
                     self.state["annotation_positions"]=}\n")

        # load document
        cas = self.softwareenv.get_current_document()
        layer, feature = self.state["layer"], self.state["feature"]
        # retrieve the layer from cas
        Token = cas.typesystem.get_type(layer)
        # iterate over spans
        for classification, (start, end) in self.state["annotation_positions"]:
            # create the token
            t = Token(begin=start, end=end)
            # set feature to classification
            t[feature] = classification
            cas.add(t)
        cas.to_json('temp.json')
        return 'I highlighted the relevant spans'

    def annotate(self) -> void_INCEpTION_UI:
        """First finds most appropriate Layer and Feature from user query
            then annotates on it
            The anno pairs consist of first the text for the feature
            and second the exact corresponding span in the text"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside annotate\nparameters:")
        logger.debug(f"{self.state["layer"]=}\n{self.state["feature"]=}\n{self.state["scope"]=}\n{
                     self.state["annotation_positions"]=}\n")

        # load document
        cas = self.softwareenv.get_current_document()
        layer, feature = self.state["layer"], self.state["feature"]
        # retrieve the layer from cas
        layertype = cas.typesystem.get_type(layer)
        # iterate over spans
        for classification, (start, end) in self.state["annotation_positions"]:
            # create the token
            t = layertype(begin=start, end=end)
            # set feature to classification
            t[feature] = classification
            cas.add(t)
        cas.to_json('temp.json')

    def get_scope(self) -> str:
        """analyzes the user_query and returns the document id(s) of the relevant document"""
        print(f"\nin scope:")
        print(self.state)
        logger = logging.getLogger("functions")
        logger.debug("inside get_scope\nparameters:")
        logger.debug(f"{self.state['user_query']=}\n")

        return_result = self.callback_llm(SYSTEM_PROMPT_GETSCOPE, self.state['user_query'])

        self.state['scope'] = return_result
        return f"The scope of the query is {return_result}"

    def get_layer_and_feature(self) -> str:
        logger = logging.getLogger("functions")
        logger.debug("\ninside get_feature\nparameters:")
        logger.debug(f"{self.state['user_query']=}\n")

        print(f"\nin layerfeature:")
        print(self.state)
        # get possible layers and features
        landfs = self.softwareenv.get_layers_and_features()

        # call llm
        return_result = self.callback_llm(get_system_prompt_getlayer(landfs), self.state['user_query'])

        # extract from json response
        try:
            json_response = json.loads(return_result)
            layer = json_response['layer']
            feature = json_response['feature']
        except:
            # layer or feature does not exist in response json
            logger.error("Layer or feature was not correctly recognized in response json")
            # default to random layer + feature
            layer = list(landfs.keys())[0]
            feature = landfs[layer][0]

        self.state['layer'] = layer
        self.state['feature'] = feature

        return f"The layer is {layer} and its feature is {feature}"

    def respond(self, original_query: str, context: str) -> str:
        """Create a response for the user summarizing the functions/intents called 
            and the previous output
            Afterwards respond in the chat window"""
        """Potential Prompt, also get initialy user query as parameter
        Justify how well you answered the user query"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside respond\nparameters:")
        logger.debug(f"{context=}\n")

        return_result = self.callback_llm(
            get_system_prompt_respond(context), original_query)
        return return_result
