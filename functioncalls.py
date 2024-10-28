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


class Agentfunctions():
    def __init__(self, softwareenv: Software_environment, callback_llm) -> None:
        self.softwareenv = softwareenv
        self.callback_llm = callback_llm

        self.valid_functions = {
            "chat": self.chat,
            "search_context": self.search_context,
            "check_annotations": self.check_annotations,
            "summarize": self.summarize_document,
            "classify_span": self.classify_span,
            "get_scope": self.get_scope,
            "annotate": self.annotate,
            "highlight": self.highlight,
            "get_layer_and_feature": self.get_layer_and_feature,
            "respond": self.respond
        }

    def chat(self, user_query: str) -> str:
        "The user wants to chat"
        logger = logging.getLogger("functions")
        logger.debug("inside chat\tparameters:\tuser_query=%s", user_query)

    def search_context(self, criteria_query: str) -> str:
        """Search for relevent chunks in the text based on the given criteria"""
        """Takes criteria and returns top-k chunks from Vector Store (RAG)"""
        logger = logging.getLogger("functions")
        logger.debug(
            "inside search\tparameters:\tcriteria_query=%s", criteria_query)

    def check_annotations(self, layer_and_feature: tuple[str, str], user_query: str) -> str:
        """Iterate over annotations either solely annotations or with sliding context-window"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside check_annotations\nparameters:")
        logger.debug(f"{layer_and_feature=}\n{user_query=}")

    def summarize_document(self, scope: str) -> str:
        """Classifies text based on the scope"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside summarize\nparameters:")
        logger.debug(f"{scope=}\n")

        txt = ""
        if scope == "current document":
            txt = self.softwareenv.get_current_documenttext()
        return_result = self.callback_llm(SYSTEM_PROMPT_SUMMARIZE, txt)

        return return_result

    def classify_span(self, criteria_query: str, scope: str) -> list[tuple[str, tuple[int, int]]]:
        """Iterates over text determined by the scope and classifies test based on the criteria
            First tuple element is the criteria, second is the classified text"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside classify_span\nparameters:")
        logger.debug(f"{criteria_query=}\n{scope=}\n")

        # get text from doc/cas
        if scope == "current document":
            documenttext = self.softwareenv.get_current_documenttext()
        elif scope == "all documents":
            documenttext = self.softwareenv.get_current_documenttext()
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
        # maybe make llm call more robust/resilient by seeing if length pre and post embed are the same (after erasing tags again)

        # the found spans in the following format
        # [(categorization, (start, end)), ..]
        found_spans = []
        cnt_docs = 0

        for text in text_chunks[:4]:
            return_result = self.callback_llm(
                get_system_prompt_classify(criteria_query), text)

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
                found_spans.append(
                    (m.group(1), (true_start, true_start + len(m.group(2)))))
            cnt_docs += len(text)

        # safety test, applying start:end onto the initial text
        found_words = [(documenttext[s:e], s, e) for _, (s, e) in found_spans]
        logger.debug(
            "Found these words in the text: %s\n--------------------------------\n", found_words)
        return found_spans

    def highlight(self, layer_and_feature: tuple[str, str], scope: str, text_to_highlight: list[tuple[str, tuple[int, int]]]) -> void_INCEpTION_UI:
        """Highlights the given spans from the text"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside highlight\nparameters:")
        logger.debug(f"{layer_and_feature=}\n{scope=}\n{text_to_highlight=}\n")

    def annotate(self, layer_and_feature: tuple[str, str], scope: str, annotation_positions: list[tuple[str, tuple[int, int]]]) -> void_INCEpTION_UI:
        """First finds most appropriate Layer and Feature from user query
            then annotates on it
            The anno pairs consist of first the text for the feature
            and second the exact corresponding span in the text"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside annotate\nparameters:")
        logger.debug(f"{layer_and_feature=}\n{scope=}\n{
                     annotation_positions=}\n")

    def get_scope(self, user_query: str):
        """analyzes the user_query and returns the document id(s) of the relevant document"""
        logger = logging.getLogger("functions")
        logger.debug("inside get_scope\nparameters:")
        logger.debug(f"{user_query=}\n")

        return_result = self.callback_llm(SYSTEM_PROMPT_GETSCOPE, user_query)

        return return_result

    def get_layer_and_feature(self, original_user_query: str):
        logger = logging.getLogger("functions")
        logger.debug("\ninside get_feature\nparameters:")
        logger.debug(f"{original_user_query=}\n")

        # get possible layers and features
        landfs = self.softwareenv.get_layers_and_features()

        # call llm
        return_result = self.callback_llm(
            get_system_prompt_getlayer(landfs), original_user_query)

        # extract from json response
        json_response = json.loads(return_result)
        layer = json_response['layer']
        feature = json_response['feature']

        return layer, feature

    def respond(self, context: str) -> str:
        """Create a response for the user summarizing the functions/intents called 
            and the previous output
            Afterwards respond in the chat window"""
        """Potential Prompt, also get initialy user query as parameter
        Justify how well you answered the user query"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside respond\nparameters:")
        logger.debug(f"{context=}\n")
        return context
