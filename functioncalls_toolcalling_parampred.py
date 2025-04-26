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
from mock_annotation_tool import MockAnnotationTool

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')


class void_INCEpTION_UI:
    """empty class that indicates that the function has sideeffects in the INCEpTION UI
    otherwise void"""
    pass


class AgentfunctionsToolcallingParampred():
    def __init__(self, softwareenv: MockAnnotationTool, callback_llm, callback_getstate, testing=False) -> None:
        self.softwareenv = softwareenv
        self.callback_llm = callback_llm
        self.callback_getstate = callback_getstate
        self.testing = testing

        self.valid_functions = {
            "search_context": self.search_context,
            "search_annotations": self.search_annotations,
            "summarize_document": self.summarize_document,
            "classify_span": self.classify_span,
            "annotate": self.annotate,
            "highlight": self.highlight,
            "respond": self.respond
        }

    def search_context(self, key_phrase: str) -> str:
        """Searches for relevant context in the document text. Helps retrieving more information about a topic. This returns no information about annotations."""
        logger = logging.getLogger("functions")
        logger.debug("inside search\ncurrent state:")
        logger.debug(self.callback_getstate())

        # get relevant chunks from vector store
        contxt = self.softwareenv.get_vectorstore().similarity_search(key_phrase, filter={
            # , filter={"doc_id": self.softwareenv.current_document_id}
            "doc_id": self.softwareenv.current_document_id})
        [logger.debug(f"{i}: doc {d.metadata}\n{d.page_content}\n") for i, d in enumerate(contxt)]

        # highlight best context
        # create return string
        contxt_string = "\n\n".join([f"{i+1}: {el.page_content}" for i, el in enumerate(contxt)])
        # ask LLM, which context is the best and which segment of that context is relevant
        return_result = self.callback_llm(
            get_system_prompt_search_context(contxt_string),
            self.callback_getstate()['user_query']
        )
        try:
            # try reading the return json and extracting the most relevant text
            json_response = json.loads(return_result)
            id = int(json_response['id']) - 1
            text = json_response['text']
            if text in contxt[id].page_content:
                best_contxt = contxt[id]
                start_idx = best_contxt.metadata['start_index'] + best_contxt.page_content.index(text)
                end_idx = start_idx + len(text)
            else:
                # if text cant be found in the context, its most likely an LLM halluzination, so do fallback
                best_contxt = contxt[0]
                start_idx = best_contxt.metadata["start_index"]
                end_idx = start_idx + len(best_contxt.page_content)
        except:
            # as a fallback if json is unreadable, just highlight most similar context from vector store
            best_contxt = contxt[0]
            start_idx = best_contxt.metadata["start_index"]
            end_idx = start_idx + len(best_contxt.page_content)

        ann_pos = [(key_phrase, (start_idx, end_idx))]
        self.callback_getstate()['annotation_positions'] = ann_pos
        self.callback_getstate()['search_context'] = contxt_string
        self.highlight()

        # self.callback_getstate()['context'] = contxt_string

        # return 'I saved relevant context in the memory.', leads to the llm calling summarize
        # because it doesn't know if it has enough information to answer the users query yet
        # so returning the actual context is important for the llm to make its decision
        return contxt_string

    def search_annotations(self, search_query: str) -> str:
        """Analyze existing annotations in the document."""
        logger = logging.getLogger("functions")
        logger.debug("\ninside check_annotations\ncurrent state:")
        logger.debug(self.callback_getstate())

        # why does where not accept multiple k:v's inside the dict ?!?!?!?!??!
        annos_docs = self.softwareenv.get_vectorstore().get(where={'isAnnotation': True})

        # get relevant layer+feature
        # should this always be specific to a layer + feature?
        annos = []
        for text, metadata in zip(annos_docs['documents'], annos_docs['metadatas']):
            annos.append((text, metadata['text']))

        # TODO: cap the length of this and do multiple LLM calls
        annos_string = "\n\n".join([f"{i+1}: {text}\ncontext: {context}" for i, (context, text) in enumerate(annos)])
        logger.debug(get_system_prompt_verify_annos(annos_string))
        return_result = self.callback_llm(get_system_prompt_verify_annos(
            annos_string), search_query)
        return return_result

    def summarize_document(self, scope: str) -> str:
        """Summarizes a document. Depending of the scope of the user query, this might be the current document or all documents"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside summarize\ncurrent state:")
        logger.debug(self.callback_getstate())

        txt = ""
        if scope == "current document":
            txt = self.softwareenv.get_current_documenttext()
        elif scope == "all documents":
            txt = self.softwareenv.get_all_documenttext()
        return_result = self.callback_llm(SYSTEM_PROMPT_SUMMARIZE, txt)

        self.callback_getstate()['summarization'] = return_result

        return return_result

    def classify_span(self, scope: str, classification_query: str) -> list[tuple[str, tuple[int, int]]]:
        """Classifies spans in the document that are relevant according to the classification_query. These are then saved in memory as annotation_positions."""
        logger = logging.getLogger("functions")
        logger.debug("\ninside classify_span\ncurrent state:")
        logger.debug(self.callback_getstate())

        self.callback_getstate()["scope"] = scope

        if self.testing:
            self.callback_getstate()['annotation_positions'] = []
            return f"I classified relevant spans regarding '{classification_query}' and saved them in memory under 'annotation_positions'"

        # TODO: ask llm for better criteria query from self.callback_getstate()["user_query"]
        # get text from doc/cas
        if scope == "current document":
            documenttext = self.softwareenv.get_current_documenttext()
        elif scope == "all documents":
            documenttext = self.softwareenv.get_all_documenttext()
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
                get_system_prompt_classify(classification_query), text)

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
                    # check if the text is mentioned somewhere
                    if re.search(m.group(2), documenttext) is None:
                        # if none exist just continue
                        continue
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
        found_words = [(documenttext[s:e], (s, e)) for _, (s, e) in found_spans]
        self.callback_getstate()['annotation_positions'] = found_spans
        logger.debug(
            "Found these words in the text: %s\n--------------------------------\n", found_words)
        # stringify_spans = [f"({cat}, ({s}, {e}))" for (cat, (s, e)) in found_spans]
        return f"I classified relevant spans regarding '{classification_query}' and saved them in memory under 'annotation_positions'"

    def highlight(self) -> void_INCEpTION_UI:
        """Highlights the spans, that are saved in the memory, in the document"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside highlight\ncurrent state:")
        logger.debug(self.callback_getstate())

        # load document
        cas = self.softwareenv.get_current_document()
        # retrieve the layer from cas
        Token = cas.typesystem.get_type('highlights')
        # iterate over spans
        for _, (start, end) in self.callback_getstate()["annotation_positions"]:
            # create the token
            t = Token(begin=start, end=end)
            # set feature to classification
            cas.add(t)
        cas.to_json('temp.json')
        return 'I highlighted the relevant spans from the memory "annotation_positions".'

    def annotate(self, layer: str, feature: str, scope: str) -> void_INCEpTION_UI:
        """Creates new annotations on a given layer at a given feature. Uses the spans in the memory to do so. Should only be done if user specifically asks to create annotations."""
        logger = logging.getLogger("functions")
        logger.debug("\ninside annotate\ncurrent state:")
        logger.debug(self.callback_getstate())

        self.callback_getstate()["layer"] = layer
        self.callback_getstate()["feature"] = feature
        self.callback_getstate()["scope"] = scope

        # load document
        cas = self.softwareenv.get_current_document()

        # retrieve the layer from cas
        layertype = cas.typesystem.get_type(layer)
        # iterate over spans
        for classification, (start, end) in self.callback_getstate()["annotation_positions"]:
            # create the token
            t = layertype(begin=start, end=end)
            # set feature to classification
            t[feature] = classification
            cas.add(t)
        cas.to_json('temp.json')
        return 'I annotated the relevant spans from the memory "annotation_positions".'

    def respond(self) -> str:
        """Create a response for the user summarizing the functions/intents called and answering the user query"""
        logger = logging.getLogger("functions")
        logger.debug("\ninside respond\ncurrent state:")
        logger.debug(self.callback_getstate())

        return_result = self.callback_llm(
            get_system_prompt_respond(self.callback_getstate()), self.callback_getstate()['user_query'])

        l = logging.getLogger('output')
        l.info(f"""{return_result}

--------------------------""")
        return return_result
