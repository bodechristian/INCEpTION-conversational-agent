import re
import json
import logging
import os
from pydantic import BaseModel
from typing import List

from prompts import *
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mock_annotation_tool import MockAnnotationTool
import utils

load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']


class AgentfunctionsClassify:
    def __init__(self, softwareenv: MockAnnotationTool, callback_llm, callback_llm_with_format, callback_getstate, is_ollama) -> None:
        self.softwareenv = softwareenv
        self.callback_llm = callback_llm
        self.callback_llm_with_format = callback_llm_with_format
        self.callback_getstate = callback_getstate
        self.is_ollama = is_ollama

        self.valid_functions = {
            "classify_span1": self.classify_span1,
            "classify_span_no_icl": self.classify_span_no_icl,
            "classify_span_exactwordcheck2": self.classify_span_exactwordcheck2,
        }

    def classify_span1(self, criteria_query: str, scope: str, chunk_size=1000) -> list[tuple[str, tuple[int, int]]]:
        """Classifies spans in the given document-scope that are relevant according to the classification_query. 
        tuple element is the categorization, second is start and end index of the classified text.
        Returns the annotation positions.
        """
        logger = logging.getLogger("functions")
        logger.debug("\ninside classify_span1\nparameters:")
        logger.debug(f"{criteria_query=}\n{scope=}\n")

        # get text from doc/cas
        if scope == "current document":
            documenttext = self.softwareenv.get_current_documenttext()
        else:  # scope == "all documents":
            documenttext = self.softwareenv.get_current_documenttext()
        # chunk texts
        # long texts may go out of context window and make llm ignore the prompt 'only respond with embedded text'
        # long texts also make llm embelish (e.g. change 'Clinton' to 'Hillary Clinton')
        # changing the length of text and making indexes inaccurate
        # too short texts make llm add additional text (e.g. 'candidates' -> the llm answers)
        # PROBLEM: LLM loves turning \r\n\r\n into \n\n
        # or sometimes \r\n into \r\n\r\n
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0, separators=["\r\n\r\n", "\n"], keep_separator="end", strip_whitespace=False)
        text_chunks = text_splitter.split_text(documenttext)

        # the found spans in the following format
        # [(categorization, (start, end)), ..]
        found_spans = []
        cnt_docs = 0

        for text in text_chunks:
            return_result = self.callback_llm(get_system_prompt_classify(criteria_query), text)

            if isinstance(return_result, utils.ParsingException):
                print(return_result)
                continue

            return_result = utils.parse_deepseek_response(return_result)
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
            found_tags = re.finditer(r'\<([a-z A-Z]+)\>([^<]*?)<\/([a-z A-Z]+)\>', return_result)

            # count of tag-characters to subtract to get correct index position in original text
            # could alternatively be done in regex with lookarounds
            cnt = 0
            # iterate over the found tags and count their index position
            for m in found_tags:
                true_start = m.start()-cnt + cnt_docs
                # add the number of tag-symbols and letters in tags
                cnt += 5 + len(m.group(1)) + len(m.group(3))
                if (documenttext[true_start:true_start + len(m.group(2))] == m.group(2)):
                    found_spans.append((m.group(1), (true_start, true_start + len(m.group(2)))))
                else:  # index is incorrect
                    # check if the text is mentioned somewhere
                    if re.search(re.escape(m.group(2)), documenttext) is None:
                        # if none exist just continue
                        continue
                    # find all text mentions
                    matches = re.finditer(re.escape(m.group(2)), documenttext)
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
        self.callback_getstate()['annotation_positions'] = found_words
        logger.debug(
            "Found these words in the text: %s\n--------------------------------\n", found_words)
        return found_spans

    def classify_span_no_icl(self, criteria_query: str, scope: str, chunk_size=1000) -> list[tuple[str, tuple[int, int]]]:
        """Classifies spans in the given document-scope that are relevant according to the classification_query. 
        tuple element is the categorization, second is start and end index of the classified text.
        Returns the annotation positions.
        """
        logger = logging.getLogger("functions")
        logger.debug("\ninside classify_span1\nparameters:")
        logger.debug(f"{criteria_query=}\n{scope=}\n")

        # get text from doc/cas
        if scope == "current document":
            documenttext = self.softwareenv.get_current_documenttext()
        else:  # scope == "all documents":
            documenttext = self.softwareenv.get_current_documenttext()
        # chunk texts
        # long texts may go out of context window and make llm ignore the prompt 'only respond with embedded text'
        # long texts also make llm embelish (e.g. change 'Clinton' to 'Hillary Clinton')
        # changing the length of text and making indexes inaccurate
        # too short texts make llm add additional text (e.g. 'candidates' -> the llm answers)
        # PROBLEM: LLM loves turning \r\n\r\n into \n\n
        # or sometimes \r\n into \r\n\r\n
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0, separators=["\r\n\r\n", "\n"], keep_separator="end", strip_whitespace=False)
        text_chunks = text_splitter.split_text(documenttext)

        # the found spans in the following format
        # [(categorization, (start, end)), ..]
        found_spans = []
        cnt_docs = 0

        for text in text_chunks:
            return_result = self.callback_llm(get_system_prompt_classify_no_icl(criteria_query), text)

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
            found_tags = re.finditer(r'\<([a-z A-Z]+)\>([^<]*?)<\/([a-z A-Z]+)\>', return_result)

            # count of tag-characters to subtract to get correct index position in original text
            # could alternatively be done in regex with lookarounds
            cnt = 0
            # iterate over the found tags and count their index position
            for m in found_tags:
                true_start = m.start()-cnt + cnt_docs
                # add the number of tag-symbols and letters in tags
                cnt += 5 + len(m.group(1)) + len(m.group(3))
                if (documenttext[true_start:true_start + len(m.group(2))] == m.group(2)):
                    found_spans.append((m.group(1), (true_start, true_start + len(m.group(2)))))
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
        found_words = [(documenttext[s:e], s, e) for _, (s, e) in found_spans]
        logger.debug(
            "Found these words in the text: %s\n--------------------------------\n", found_words)
        return found_spans

    def classify_span_exactwordcheck2(self, criteria_query: str, scope: str, chunk_size=1000) -> list[tuple[str, tuple[int, int]]]:
        """Classifies spans in the given document-scope that are relevant according to the classification_query. 
        tuple element is the categorization, second is start and end index of the classified text.
        Returns the annotation positions.
        """
        systemprompt = f"""Your job is to identify spans in the text that satisfy this query: {criteria_query}.
Return a list of all the identified spans in the given JSON format where span is the exact text as found in the document. Only reply with the JSON.

Example 1:
Input:
    Query: animals
    Duke asked Lulu to tell him a story about cats and dogs living together in harmony.

Output:
{{
    "spans":
    [{{
        "span": "cats",
        "categorization": "animal",
        "reason": "A cat is an animal"
    }},
    {{
        "span": "dogs",
        "categorization": "animal",
        "reason": "A dog is an animal"
    }}]
}}

Example 2:
Input:
    Query: food
    There is a saying that an apple a day keeps the doctor away. But I much prefer peaches or bananas.

Output:
{{
    "spans":
    [{{
        "span": "apple",
        "categorization": "food",
        "reason": "An apple is a type of food"
    }},
    {{
        "span': "peaches",
        "categorization": "food",
        "reason": "A peach is a type of food"
    }},
    {{
        "span": "bananas",
        "categorization": "food",
        "reason": "A banana is a type of food"
    }}]
}}
"""
        # get text from doc/cas
        if scope == "current document":
            documenttext = self.softwareenv.get_current_documenttext()
        else:  # scope == "all documents":
            documenttext = self.softwareenv.get_current_documenttext()
        # chunk texts
        # long texts may go out of context window and make llm ignore the prompt 'only respond with embedded text'
        # long texts also make llm embelish (e.g. change 'Clinton' to 'Hillary Clinton')
        # changing the length of text and making indexes inaccurate
        # too short texts make llm add additional text (e.g. 'candidates' -> the llm answers)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0, separators=["\r\n\r\n", "\n"], keep_separator="end", strip_whitespace=False)
        text_chunks = text_splitter.split_text(documenttext)

        # the found spans in the following format
        # [(categorization, (start, end)), ..]
        found_spans = []
        cnt_docs = 0

        for text in text_chunks:
            spans = []
            if self.is_ollama:
                return_result = self.callback_llm_with_format(
                    systemprompt, text, JSON_schema_unit_classify_span_exactwordcheck2)
                try:
                    spans = return_result.spans
                except:
                    print("couldnt load json")
            else:  # openai API, no json schema
                try:
                    return_result = self.callback_llm(systemprompt, text)
                    return_result = utils.parse_deepseek_response(return_result)
                    return_result_json = json.loads(return_result)
                    if isinstance(return_result_json, list):
                        for s in return_result_json:
                            spans.append(JSON_schema_unit_classify_span_exactwordcheck2.model_validate(s))
                    else:
                        raise Exception
                except:
                    print("couldnt load json")

            # iterate over the found spans and locate their index position
            for el in spans:
                all_finds = re.finditer(re.escape(el.span), text)
                for find in all_finds:
                    start = find.start() + cnt_docs
                    found_spans.append((el.categorization, (start, start+len(el.span))))
            cnt_docs += len(text)
        return found_spans


class JSON_schema_unit_classify_span_exactwordcheck2_unit(BaseModel):
    span: str
    categorization: str
    reason: str


class JSON_schema_unit_classify_span_exactwordcheck2(BaseModel):
    spans: list[JSON_schema_unit_classify_span_exactwordcheck2_unit]
