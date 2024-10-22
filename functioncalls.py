import re
import logging
import os
import math
import sys

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
    def __init__(self, softwareenv: Software_environment) -> None:
        self.softwareenv = softwareenv

        self.valid_functions = {
            "chat": self.chat,
            "search_context": self.search_context,
            "check_annotations": self.check_annotations,
            "summarize": self.summarize_document,
            "classify_span": self.classify_span,
            "get_layer": self.get_layer,
            "get_scope": self.get_scope,
            "annotate": self.annotate,
            "highlight": self.highlight,
            "get_feature": self.get_feature,
            "respond": self.respond
        }

    def chat(self, user_query: str) -> str:
        "The user wants to chat"
        logger = logging.getLogger("tests")
        logger.debug("inside chat\tparameters:\tuser_query=%s", user_query)

    def search_context(self, criteria_query: str) -> str:
        """Search for relevent chunks in the text based on the given criteria"""
        """Takes criteria and returns top-k chunks from Vector Store (RAG)"""
        logger = logging.getLogger("tests")
        logger.debug(
            "inside search\tparameters:\tcriteria_query=%s", criteria_query)

    def check_annotations(self, layer: str, feature: str, user_query: str) -> str:
        """Iterate over annotations either solely annotations or with sliding context-window"""
        logger = logging.getLogger("tests")
        logger.debug("\ninside check_annotations\nparameters:")
        logger.debug(f"{layer=}\n{feature=}\n{user_query=}")

    def summarize_document(self, scope: str) -> str:
        """Classifies text based on the scope"""
        logger = logging.getLogger("tests")
        logger.debug("\ninside summarize\nparameters:")
        logger.debug(f"{scope=}\n")

    def classify_span(self, criteria_query: str, scope: str) -> list[tuple[str, tuple[int, int]]]:
        """Iterates over text determined by the scope and classifies test based on the criteria
            First tuple element is the criteria, second is the classified text"""
        logger = logging.getLogger("tests")
        logger.debug("\ninside classify_span\nparameters:")
        logger.debug(f"{criteria_query=}\n{scope=}\n")

        # get text from doc/cas
        if scope == "current document":
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
        client = Groq(
            api_key=GROQ_API_KEY,
        )

        SYSTEM_PROMPT_CLASSIFY = f"""Your job is to identify spans in the text that satisfy this query: {criteria_query}.
        Wrap each identified span into a tag, where you describe the criteria. Such as <animal>dog</animal>.
        Respond only with the given text and their embedded tags. Dont write anything that isn't in the text.
        Pay special attention to using the same whitespace and newline characters as the input.

        Example 1:
        Input:
            Query: animals
            Duke asked Lulu to tell him a story about cats and dogs living together in harmony.

        Output:
            Duke asked Lulu to tell him a story about <animal>cats</animal> and <animal>dogs</animal> living together in harmony.

        Example 2:
        Input:
            Query: food
            There is a saying that an apple a day keeps the doctor away. But I much prefer peaches or bananas.

        Output:
            There is a saying that an <food>apple</food> a day keeps the doctor away. But I much prefer <food>peaches</food> or <food>bananas</food>."""

        # the found spans in the following format
        # [(categorization, (start, end)), ..]
        found_spans = []
        cnt_docs = 0

        for text in text_chunks:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_CLASSIFY,
                    },
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                model="llama3-70b-8192",
                temperature=0.0
            )
            return_result = chat_completion.choices[0].message.content
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

    def highlight(self, layer: str, feature: str, scope: str, text_to_highlight: list[tuple[str, tuple[int, int]]]) -> void_INCEpTION_UI:
        """Highlights the given spans from the text"""
        logger = logging.getLogger("tests")
        logger.debug("\ninside highlight\nparameters:")
        logger.debug(f"{layer=}\n{feature=}\n{scope=}\n{text_to_highlight=}\n")

    def annotate(self, layer: str, feature: str, scope: str, annotation_positions: list[tuple[str, tuple[int, int]]]) -> void_INCEpTION_UI:
        """First finds most appropriate Layer and Feature from user query
            then annotates on it
            The anno pairs consist of first the text for the feature
            and second the exact corresponding span in the text"""
        logger = logging.getLogger("tests")
        logger.debug("\ninside annotate\nparameters:")
        logger.debug(f"{layer=}\n{feature=}\n{scope=}\n{
                     annotation_positions=}\n")

    def get_scope(self, user_query: str):
        """analyzes the user_query and returns the document id(s) of the relevant document"""
        logger = logging.getLogger("tests")
        logger.debug("inside get_scope\nparameters:")
        logger.debug(f"{user_query=}\n")

        client = Groq(
            api_key=GROQ_API_KEY,
        )

        SYSTEM_PROMPT_GETSCOPE = f"""You are an assistant for an annotation software.
Your job is to identify whether a query written by a user refers only to the current document or all documents.
Respond only with either 'current document' or 'all documents'. By default the user is refering to the current document.
Only respond with 'all documents' if the user specifically mentions it.

user_query:"""

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_GETSCOPE,
                },
                {
                    "role": "user",
                    "content": user_query,
                }
            ],
            model="llama3-70b-8192",
            temperature=0.0
        )
        return_result = chat_completion.choices[0].message.content

        return return_result

    def get_text(self, scope: str):
        logger = logging.getLogger("tests")
        logger.debug("\ninside get_text\nparameters:")
        logger.debug(f"{scope=}\n")
        return "The Republican ticket, businessman Donald Trump and Indiana governor Mike Pence, defeated the Democratic ticket of former secretary of state and First Lady of the United States Hillary Clinton. LeBron James voted that year. Johannes Fliederman is a local german politician."

    def get_layer(self, original_user_query: str):
        logger = logging.getLogger("tests")
        logger.debug("\ninside get_layer\nparameters:")
        logger.debug(f"{original_user_query=}\n")

    def get_feature(self, original_user_query: str, layer: str):
        logger = logging.getLogger("tests")
        logger.debug("\ninside get_feature\nparameters:")
        logger.debug(f"{original_user_query=}\n{layer=}\n")

    def respond(self, context: str) -> str:
        """Create a response for the user summarizing the functions/intents called 
            and the previous output
            Afterwards respond in the chat window"""
        """Potential Prompt, also get initialy user query as parameter
        Justify how well you answered the user query"""
        logger = logging.getLogger("tests")
        logger.debug("\ninside respond\nparameters:")
        logger.debug(f"{context=}\n")
        return context


if __name__ == "__main__":
    logger = logging.getLogger("tests")
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stdout)

    softwareenv = Software_environment()
    functionclass = Agentfunctions(softwareenv)
    # functionclass.classify_span("Annotate all politicians",
    #                             "current document")
    print(functionclass.get_scope("annotate all politicians"))


# TEMPDUMP

"""
CLASSIFY SPAN 
Your job is to identify spans in the text that satisfy this query: {criteria}.
                                for each identified span return a tuple of the classification and the start and end position of the span.
                                The position is counted by each letter, punctuation and whitespace counting as 1 position.
                                Answer only with the list.
                                Here is an example of an input and an output, but for a different criteria:

                                Example 1:
                                Input:
                                    Query: animals
                                    Duke asked Lulu to tell him a story about cats and dogs living together in harmony.

                                Thought:
                                    The words 'cats' and 'dogs' satisfy the query. 'cats' start position in the text is 42 and ends at 46.
                                    'dogs' starts at 51 and ends at 55.

                                Output:
                                    [("animals", (42, 46)), ("animals", (51, 55))]


                                Example 2:
                                Input:
                                    Query: politicians
                                    The Republican ticket, businessman Donald Trump and Indiana governor Mike Pence, defeated the Democratic ticket of former secretary of state and First Lady of the United States Hillary Clinton.

                                Thought:
                                    Donald Trump is a politician and at position 35:47, Mike pence is at position 69:79 and Hillary Clinton is at 177:192.

                                Output:
                                    [("politician", (35, 47)), ("politician", (69, 79), ("politician", (177, 192))] 
                                    
                Example 2:
        Input:
            Query: politicians
            The Republican ticket, businessman Donald Trump and Indiana governor Mike Pence, defeated the Democratic ticket of former secretary of state and First Lady of the United States Hillary Clinton.

        Output:
            The Republican ticket, businessman <politician>Donald Trump</politician> and Indiana governor <politician>Mike Pence</politician>, defeated the Democratic ticket of former secretary of state and First Lady of the United States <politician>Hillary Clinton</politician>."""
