import re
import logging
import os
import sys

from groq import Groq
from dotenv import load_dotenv

from software_environment import Software_environment

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')


class void_INCEpTION_UI:
    """empty class that indicates that the function has sideeffects in the INCEpTION UI
    otherwise void"""
    pass


class Agentfunctions():
    def __init__(self, softwareenv) -> None:
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

    def search_context(self, criteria: str) -> str:
        """Search for relevent chunks in the text based on the given criteria"""
        """Takes criteria and returns top-k chunks from Vector Store (RAG)"""
        logger = logging.getLogger("tests")
        logger.debug("inside search\tparameters:\tcriteria=%s", criteria)

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

        # TODO: get text from doc/cas
        # faking it right now
        text = "The Republican ticket, businessman Donald Trump and Indiana governor Mike Pence, defeated the Democratic ticket of former secretary of state and First Lady of the United States Hillary Clinton. LeBron James voted that year. Johannes Fliederman is a local german politician."

        if scope == "current document":
            text = ""  # TODO

        client = Groq(
            api_key=GROQ_API_KEY,
        )

        SYSTEM_PROMPT_CLASSIFY = f"""Your job is to identify spans in the text that satisfy this query: {criteria_query}.
        Wrap each identified span into a tag, where you describe the criteria. Such as <animal>dog</animal>
        only respond with the embedded sentence.

        Example 1:
        Input:
            Query: animals
            Duke asked Lulu to tell him a story about cats and dogs living together in harmony.

        Output:
            Duke asked Lulu to tell him a story about <animal>cats</animal> and <animal>dogs</animal> living together in harmony.

        Example 2:
        Input:
            Query: politicians
            The Republican ticket, businessman Donald Trump and Indiana governor Mike Pence, defeated the Democratic ticket of former secretary of state and First Lady of the United States Hillary Clinton.

        Output:
            The Republican ticket, businessman <politician>Donald Trump</politician> and Indiana governor <politician>Mike Pence</politician>, defeated the Democratic ticket of former secretary of state and First Lady of the United States <politician>Hillary Clinton</politician>."""

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
        logger.debug(f"\n{SYSTEM_PROMPT_CLASSIFY}\n{text}\n")
        logger.debug(f"\noutput:\n{return_result}")

        # extract tags
        found_tags = re.finditer(
            r'\<([a-z A-Z]+)\>([^<]*?)<\/([a-z A-Z]+)\>', return_result)

        # count of tag-characters to subtract to get correct index position in original text
        # could alternatively be done in regex with lookarounds
        cnt = 0
        # the found spans in the following format
        # [(categorization, (start, end)), ..]
        found_spans = []
        # iterate over the found tags and count their index position
        for m in found_tags:
            true_start = m.start()-cnt
            # add the number of tag-symbols and letters in tags
            cnt += 5 + len(m.group(1)) + len(m.group(3))
            found_spans.append(
                (m.group(1), (true_start, true_start + len(m.group(2)))))
        logger.debug(found_spans)

        # safety test, applying start:end onto the initial text
        found_words = [text[s:e] for _, (s, e) in found_spans]
        logger.debug("Found these words in the text: %s", found_words)
        return found_spans

    def highlight(self, layer: str, feature: str, text_to_highlight: list[tuple[str, tuple[int, int]]]) -> void_INCEpTION_UI:
        """Highlights the given spans from the text"""
        logger = logging.getLogger("tests")
        logger.debug("\ninside highlight\nparameters:")
        logger.debug(f"{layer=}\n{feature=}\n{text_to_highlight=}\n")

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
        return "current document"

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

    def respond(self, context: str) -> void_INCEpTION_UI:
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
    logger.setLevel(logging.INFO)
    logger.addHandler(stdout)

    softwareenv = Software_environment()
    functionclass = Agentfunctions(softwareenv)
    functionclass.classify_span("Annotate all politicians",
                                "The Republican ticket, businessman Donald Trump and Indiana governor Mike Pence, defeated the Democratic ticket of former secretary of state and First Lady of the United States Hillary Clinton. LeBron James voted that year. Johannes Fliederman is a local german politician."
                                )


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
                                    [("politician", (35, 47)), ("politician", (69, 79), ("politician", (177, 192))] """
