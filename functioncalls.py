import re
import logging
import os

from groq import Groq

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')


class void_INCEpTION_UI:
    """empty class that indicates that the function has sideeffects in the INCEpTION UI
    otherwise void"""
    pass


def chat(user_query: str) -> str:
    "The user wants to chat"
    logging.debug("inside chat\tparameters:\tuser_query=%s", user_query)


def search_context(criteria: str) -> str:
    """Search for relevent chunks in the text based on the given criteria"""
    """Takes criteria and returns top-k chunks from Vector Store (RAG)"""
    logging.debug("inside search\tparameters:\tcriteria=%s", criteria)


def check_annotations(layer: str, feature: str, user_query: str) -> str:
    """Iterate over annotations either solely annotations or with sliding context-window"""
    logging.debug("\ninside check_annotations\nparameters:")
    logging.debug(f"{layer=}\n{feature=}\n{user_query=}")


def summarize_document(scope: str) -> str:
    """Classifies text based on the scope"""
    logging.debug("\ninside summarize\nparameters:")
    logging.debug(f"{scope=}\n")


def classify_span(criteria_query: str, scope: str) -> list[tuple[str, tuple[int, int]]]:
    """Iterates over text determined by the scope and classifies test based on the criteria
        First tuple element is the criteria, second is the classified text"""
    logging.debug("\ninside classify_span\nparameters:")
    logging.debug(f"{criteria_query=}\n{scope=}\n")

    # TODO: get text from doc
    # faking it right now
    text = "The Republican ticket, businessman Donald Trump and Indiana governor Mike Pence, defeated the Democratic ticket of former secretary of state and First Lady of the United States Hillary Clinton. LeBron James voted that year. Johannes Fliederman is a local german politician."

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
        The Republican ticket, businessman <politician>Donald Trump<politician> and Indiana governor <politician>Mike Pence<politician>, defeated the Democratic ticket of former secretary of state and First Lady of the United States <politician>Hillary Clinton<politician><politician>."""

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
    logging.debug(f"\n{SYSTEM_PROMPT_CLASSIFY}\n{text}\n")
    logging.debug(f"\noutput:\n{return_result}")

    # extract tags
    resis = re.finditer(
        r'\<([a-z A-Z]+)\>([a-z A-Z]+)\<\/([a-z A-Z]+)\>', return_result)
    cnt = 0
    ressi = []
    for m in resis:
        true_start = m.start()-cnt
        cnt += 5 + len(m.group(1)) + len(m.group(3))
        ressi.append(
            (m.group(1), (true_start, true_start + len(m.group(2)))))
    logging.debug(ressi)

    # safety test, applying start:end onto the initial text
    logging.debug([text[s:e] for _, (s, e) in ressi])
    return ressi


def highlight(layer: str, feature: str, text_to_highlight: list[tuple[str, tuple[int, int]]]) -> void_INCEpTION_UI:
    """Highlights the given spans from the text"""
    logging.debug("\ninside highlight\nparameters:")
    logging.debug(f"{layer=}\n{feature=}\n{text_to_highlight=}\n")


def annotate(layer: str, feature: str, scope: str, annotation_positions: list[tuple[str, tuple[int, int]]]) -> void_INCEpTION_UI:
    """First finds most appropriate Layer and Feature from user query
        then annotates on it
        The anno pairs consist of first the text for the feature
        and second the exact corresponding span in the text"""
    logging.debug("\ninside annotate\nparameters:")
    logging.debug(f"{layer=}\n{feature=}\n{scope=}\n{annotation_positions=}\n")


def get_scope(user_query: str):
    logging.debug("\ninside get_scope\nparameters:")
    logging.debug(f"{user_query=}\n")
    return "current document"


def get_text(scope: str):
    logging.debug("\ninside get_text\nparameters:")
    logging.debug(f"{scope=}\n")
    return "The Republican ticket, businessman Donald Trump and Indiana governor Mike Pence, defeated the Democratic ticket of former secretary of state and First Lady of the United States Hillary Clinton. LeBron James voted that year. Johannes Fliederman is a local german politician."


def get_layer(original_user_query: str):
    logging.debug("\ninside get_layer\nparameters:")
    logging.debug(f"{original_user_query=}\n")


def get_feature(original_user_query: str, layer: str):
    logging.debug("\ninside get_feature\nparameters:")
    logging.debug(f"{original_user_query=}\n{layer=}\n")


def respond(context: str) -> void_INCEpTION_UI:
    """Create a response for the user summarizing the functions/intents called 
        and the previous output
        Afterwards respond in the chat window"""
    """Potential Prompt, also get initialy user query as parameter
    Justify how well you answered the user query"""
    logging.debug("\ninside respond\nparameters:")
    logging.debug(f"{context=}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    classify_span("Annotate all politicians",
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
