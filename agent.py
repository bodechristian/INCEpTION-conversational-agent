import sys
import parser
import logging
import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

SYSTEM_PROMPT = """
You are a friendly intelligent assistant.
You work inside the annotation tool INCEpTION. 
INCEpTION can contain multiple documents, but by default the user is refering to the current document.
Annotations have a layer that they are on, and a feature that is a string.
Your goal is to support the user in performing their annotation tasks.

There are several FUNCTIONS you can call to help you answer the user's query:

search_context(criteria_query: str) -> str: 
    '''Search for relevent context in the data based on the given criteria'''

check_annotations(user_query: str, layer:str, feature: str) -> str:
    '''Iterate over existing annotations to answer a query'''

summarize_document(scope: str) -> str:
    '''Summarizes the given text'''

classify_span(criteria_query: str, scope: str) -> list[tuple[str, tuple[int, int]]]:
    '''Looks through the text and returns the start and end position for relevant spans in those documents
        Relevant spans are determined by the criteria_query'''

highlight(layer: str, feature: str, scope: str, text_to_highlight: list[str, tuple[int, int]]):
    '''Highlights the given spans from the text'''

annotate(layer: str, feature: str, scope: str, annotation_positions: list[tuple[str, tuple[int, int]]]):
    '''Creates new annotations on a given layer at a given feature
        the scope describes which documents are being newly annotated.
        The anno pairs consist of first the categorization for the feature 
        and second the exact position (start:end) of the span in the document'''

get_scope(user_query: str) -> str:
    ''' Returns the scope. By default this is 'current document', but can also be 'all documents' '''

get_layer(original_user_query: str) -> str:
    '''Returns the annotation layer that the user is most likely refering to for new annotations'''

get_feature(original_user_query: str, layer: str) -> str:
    '''Returns the annotation feature on a specific layer that the user is most likely refering to for new annotations'''

respond(context: str):
    ''' Creates a response to the user with the given context
        If no function call is needed, just immediately respond'''


Answer only in a list where each call is in its own line. Each line can only have one function call. Parameters can not be other functions. Each line begins with '$n = ' where n is the number of the line.
Here are some examples:
```
Example 1:
Input:
    How fast does a cheetah run?
Output:
    $1 = search_context(criteria_query="How fast does a cheetah run?")
    $2 = respond(context=$1)

Example 2:
Input:
    Annotate every animal as such
Output:
    $1 = get_scope(user_query="annotate every animal as such")
    $2 = classify_span(criteria_query="animal", scope=$1)
    $3 = get_layer(original_user_query="annotate every animal as such")
    $4 = get_feature(original_user_query="annotate every animal as such", layer=$3)
    $5 = annotate(layer=$3, feature=$4, scope=$1, annotation_positions=$2)
    $6 = respond(context="I Annotated every animal")
```
"""
USER_QUERY = """
Please annotate every animal as such?
"""


def call_llm_planner(client, model, user_query):

    chat_completion = client.chat.completions.create(
        messages=[
            # system prompt
            {
                "role": "system",
                "content": SYSTEM_PROMPT,

            },
            {
                "role": "user",
                "content": user_query,
            }
        ],
        model=model,
        temperature=0.0
    )
    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    logger = logging.getLogger("tests")
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)
    logger.addHandler(stdout)
    # check if user prompt was given
    if len(sys.argv) > 1:
        USER_QUERY = sys.argv[1]

    client = Groq(
        api_key=GROQ_API_KEY,
    )

    llm_response = call_llm_planner(client, "llama3-70b-8192", USER_QUERY)

    # printing response
    logger.debug("System prompt:\n%s", SYSTEM_PROMPT)
    logger.info("""
--------------------------\n
user input:
%s
--------------------------\n
output:
%s
\n--------------------------\n""", USER_QUERY, llm_response)
    # call parser and functions
    parsed_dollar_lines = parser.parse_dollars_lines(llm_response)
    logger.info("""
response: 
--------------------------\n      
%s
\n--------------------------\n""", parsed_dollar_lines)
