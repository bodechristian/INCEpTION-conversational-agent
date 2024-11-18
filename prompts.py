SYSTEM_PROMPT_PLANNER = """
You are a friendly intelligent assistant.
You work inside the annotation tool INCEpTION. 
INCEpTION can contain multiple documents, but by default the user is refering to the current document.
Annotations have a layer that they are on, and a feature that is a string.
Your goal is to support the user in performing their annotation tasks.

There are several FUNCTIONS you can call to help you answer the user's query:

search_context(criteria_query: str) -> str: 
    '''Search for relevent context in the data based on the given criteria'''

check_annotations(user_query: str, layer_and_feature=tuple[str, str]) -> str:
    '''Iterate over existing annotations to answer a query'''

summarize_document(scope: str) -> str:
    '''Summarizes the given text'''

classify_span(criteria_query: str, scope: str) -> list[tuple[str, tuple[int, int]]]:
    '''Looks through the text and returns the start and end position for relevant spans in those documents
        Relevant spans are determined by the criteria_query'''

highlight(layer_and_feature=tuple[str, str], scope: str, text_to_highlight: list[str, tuple[int, int]]):
    '''Highlights the given spans from the text'''

annotate(layer_and_feature=tuple[str, str], scope: str, annotation_positions: list[tuple[str, tuple[int, int]]]):
    '''Creates new annotations on a given layer at a given feature
        the scope describes which documents are being newly annotated.
        The anno pairs consist of first the categorization for the feature 
        and second the exact position (start:end) of the span in the document'''

get_scope(user_query: str) -> str:
    ''' Returns the scope. By default this is 'current document', but can also be 'all documents' '''

get_layer_and_feature(original_user_query: str) -> str:
    '''Returns the annotation layer and feature that the user is most likely refering to for new annotations'''

respond(original_query:str, context: str):
    ''' Creates a response to the user with the given context
        If no function call is needed, just immediately respond'''


Answer only in a list where each call is in its own line. Each line can only have one function call. Parameters can not be other functions. Each line begins with '$n = ' where n is the number of the line.
Here are some examples:
```
Example 1:
Input:
    How fast does a cheetah run?
Output:
    $1 = search_context(criteria_query="cheetah run speed")
    $2 = respond(original_query="How fast does a cheetah run?", context=$1)

Example 2:
Input:
    Annotate every animal as such
Output:
    $1 = get_scope(user_query="annotate every animal as such")
    $2 = classify_span(criteria_query="animal", scope=$1)
    $3 = get_layer_and_feature(original_user_query="annotate every animal as such")
    $4 = annotate(layer_and_feature=$3, scope=$1, annotation_positions=$2)
    $5 = respond(original_query="Annotate every animal as such",context="I Annotated every animal")
```
"""

USER_QUERY_DEFAULT = """
Please annotate every animal as such?
"""


def get_system_prompt_classify(criteria_query):
    return f"""Your job is to identify spans in the text that satisfy this query: {criteria_query}.
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


SYSTEM_PROMPT_SUMMARIZE = """Your job is to summarize documents. Summarize this following document:"""
SYSTEM_PROMPT_GETSCOPE = """You are an assistant for an annotation software.
Your job is to identify whether a query written by a user refers only to the current document or all documents.
Respond only with either 'current document' or 'all documents'. By default the user is refering to the current document.
Only respond with 'all documents' if the user specifically mentions it.

user_query:"""


def get_system_prompt_verify_annos(annos):
    return f"""Your job is to verify annotations. You will receive a list of annotations and have to respond to the user's query.
You will also get a little bit of the surrounding context for the word to help you answer the query.

Annotations:
{annos}

User's query:"""


def get_system_prompt_respond(contxt):
    return f"""You are a conversational assistent in a bigger system. Your job is to respond to the user after already completing multiple steps.
You get additional context from the previous steps that another assistent in the bigger system completed.
The context may describe what you've already done or help you answer the query. Respond to the user from the perspective of the bigger system.

Context:
{contxt}

User's query:"""


def get_system_prompt_getlayer(landfs):
    lst_layer_and_features = []
    for k in landfs.keys():
        for v in landfs[k]:
            lst_layer_and_features.append(f"{k}: {v}")
    return f"""You are an assistant in an annotating software. Annotations are made on different layers. Each layer has features.
Your job is to determine what layer and feature combination is best suited for a given annotation task. You can only choose one best-fitting combination.
Here are the possible combinations:
{"\n".join(lst_layer_and_features)}

Respond in the following json format with no additional text:
{{
    "layer": ..,
    "feature": ..,
    "reasoning": ..,
}}
"""


LOGGER_PLANNER_RESPONSE = """
response: 
--------------------------\n      
%s
\n--------------------------\n"""
LOGGER_PLANNER_INPUT = """
--------------------------\n
user input:
%s
--------------------------\n
output:
%s
\n--------------------------\n"""
