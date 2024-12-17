from inspect import signature


def get_system_prompt_planner(functions):
    functions_string = "".join([f"{fun.__name__}{signature(fun)}:\n\t'''{fun.__doc__}'''\n\n" for fun in functions])
    return f"""You are a friendly intelligent assistant.
You work inside the annotation tool INCEpTION.
INCEpTION can contain multiple documents, but by default the user is refering to the current document.
Annotations have a layer that they are on, and a feature that is a string.
Your goal is to support the user in performing their annotation tasks.

There are several FUNCTIONS you can call to help you answer the user's query:

{functions_string}

Answer only in a list where each function call is in its own line. Parameters can not be other functions.
NEVER nest functions.
Each line begins with '$n = ' where n is the number of the line.
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
```"""


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


def get_system_prompt_search_context(ctxt_string):
    return f"""Answer which segment is best to answer a given user query. Use the number as the id.
Also specify which segment of that text helps the most to answer the user query and write it word for word in the 'text' field.
Respond only in JSON:

JSON Format:
{{
    "id": ..,
    "text": ..,
    "reason": ..
}}

segments:
{ctxt_string}

user query:
"""


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
plan:
%s
\n--------------------------\n"""

SYSTEM_PROMPT_TOOLCALLING = """
You are an assistant for an annotation software. Your job is to help execute users queries.
You have functions you can call to gather information that may be required for other functions.
These informations are stored in your memory. The user_query is already stored in your memory.
Work step by step and only call one function at a time.

Here is an example process:

```
user_query:
    annotate all politicians

process:
    - get_scope
    - get_layer_and_feature
    - classify_span
    - annotate
```
```
user_query:
    who won the election?

process:
    - search_context('election winner')
```

user_query:
"""
TOOLCALLING_INPUT = """--------------------------\n
user input:
%s
            
--------------------------\n"""

TOOLCALLING_OUTPUT = """%s

--------------------------"""
TOOLCALLING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_context",
            "description": "Searches for relevant context in the document text. Helps retrieving more information about a topic. This returns no information about annotations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key_phrase": {
                        "type": "string",
                        "description": "The topic you want to know more about.",
                    }
                },
                "required": ["key_phrase"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_annotations",
            "description": "Analyze existing annotations in the document. Requires a specific layer and feature",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "Summarizes a document. Depending of the scope of the user query, this might be the current document or all documents.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_span",
            "description": "Classifies spans in the document that are relevant according to the classification_criteria. These are then saved in memory as annotation_poisitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "classification_query": {
                        "type": "string",
                        "description": "The query to classify spans by in the document. Can be whole sentences that include logic.",
                    }
                },
                "required": ["classification_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "highlight",
            "description": "Highlights the spans, that are saved in the memory, in the document",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "annotate",
            "description": '''Creates new annotations on a given layer at a given feature. Uses the spans in the memory to do so. Should only be done if user specifically asks to create annotations.''',
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scope",
            "description": "Analyzes the user query and returns which documents should be considered. By default this returns 'current document'. If specifically asked for in the user query, this might return 'all documents'. This is then saved in memory.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_layer_and_feature",
            "description": "Returns the annotation layer and annotation feature you should use for new annotations. This is then saved in memory.",
        },
    },
]
