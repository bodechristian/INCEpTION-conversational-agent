import json
import os
from groq import Groq

from agent import Agent
from cerebras.cloud.sdk import Cerebras

# from functioncalling import *

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_context",
            "description": "Search for relevent context in the documents based on the given criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria_query": {
                        "type": "string",
                        "description": "The criteria or query to look for",
                    }
                },
                "required": ["criteria_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_annotations",
            "description": "Iterate over existing annotations to answer the user_query",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "The user query to answer while iterating over the existing annotations",
                    },
                    "layer_and_feature": {
                        "type": {
                            "type": "string",
                            "description": "The layer and the feature to analyze",
                        }
                    }
                },
                "required": ["user_query", "layer_and_feature"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "Summarizes a document",
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "scope over the documents of the query",
                        "enum": ['current document', 'all documents'],
                    },
                },
                "required": ["scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_span",
            "description": "Classifies spans in the document that are relevant according to the critera_query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria_query": {
                        "type": "string",
                        "description": "The categorization to identify spans by",
                    },
                    "scope": {
                        "type": "string",
                        "description": "The scope of the document to analyze",
                        "enum": ['current document', 'all documents'],
                    },
                },
                "required": ["criteria_query", "scope"],

            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "highlight",
            "description": "Highlights the given spans from the text",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_and_feature": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "An array of length 2 containing [layer, feature]",
                    },
                    "scope": {
                        "type": "string",
                        "description": "The scope of the document to analyze",
                        "enum": ['current document', 'all documents'],
                    },
                    "text_to_highlight": {
                        "type": "object",
                        "description": "The spans with start and end position that are to be highlighted",
                    },
                },
                "required": ["text_to_highlight"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "annotate",
            "description": '''Creates new annotations on a given layer at a given feature
                                The annotation positions consist of first the text for the feature 
                                and second the exact position of the span in the text''',
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_and_feature": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "An array of length 2 containing [layer, feature]",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Which documents to annotate",
                        "enum": ['current document', 'all documents'],
                    },
                    "annotation_positions": {
                        "type": "array",
                        "description": "The description of the feature and the start and end position of the span",
                    },
                },
                "required": ["layer", "feature", "annotation_positions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scope",
            "description": "Returns which documents should be considered. By default is 'current document'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "The user query to analyze",
                    },
                },
                "required": ["user_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_layer_and_feature",
            "description": "Returns the annotation layer and annotation feature you should use for new annotations",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_user_query": {
                        "type": "string",
                        "description": "The user query to analyze",
                    },
                },
                "required": ["original_user_query"],
            },
        },
    },
]
SYSTEM_PROMPT = """
You are a friendly intelligent assistant.
You work inside the annotation tool INCEpTION. 
INCEpTION can contain multiple documents, but by default the user is refering to the current document.
You may call multiple functions.
Annotations have a layer that they are on, and a feature that is a string.
Your goal is to support the user in performing their annotation tasks.
You might need to call some functions before others to get more info about the query. 
Such as getting the scope first, so that later functions can have that as extra information.
Work step by step.
"""

USER_QUERY = """Highlight every animal"""

# cerebras: llama3.1-70b, groq:llama3-70b-8192
ag = Agent(toolcalling_functions=True)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# client = Groq(
#     api_key=GROQ_API_KEY,
# )
CEREBRAS_API_KEY = os.getenv('CEREBRAS_API_KEY')
client = Cerebras(
    api_key=CEREBRAS_API_KEY,
)
messages = [
    # system prompt
    {
        "role": "system",
        "content": SYSTEM_PROMPT,

    },
    {
        "role": "user",
        "content": USER_QUERY,
    }
]

chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama3.1-70b",
    tools=TOOLS,
    temperature=0.0,
    parallel_tool_calls=True,
)
print(chat_completion)
return_result = chat_completion.choices[0].message

while chat_completion.choices[0].finish_reason != "stop":
    print(f"\nin loop:")
    print(ag.state)
    tool_calls = return_result.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            print("CALLED: ", tool_call)
            func = ag.functionclass.valid_functions[tool_call.function.name]
            arguments = json.loads(tool_call.function.arguments)

            # # check if nested functions exist in arguments and run them
            # for key, val in arguments.items():
            #     if type(val) is dict:
            #         # nested function will only have 1 key (name of the new function)
            #         nested_func = list(val.keys())[0]
            #         arguments[key] = ag.functionclass.valid_functions[nested_func](**list(val.values())[0])

            response = func(**arguments)
            messages.append(return_result)
            messages.append({'role': 'tool', 'content': response, 'tool_call_id': tool_call.id})
    print()
    [print(el) for el in messages]
    print()
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3.1-70b",
        tools=TOOLS,
        temperature=0.0,
        parallel_tool_calls=True,
    )
    print(chat_completion)
    return_result = chat_completion.choices[0].message
