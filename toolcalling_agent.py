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
            "description": "Searches for relevent context in the documents and uses that to answer the user query",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_annotations",
            "description": "Iterate over existing annotations on a layer (and maybe also a feature) to answer the user query",
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
            "description": "Classifies spans in the document that are relevant according to the user query. These are then saved in memory.",
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
            "description": '''Creates new annotations on a given layer at a given feature. Uses the spans in the memory to do so''',
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
SYSTEM_PROMPT = """
You are a friendly intelligent assistant.
You work inside the annotation tool INCEpTION. 
INCEpTION can contain multiple documents, but by default the user is refering to the current document.
You may call multiple functions.
Annotations have a layer that they are on, and a feature that is a string.
Your goal is to support the user in performing their annotation tasks.
You might need to call some functions before others to get more info about the query. 
Such as getting the scope first, so that later functions can have that as extra information.
You have a memory where information such as scope, layer and more is stored.
These informations do not have to be given to functions as parameters, as they will be taken from the memory.
Work step by step.
"""

USER_QUERY = """Highlight every animal"""


def print_memory(mem):
    return f"Your current memory contains values for {", ".join(mem.keys())}"


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
ag.state['user_query'] = USER_QUERY
messages = [
    # system prompt
    {
        "role": "system",
        "content": SYSTEM_PROMPT,

    },
    {
        'role': 'assistant',
        'content': print_memory(ag.state)
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
