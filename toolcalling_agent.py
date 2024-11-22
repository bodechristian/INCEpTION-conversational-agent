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
            "description": "Searches for relevant context in the documents and uses that to answer the user query",
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
            "description": "Classifies spans in the document that are relevant according to the user query in the memory. These are then saved in memory.",
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
You are an assistant for an annotation software. Your job is to help execute users queries.
You have functions you can call to gather information that may be required for other functions.
These informations are stored in your memory. Functions do not need parameters as they can read the memory as well.
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
"""

# USER_QUERY = """How fast does a cheetah run?"""
# USER_QUERY = """Hi how are you?"""
USER_QUERY = """highlight all animals"""


def print_memory(mem):
    return f"Your current memory contains values for {", ".join(mem.keys())}"


# cerebras: llama3.1-70b, groq:llama3-groq-70b-8192-tool-use-preview
ag = Agent(toolcalling_functions=True)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = Groq(
    api_key=GROQ_API_KEY,
)
CEREBRAS_API_KEY = os.getenv('CEREBRAS_API_KEY')
# client = Cerebras(
#     api_key=CEREBRAS_API_KEY,
# )
ag.state['user_query'] = USER_QUERY
ag.logger.info("""--------------------------\n
user input:
%s
               
--------------------------\n""", USER_QUERY)
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
    model="llama3-groq-70b-8192-tool-use-preview",
    tools=TOOLS,
    temperature=0.0,
    parallel_tool_calls=False,
)
return_result = chat_completion.choices[0].message

if chat_completion.choices[0].finish_reason == "stop":
    # no tools need to be called, just respond
    ag.logger.info(f"""{return_result.content}

--------------------------""")
else:
    # tools are called, keep calling them until llm says stop
    while chat_completion.choices[0].finish_reason != "stop":
        tool_calls = return_result.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                ag.logger.debug("CALLED: ", tool_call)
                func = ag.functionclass.valid_functions[tool_call.function.name]
                arguments = json.loads(tool_call.function.arguments)

                response = func(**arguments)
                messages.append(return_result)
                messages.append({'role': 'tool', 'content': response, 'tool_call_id': tool_call.id})
        [ag.logger.debug(m) for m in messages]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-groq-70b-8192-tool-use-preview",
            tools=TOOLS,
            temperature=0.0,
            parallel_tool_calls=False,
        )
        return_result = chat_completion.choices[0].message
    # this response considers what was done and the state
    # therefore it should be better than just the normal llm content response
    ag.functionclass.respond()


# DUMP

"""nested functions
# check if nested functions exist in arguments and run them
for key, val in arguments.items():
    if type(val) is dict:
        # nested function will only have 1 key (name of the new function)
        nested_func = list(val.keys())[0]
        arguments[key] = ag.functionclass.valid_functions[nested_func](**list(val.values())[0])"""
