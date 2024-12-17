import sys
import logging
import json
import os
import time

from prompts import *
from argparse import ArgumentParser
from cerebras.cloud.sdk import Cerebras
from parser import Dollarparser
from functioncalls import Agentfunctions
from mock_annotation_tool import MockAnnotationTool
from groq import Groq

from toolcalling_functioncalls import AgentfunctionsToolcalling
import utils


class Agent():

    # cerebras: llama3.1-70b, groq:llama3-70b-8192, llama3-groq-70b-8192-tool-use-preview
    def __init__(self, model="llama3.1-70b", client="cerebras", toolcalling_functions=False, testing=False, debug=False) -> None:
        self.GROQ_API_KEY = os.environ['GROQ_API_KEY']
        self.CEREBRAS_API_KEY = os.environ['CEREBRAS_API_KEY']

        self.state = {}
        self.toolcalling_functions = toolcalling_functions
        self.testing = testing
        self.debug = debug

        # creating logger
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(logging.DEBUG)
        self.logger = logging.getLogger("output")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.addHandler(stdout)
        l = logging.getLogger("functions")
        if debug:
            l.setLevel(logging.DEBUG)
        else:
            l.setLevel(logging.INFO)
        l.addHandler(stdout)

        # initialize api and software env
        self.model = model
        self.softwareenv = MockAnnotationTool()

        self.set_toolcalling_functions(toolcalling_functions)
        self.set_client(client)
        # toolcalling method uses this for tool-calls, as non-finetuned models often return invalid reponses
        self.client_toolcalling = Groq(
            api_key=self.GROQ_API_KEY,
        )
        self.model_toolcalling = "llama3-groq-70b-8192-tool-use-preview"

    def set_client(self, client):
        if client == "groq":
            self.client = Groq(
                api_key=self.GROQ_API_KEY,
            )
        elif client == "cerebras":
            self.client = Cerebras(
                api_key=self.CEREBRAS_API_KEY,
            )

    def set_toolcalling_functions(self, is_toolcalling):
        self.toolcalling_functions = is_toolcalling
        if is_toolcalling:
            self.functionclass = AgentfunctionsToolcalling(
                self.softwareenv, self.call_llm, self.get_state, testing=self.testing)
        else:
            self.functionclass = Agentfunctions(self.softwareenv, self.call_llm)
            self.parser = Dollarparser(self.functionclass)

    def call_llm(self, system_prompt, user_prompt):
        chat_completion = self.client.chat.completions.create(
            messages=[
                # system prompt
                {
                    "role": "system",
                    "content": system_prompt,

                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model=self.model,
            temperature=0.0
        )
        llm_response = chat_completion.choices[0].message.content
        return llm_response

    def call_llm_planner(self, user_query, execute_functions=True):
        system_prompt_planner = get_system_prompt_planner(self.functionclass.valid_functions.values())
        llm_response = self.call_llm(system_prompt_planner, user_query)
        # printing response
        self.logger.debug("System prompt:\n%s", system_prompt_planner)
        self.logger.info(LOGGER_PLANNER_INPUT, user_query, llm_response)
        if execute_functions:
            # call parser and functions
            parsed_dollar_syntax = self.parser.analyze_and_execute_dollar_syntax(llm_response)
            self.logger.info(LOGGER_PLANNER_RESPONSE, parsed_dollar_syntax)
        return llm_response

    def get_state(self):
        return self.state

    def call_llm_toolcalling(self, user_query):
        self.state['user_query'] = user_query
        self.logger.info(TOOLCALLING_INPUT, user_query)
        messages = [
            # system prompt
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TOOLCALLING,

            },
            {
                "role": "user",
                "content": user_query,
            }
        ]
        [self.logger.debug(m) for m in messages]
        try:
            chat_completion = self.client_toolcalling.chat.completions.create(
                messages=messages,
                model=self.model_toolcalling,
                tools=TOOLCALLING_TOOLS,
                temperature=0.0,
                parallel_tool_calls=False,
            )
            return_result = chat_completion.choices[0].message
        except Exception as error:
            self.logger.debug(error)
            return 'Unable to parse LLM response'

        if chat_completion.choices[0].finish_reason == "stop":
            # no tools need to be called, just respond
            self.logger.info(TOOLCALLING_OUTPUT, return_result.content)
            return messages
        else:
            # tools are called, keep calling them until llm says stop
            while chat_completion.choices[0].finish_reason != "stop":
                # if tool calls exist
                if return_result.tool_calls:
                    # execute each of them (should only be one usually)
                    for tool_call in return_result.tool_calls:
                        self.logger.debug(f"CALLED: {tool_call}\n")
                        # extract function and arguments
                        func = self.functionclass.valid_functions[tool_call.function.name]
                        arguments = json.loads(tool_call.function.arguments)
                        # execute function
                        response = func(**arguments)
                        # append functioncall and the response to LLM messages
                        messages.append(return_result)
                        messages.append({'role': 'tool', 'content': response,
                                        'tool_call_id': tool_call.id, 'name': tool_call.function.name})
                [self.logger.debug(m) for m in messages]
                # call LLM again with new appended messages
                try:
                    chat_completion = self.client_toolcalling.chat.completions.create(
                        messages=messages,
                        model=self.model_toolcalling,
                        tools=TOOLCALLING_TOOLS,
                        temperature=0.0,
                        parallel_tool_calls=False,
                    )
                    return_result = chat_completion.choices[0].message
                except Exception as error:
                    self.logger.debug(error)
                    return 'Unable to parse LLM response'
            # this response considers what was done and the state
            # therefore it should be better than just the normal llm content response
            self.functionclass.respond()
            self.logger.info(return_result.content)
            self.logger.debug(f"Functions that were called: {utils.get_toolcalls_from_messages(messages)}")
            return messages

    def direct(self, prompt):
        # differentiate between toolcalling method and planner method
        if self.toolcalling_functions:
            # call the toolcalling method
            self.call_llm_toolcalling(prompt)
        else:
            # call the llm planner method
            self.call_llm_planner(prompt)

    def run(self):
        # differentiate between toolcalling method and planner method
        if self.toolcalling_functions:
            # call the toolcalling method
            func_call = self.call_llm_toolcalling
        else:
            # call the llm planner method
            func_call = self.call_llm_planner

        # start conversation
        while True:
            # wait for user input
            user_prompt = input("Enter prompt: ")
            func_call(user_prompt)
            time.sleep(1)
            self.state = {}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--toolcalling", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--compare", type=str)
    args = parser.parse_args()

    if args.compare:
        # create conversational agent
        agent = Agent(toolcalling_functions=False, debug=args.debug)
        # call planner
        agent.direct(args.compare)
        # call tool funtions
        agent.set_toolcalling_functions(True)
        agent.direct(args.compare)
    else:
        # create conversational agent
        agent = Agent(toolcalling_functions=args.toolcalling, debug=args.debug)
        # run it
        agent.run()
