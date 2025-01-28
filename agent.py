import sys
import logging
import json
import os
import time
import utils
import re

from prompts import *
from argparse import ArgumentParser
from cerebras.cloud.sdk import Cerebras
from parser import Dollarparser
from functioncalls import Agentfunctions
from mock_annotation_tool import MockAnnotationTool
from groq import Groq
from openai import OpenAI

from functioncalls_toolcalling import AgentfunctionsToolcalling

CLIENTMODELS = {
    "cerebras": "llama3.3-70b",
    "groq": "llama3-70b-8192",
    "ukp": "qwen2.5:32b",  # deepseek-r1:70b, llama3.2, phi4:latest
    "openai": "gpt-4o",
    "ollama": "llama3.2:latest",
    "deepseek": "deepseek-chat",
}


class Agent():
    # cerebras: llama3.3-70b, groq:llama3-70b-8192, llama3-groq-70b-versatile, ukp: llama3.2, openai: gpt-4o, gpt-4o-mini, ollama: mistral-latest
    def __init__(self, model="llama3.3-70b", client="cerebras", toolcalling_functions=False, testing=False, debug=False, no_logs=False) -> None:
        # save options
        self.state = {}
        self.toolcalling_functions = toolcalling_functions
        self.testing = testing
        self.debug = debug
        self.no_logs = no_logs
        self.model = model
        self.max_iterations = 10

        # initialize important numbers to keep track of
        self.nb_api_calls = 0
        self.nb_tokens_prompt = 0
        self.nb_tokens_completion = 0

        self.initialize_loggers(debug)

        # initialize api and software env
        self.softwareenv = MockAnnotationTool()

        self.set_toolcalling_functions(toolcalling_functions)
        self.set_client(client)
        # toolcalling method uses this for tool-calls, as non-finetuned models often return invalid reponses
        self.client_toolcalling = Groq(
            api_key=os.environ['GROQ_API_KEY'],
        )
        self.model_toolcalling = "llama3-groq-70b-8192-tool-use-preview"  # discontinued, potentially use llama-3.3-70b-versatile

    def initialize_loggers(self, debug):
        # creating logger
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(logging.DEBUG)

        self.logger = logging.getLogger("output")
        l = logging.getLogger("functions")

        # set levels
        if debug:
            self.logger.setLevel(logging.DEBUG)
            l.setLevel(logging.DEBUG)
        elif self.no_logs:
            self.logger.setLevel(logging.ERROR)
            l.setLevel(logging.ERROR)
        else:
            self.logger.setLevel(logging.INFO)
            l.setLevel(logging.INFO)

        self.logger.addHandler(stdout)
        l.addHandler(stdout)

    def set_client(self, client):
        if client == "groq":
            self.client = Groq(
                api_key=os.environ['GROQ_API_KEY'],
            )
        elif client == "cerebras":
            self.client = Cerebras(
                api_key=os.environ['CEREBRAS_API_KEY'],
            )
        elif client == "openai":
            self.client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY']
            )
        elif client == "ollama":
            self.client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama'
            )
        elif client == "ukp":
            self.client = OpenAI(
                base_url='http://10.167.31.201:11434/v1',
                api_key='ollama'
            )
        elif client == "deepseek":
            self.client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key=os.environ['OPENAI_API_KEY']
            )

    def set_toolcalling_functions(self, is_toolcalling):
        self.toolcalling_functions = is_toolcalling
        if is_toolcalling:
            self.functionclass = AgentfunctionsToolcalling(
                self.softwareenv, self.call_llm, self.get_state, testing=self.testing)
        else:
            self.functionclass = Agentfunctions(self.softwareenv, self.call_llm, testing=self.testing)
            self.parser = Dollarparser(self.functionclass)

    def call_llm(self, system_prompt, user_prompt):
        try:
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
                temperature=0.0,
            )
            # count api calls and tokens
            self.nb_api_calls += 1
            self.nb_tokens_prompt += chat_completion.usage.prompt_tokens
            self.nb_tokens_completion += chat_completion.usage.completion_tokens

            llm_response = chat_completion.choices[0].message.content
            if self.model.startswith('deepseek'):
                llm_response = utils.parse_deepseek_response(llm_response)
            return llm_response
        except Exception as error:
            self.logger.debug(error)
            return utils.ParsingException(message=f'calling LLM| {error}')

    def call_llm_planner(self, user_query, execute_functions=True):
        system_prompt_planner = get_system_prompt_planner(self.functionclass.valid_functions.values())
        llm_response = self.call_llm(system_prompt_planner, user_query)

        # printing response
        if isinstance(llm_response, utils.ParsingException):
            return llm_response.message
        self.logger.debug("System prompt:\n%s", system_prompt_planner)
        self.logger.info(LOGGER_PLANNER_INPUT, user_query, llm_response)
        if execute_functions:
            # call parser and functions
            parsed_dollar_syntax = self.parser.analyze_and_execute_dollar_syntax(llm_response)
            self.logger.info(LOGGER_PLANNER_RESPONSE, parsed_dollar_syntax)
        return llm_response

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
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                tools=TOOLCALLING_TOOLS,
                temperature=0.0,
                parallel_tool_calls=False,
            )
            # count api calls and tokens
            self.nb_api_calls += 1
            self.nb_tokens_prompt += chat_completion.usage.prompt_tokens
            self.nb_tokens_completion += chat_completion.usage.completion_tokens

            return_result = chat_completion.choices[0].message
        except Exception as error:
            self.logger.debug(error)
            return utils.ParsingException(message=f'calling LLM with toolcalling, initial call| {error}')

        self.logger.debug(chat_completion)
        if chat_completion.choices[0].finish_reason == "stop":
            # no tools need to be called, just respond
            self.logger.info(TOOLCALLING_OUTPUT, return_result.content)
            return messages
        else:
            # tools are called, keep calling them until llm says stop
            nb_iters = 1
            while chat_completion.choices[0].finish_reason != "stop" and nb_iters < self.max_iterations:
                # if tool calls exist
                if return_result.tool_calls:
                    # execute each of them (should only be one usually)
                    for tool_call in return_result.tool_calls:
                        self.logger.debug(f"CALLED: {tool_call}\n")
                        try:
                            # extract function and arguments
                            func = self.functionclass.valid_functions[tool_call.function.name]
                            arguments = json.loads(tool_call.function.arguments)
                            # execute function
                            response = func(**arguments)
                        except Exception as error:
                            self.logger.debug(error)
                            return utils.ParsingException(message=f'calling LLM with toolcalling, executing tool call| {error}')
                        # append functioncall and the response to LLM messages
                        messages.append(return_result)
                        messages.append({'role': 'tool', 'content': response,
                                        'tool_call_id': tool_call.id, 'name': tool_call.function.name})
                [self.logger.debug(m) for m in messages]
                # call LLM again with new appended messages
                try:
                    chat_completion = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model,
                        tools=TOOLCALLING_TOOLS,
                        temperature=0.0,
                        parallel_tool_calls=False,
                    )
                    # count api calls and tokens
                    self.nb_api_calls += 1
                    self.nb_tokens_prompt += chat_completion.usage.prompt_tokens
                    self.nb_tokens_completion += chat_completion.usage.completion_tokens

                    return_result = chat_completion.choices[0].message
                    nb_iters += 1
                except Exception as error:
                    self.logger.debug(error)
                    return utils.ParsingException(message=f'calling LLM with toolcalling, awaiting next step| {error}')
            # this response considers what was done and the state
            # therefore it should be better than just the normal llm content response
            final_response = self.functionclass.respond()
            messages.append({"role": "assistant",  "content": final_response})
            # self.logger.info(return_result.content)
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

    def get_state(self):
        return self.state


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--toolcalling", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--compare", type=str)
    parser.add_argument("--client", type=str)
    args = parser.parse_args()

    if args.client in CLIENTMODELS:
        model = CLIENTMODELS[args.client]
        client = args.client
    else:
        # else set to default model
        client = "cerebras"
        model = "llama3.1-70b"

    if args.compare:
        # create conversational agent
        agent = Agent(toolcalling_functions=True, client=client, model=model, debug=args.debug)
        # call planner
        agent.direct(args.compare)
        agent.logger.info(f"API calls: {agent.nb_api_calls},")  # total amount of tokens: {agent.nb_tokens}")
        agent.nb_api_calls = 0
        # agent.nb_tokens = 0
        # call tool funtions
        agent.set_toolcalling_functions(False)
        agent.direct(args.compare)
        agent.logger.info(f"API calls: {agent.nb_api_calls},")  # total amount of tokens: {agent.nb_tokens}")
    else:
        # create conversational agent
        agent = Agent(toolcalling_functions=args.toolcalling, client=client, model=model, debug=args.debug)
        # run it
        agent.run()
