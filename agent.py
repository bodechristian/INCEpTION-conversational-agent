import sys
import logging
import json
import os
import time
from functioncalls_toolcalling_parampred import AgentfunctionsToolcallingParampred
from functioncalls_toolcalling_parampred_compound import AgentfunctionsToolcallingParampredCompound
import utils
import re
import uuid

from ollama import Client as OllamaClient
from prompts import *
from pydantic import BaseModel, ConfigDict
from argparse import ArgumentParser
from cerebras.cloud.sdk import Cerebras
from parser import Dollarparser, HuggingGPTparser
from functioncalls_planner import Agentfunctions
from mock_annotation_tool import MockAnnotationTool
from groq import Groq
from openai import OpenAI
from functioncalls_toolcalling import AgentfunctionsToolcalling
from functioncalls_classify import AgentfunctionsClassify
from openai.lib._parsing._completions import type_to_response_format_param


CLIENTMODELS = {
    "cerebras": "llama3.1-8b",  # llama3.1-8b, llama3.3-70b
    "groq": "llama3-70b-8192",  # llama3-70b-8192, gemma2-9b-it
    # deepseek-r1:70b, llama3.2, phi4:latest, llama3.3, command-r7b, dolphin3:latest, qwen2.5:32b, llama3.2:3b-instruct-q4_K_M
    "ukp": "qwen2.5:14b",
    "openai": "gpt-4o",
    "ollama": "llama3.2:latest",
    "deepseek": "deepseek-chat",
}


class SequentialWithoutToolcalling(BaseModel):
    """class for structued output, support only by openAI. Thus currently unused
    self.client.beta.chat.completions.parse(response_format=SequentialWithoutToolcalling)"""
    functionname: str
    parameters: dict[str, str]
    stop: bool
    content: str


class PlannerHuggingGPTFunction(BaseModel):
    """class for structured output for HuggingGPTs json syntax"""
    model_config = ConfigDict(extra="forbid")
    task: str
    id: int
    dep: list[int]
    args: dict


class PlannerHuggingGPTJSON(BaseModel):
    """class for structured output for HuggingGPTs json syntax"""
    model_config = ConfigDict(extra="forbid")
    plan: list[PlannerHuggingGPTFunction]


class Agent():
    # cerebras: llama3.3-70b, groq:llama3-70b-8192, llama3-groq-70b-versatile, ukp: llama3.2, openai: gpt-4o, gpt-4o-mini, ollama: mistral-latest
    def __init__(self, mode="planner", model="qwen2.5:32b", client="ukp", toolcalling_functions=False, huggingGPT=False, param_prediction_by_planner=False, add_thoughts=False,
                 testing=False, debug=False, no_logs=False, test_classify=False) -> None:
        # save options
        self.mode = mode  # 'planner', 'sequential'
        self.state = {}
        self.toolcalling_functions = toolcalling_functions
        self.param_prediction_by_planner = param_prediction_by_planner
        self.add_thoughts = add_thoughts
        self.testing = testing
        self.debug = debug
        self.no_logs = no_logs
        self.model = model
        self.max_iterations = 10  # for sequential reasoning

        self.huggingGPT = huggingGPT

        # initialize important numbers to keep track of
        self.nb_api_calls = 0
        self.nb_tokens_prompt = 0
        self.nb_tokens_completion = 0

        self.initialize_loggers(debug)

        # initialize api and software env
        self.softwareenv = MockAnnotationTool()

        self.set_client(client)
        self.set_functionclass(mode=mode, toolcalling=toolcalling_functions)
        if test_classify:
            self.functionclass_classify = AgentfunctionsClassify(
                self.softwareenv, self.call_llm, self.call_llm_with_format, self.get_state, is_ollama=self.is_ollama)

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
        self.client_name = client
        self.is_ollama = (self.client_name == "ukp" or self.client_name == "ollama")

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
            self.client = OllamaClient(
                host='http://10.167.31.201:11434',
            )
            # self.client = OpenAI(
            #     base_url='http://10.167.31.201:11434/v1',
            #     api_key='ollama'
            # )
        elif client == "deepseek":
            self.client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key=os.environ['OPENAI_API_KEY']
            )

    def set_functionclass(self, mode, toolcalling):
        self.mode = mode
        self.toolcalling_functions = toolcalling
        if self.mode == 'sequential':
            if self.toolcalling_functions:
                # sequential with tools=
                self.functionclass = AgentfunctionsToolcalling(
                    self.softwareenv, self.call_llm, self.get_state, testing=self.testing)
            else:
                # sequential without tools=
                self.functionclass = AgentfunctionsToolcalling(
                    self.softwareenv, self.call_llm, self.get_state, testing=self.testing)
        else:
            # mode == 'planner'
            if self.param_prediction_by_planner:
                self.functionclass = AgentfunctionsToolcallingParampred(
                    self.softwareenv, self.call_llm, self.get_state, testing=self.testing)
            else:
                self.functionclass = Agentfunctions(self.softwareenv, self.call_llm,
                                                    self.get_state, testing=self.testing)
            if self.huggingGPT:
                self.parser = HuggingGPTparser(self.functionclass)
            else:
                self.parser = Dollarparser(self.functionclass)

    def call_llm(self, system_prompt, user_prompt):
        try:
            if self.is_ollama:
                response_completion = self.client.chat(
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
                    options={'n_ctx ': 8192, "temperature": 0.0},
                )
                self.nb_api_calls += 1
                self.nb_tokens_prompt += response_completion.prompt_eval_count
                self.nb_tokens_completion += response_completion.eval_count
                print(f"tokens: {response_completion.prompt_eval_count} + {response_completion.eval_count}")
                if response_completion.prompt_eval_count == 2048:
                    print(f"\n\n\n\n\n\n\n\n\n\n\n\nMAYDAY MAYDAY MADAY\n\n\n\n\n\n\n\n\n\n\n\n\n")

                return_result = response_completion.message.content
                if self.model.startswith('deepseek'):
                    return_result = utils.parse_deepseek_response(return_result)
                return return_result
            else:
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
                    max_tokens=80000,
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

    def call_llm_with_format(self, system_prompt, user_prompt, format_class: BaseModel):
        """only works with Ollama. Uses json format"""
        try:
            if self.is_ollama and self.huggingGPT:
                response_completion = self.client.chat(
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
                    options={"temperature": 0.0, "num_ctx": 32000},
                    format=format_class.model_json_schema(),
                )
                self.nb_api_calls += 1
                self.nb_tokens_prompt += response_completion.prompt_eval_count
                self.nb_tokens_completion += response_completion.eval_count
                print(f"tokens: {response_completion.prompt_eval_count} + {response_completion.eval_count}")
                if response_completion.prompt_eval_count == 2048:
                    print(f"\n\n\n\n\n\n\n\n\n\n\n\nMAYDAY MAYDAY MADAY\n\n\n\n\n\n\n\n\n\n\n\n\n")

                return_result_content = format_class.model_validate_json(
                    response_completion.message.content)
                return return_result_content
        except Exception as error:
            self.logger.debug(error)
            return utils.ParsingException(message=f'calling LLM| {error}')

    def call_llm_planner(self, user_query, execute_functions=True):
        if self.huggingGPT:
            # huggingGPT syntax using strict json
            if self.is_ollama:
                # huggingGPT syntax for ollama
                system_prompt_planner = get_system_prompt_planner_hugginggpt(
                    self.functionclass.valid_functions.values())
                plan = self.call_llm_with_format(system_prompt_planner, user_query,
                                                 PlannerHuggingGPTJSON)
            elif self.client_name == "openai":
                # huggingGPT syntax for non-ollama, e.g. self.client.beta.chat.completions.parse(response_format=...)
                system_prompt_planner = get_system_prompt_planner_hugginggpt(
                    self.functionclass.valid_functions.values())
                try:
                    response_completion = self.client.chat.completions.create(
                        messages=[
                            # system prompt
                            {
                                "role": "system",
                                "content": system_prompt_planner,

                            },
                            {
                                "role": "user",
                                "content": user_query,
                            }
                        ],
                        model=self.model,
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    self.nb_api_calls += 1
                    self.nb_tokens_prompt += response_completion.usage.prompt_tokens
                    self.nb_tokens_completion += response_completion.usage.completion_tokens

                    llm_response = response_completion.choices[0].message.content
                    return_result_content = PlannerHuggingGPTJSON.model_validate_json(llm_response)

                    return return_result_content
                except Exception as error:
                    self.logger.debug(error)
                    return utils.ParsingException(message=f'calling LLM| {error}')
        else:
            system_prompt_planner = get_system_prompt_planner(self.functionclass.valid_functions.values())
            plan = self.call_llm(system_prompt_planner, user_query)

        # printing response
        if isinstance(plan, utils.ParsingException):
            return plan.message
        self.logger.debug("System prompt:\n%s", system_prompt_planner)
        self.logger.info(LOGGER_PLANNER_INPUT, user_query, plan)
        if execute_functions:
            # call parser and functions
            parsed_syntax = self.parser.analyze_and_execute_dollar_syntax(plan)
            self.logger.info(LOGGER_PLANNER_RESPONSE, parsed_syntax)
        return plan

    def call_llm_planner_parampred(self, user_query, execute_functions=True):
        self.state['user_query'] = user_query
        system_prompt_planner = get_system_prompt_planner_parampred(
            self.functionclass.valid_functions.values(), self.softwareenv.get_layers_and_features())
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
            if self.is_ollama:
                response_completion = self.client.chat(
                    messages=messages,
                    model=self.model,
                    tools=TOOLCALLING_TOOLS,
                    options={'num_ctx ': 16, "temperature": 0.0},
                )
                self.nb_api_calls += 1
                self.nb_tokens_prompt += response_completion.prompt_eval_count
                self.nb_tokens_completion += response_completion.eval_count
                print(f"tokens: {response_completion.prompt_eval_count} + {response_completion.eval_count}")
                if response_completion.prompt_eval_count == 2048:
                    print(f"\n\n\n\n\n\n\n\n\n\n\n\nMAYDAY MAYDAY MADAY\n\n\n\n\n\n\n\n\n\n\n\n\n")

                return_result = response_completion.message
            else:
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

        if self.is_ollama:
            self.logger.debug(response_completion)
        else:
            self.logger.debug(chat_completion)
        if (self.is_ollama and return_result.tool_calls == None) or (not self.is_ollama and chat_completion.choices[0].finish_reason == "stop"):
            # no tools need to be called, just respond
            self.state['info'] = return_result.content
            final_response = self.functionclass.respond()
            messages.append({"role": "assistant",  "content": final_response})
            return messages
        else:
            # tools are called, keep calling them until llm says stop
            nb_iters = 1
            while (((self.is_ollama and return_result.tool_calls != None) or
                    (not self.is_ollama and chat_completion.choices[0].finish_reason != "stop")) and
                   nb_iters < self.max_iterations):
                # if tool calls exist
                if return_result.tool_calls:
                    # execute each of them (should only be one usually)
                    for tool_call in return_result.tool_calls:
                        self.logger.debug(f"CALLED: {tool_call}\n")
                        try:
                            # extract function and arguments
                            func = self.functionclass.valid_functions[tool_call.function.name]
                            if self.is_ollama:
                                arguments = tool_call.function.arguments
                            else:
                                arguments = json.loads(tool_call.function.arguments)
                            # execute function
                            response = func(**arguments)
                        except Exception as error:
                            self.logger.debug(error)
                            return utils.ParsingException(message=f'calling LLM with toolcalling, executing tool call| {error}')
                        # append functioncall and the response to LLM messages
                        messages.append(return_result)
                        _id = response_completion.created_at if self.is_ollama else tool_call.id
                        messages.append({'role': 'tool', 'content': response,
                                        'tool_call_id': _id, 'name': tool_call.function.name})
                [self.logger.debug(m) for m in messages]
                # create Thought
                if self.add_thoughts:
                    try:
                        if self.is_ollama:
                            response_completion = self.client.chat(
                                messages=[{"role": "system", "content": get_system_prompt_toolcalling_thought_ollama(
                                    self.functionclass.valid_functions.values())}, *messages[1:]],
                                model=self.model,
                                options={'num_ctx ': 16, "temperature": 0.0},
                            )
                            self.nb_api_calls += 1
                            self.nb_tokens_prompt += response_completion.prompt_eval_count
                            self.nb_tokens_completion += response_completion.eval_count

                            return_result = response_completion.message
                        else:
                            chat_completion = self.client.chat.completions.create(
                                messages=[{"role": "system", "content": get_system_prompt_toolcalling_thought_ollama(
                                    self.functionclass.valid_functions.values())}, *messages[1:]],
                                model=self.model,
                                temperature=0.0,
                                parallel_tool_calls=False,
                                tool_choice='none',
                            )
                            # count api calls and tokens
                            self.nb_api_calls += 1
                            self.nb_tokens_prompt += chat_completion.usage.prompt_tokens
                            self.nb_tokens_completion += chat_completion.usage.completion_tokens

                            return_result = chat_completion.choices[0].message
                        nb_iters += 1
                        messages.append({'role': 'assistant', 'content': f"Thought: {return_result.content}"})
                    except Exception as error:
                        self.logger.debug(error)
                        return utils.ParsingException(message=f'calling LLM with toolcalling, awaiting next step| {error}')
                # call LLM again with new appended messages
                try:
                    if self.is_ollama:
                        response_completion = self.client.chat(
                            messages=messages,
                            model=self.model,
                            tools=TOOLCALLING_TOOLS,
                            options={'num_ctx ': 16, "temperature": 0.0},
                        )
                        self.nb_api_calls += 1
                        self.nb_tokens_prompt += response_completion.prompt_eval_count
                        self.nb_tokens_completion += response_completion.eval_count
                        print(f"tokens: {response_completion.prompt_eval_count} + {response_completion.eval_count}")
                        if response_completion.prompt_eval_count == 2048:
                            print(f"\n\n\n\n\n\n\n\n\n\n\n\nMAYDAY MAYDAY MADAY\n\n\n\n\n\n\n\n\n\n\n\n\n")

                        return_result = response_completion.message
                    else:
                        chat_completion = self.client.chat.completions.create(
                            messages=messages,
                            model=self.model,
                            tools=TOOLCALLING_TOOLS,
                            temperature=0.0,
                            parallel_tool_calls=False,
                            tool_choice='auto',
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
            # (the normal response would do so to, because it sees the messages)
            # (so maybe add here if finish_reason==stop: respond with content, and also maybe change get_toolcalls_from_messages())
            # (this can stay if it does >maxiterations iterations)
            # therefore it should be better than just the normal llm content response
            final_response = self.functionclass.respond()
            messages.append({"role": "assistant",  "content": final_response})
            # self.logger.info(return_result.content)
            self.logger.debug(f"Functions that were called: {utils.get_toolcalls_from_messages(messages)}")
            return messages

    def parse_json_sequential_no_tools(self, _str):
        """turns string to json. Json should look like:
        {{
            function_call: "xxx",
            parameters: {{"xxx": "yyy", "aaa":"bbb"}},
            stop: true/false,
            content: ""
        }}"""
        s = json.loads(_str)
        if "stop" not in s:
            s['stop'] = False
        s['tool_call_id'] = str(uuid.uuid4())
        return s

    def call_llm_sequential_no_tools(self, user_query):
        self.state['user_query'] = user_query
        self.logger.info(TOOLCALLING_INPUT, user_query)
        messages = [
            # system prompt
            {
                "role": "system",
                "content": system_prompt_sequential_no_tools(self.functionclass.valid_functions.values()),

            },
            {
                "role": "user",
                "content": user_query,
            }
        ]
        [self.logger.debug(m) for m in messages]
        try:
            chat_completion = self.client.chat(
                messages=messages,
                model=self.model,
                options={"temperature": 0.0, "num_ctx": 32000},
            )
            # count api calls and tokens
            self.nb_api_calls += 1
            self.nb_tokens_prompt += chat_completion.prompt_eval_count
            self.nb_tokens_completion += chat_completion.eval_count
            print(f"tokens: {chat_completion.prompt_eval_count} + {chat_completion.eval_count}")

            return_result = chat_completion.message
            return_json = self.parse_json_sequential_no_tools(return_result.content)
        except Exception as error:
            self.logger.debug(error)
            return utils.ParsingException(message=f'calling sequential LLM without toolcalling, initial call| {error}')

        self.logger.debug(chat_completion)
        if return_json['stop']:
            # no tools need to be called, just respond
            self.logger.info(TOOLCALLING_OUTPUT, return_json['content'])
            return messages
        else:
            # tools are called, keep calling them until llm says stop
            nb_iters = 1
            while not return_json['stop'] and nb_iters < self.max_iterations:
                self.logger.debug(f"CALLED: {return_json['function_call']}\n")
                try:
                    # extract function and arguments
                    func = self.functionclass.valid_functions[return_json['function_call']]
                    # execute function
                    response = func(**return_json['parameters'])
                except Exception as error:
                    self.logger.debug(error)
                    return utils.ParsingException(message=f'calling sequential LLM without toolcalling, executing tool call| {error}')
                # append functioncall and the response to LLM messages
                messages.append({'role': 'tool', 'content': response,
                                'name': return_json['function_call'], 'tool_call_id': return_json['tool_call_id']})
                [self.logger.debug(m) for m in messages]
                # call LLM again with new appended messages
                try:
                    chat_completion = self.client.chat(
                        messages=messages,
                        model=self.model,
                        options={"temperature": 0.0, "num_ctx": 32000},
                    )
                    # count api calls and tokens
                    self.nb_api_calls += 1
                    self.nb_tokens_prompt += chat_completion.prompt_eval_count
                    self.nb_tokens_completion += chat_completion.eval_count
                    print(f"tokens: {chat_completion.prompt_eval_count} + {chat_completion.eval_count}")

                    return_result = chat_completion.message
                    return_json = self.parse_json_sequential_no_tools(return_result.content)
                    nb_iters += 1
                except Exception as error:
                    self.logger.debug(error)
                    return utils.ParsingException(message=f'calling sequential LLM without toolcalling, awaiting next step | {error}')
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
        # differentiate between sequential method and planner method
        if self.mode == 'sequential':
            if self.toolcalling_functions:
                # call the toolcalling method
                func_call = self.call_llm_toolcalling
            else:
                func_call = self.call_llm_sequential_no_tools
        else:
            # call the llm planner method
            if self.param_prediction_by_planner:
                func_call = self.call_llm_planner_parampred
            else:
                func_call = self.call_llm_planner

        # start conversation
        while True:
            # wait for user input
            user_prompt = input("Enter prompt: ")
            func_call(user_prompt)
            time.sleep(1)
            self.clear_state()

    def get_state(self):
        return self.state

    def clear_state(self):
        self.state = {}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--toolcalling", action='store_true')
    parser.add_argument("--parampred", action='store_true')
    parser.add_argument("--thoughts", action='store_true')
    parser.add_argument("--mode", type=str, nargs='?', default='planner')  # sequential, planner
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
        agent.set_functionclass(mode='sequential', toolcalling=False)
        agent.direct(args.compare)
        agent.logger.info(f"API calls: {agent.nb_api_calls},")  # total amount of tokens: {agent.nb_tokens}")
    else:
        # create conversational agent
        agent = Agent(mode=args.mode, toolcalling_functions=args.toolcalling, param_prediction_by_planner=args.parampred,
                      client=client, model=model, debug=args.debug, add_thoughts=args.thoughts)
        # run it
        agent.run()
