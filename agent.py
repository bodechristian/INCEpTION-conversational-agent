import sys
import logging
import os

from prompts import *

from cerebras.cloud.sdk import Cerebras
from parser import Dollarparser
from functioncalls import Agentfunctions
from software_environment import Software_environment
from groq import Groq
from dotenv import load_dotenv


class Agent():

    # cerebras: llama3.1-70b, groq:llama3-70b-8192
    def __init__(self, model="llama3.1-70b", client="cerebras") -> None:
        load_dotenv()
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        self.CEREBRAS_API_KEY = os.getenv('CEREBRAS_API_KEY')

        # creating logger
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(logging.DEBUG)
        self.logger = logging.getLogger("output")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(stdout)
        l = logging.getLogger("functions")
        l.setLevel(logging.DEBUG)
        l.addHandler(stdout)

        # initialize api and software env
        self.model = model
        self.softwareenv = Software_environment()
        self.functionclass = Agentfunctions(self.softwareenv, self.call_llm)
        self.parser = Dollarparser(self.functionclass)

        if client == "groq":
            self.client = Groq(
                api_key=self.GROQ_API_KEY,
            )
        elif client == "cerebras":
            self.client = Cerebras(
                api_key=self.CEREBRAS_API_KEY,
            )

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
        llm_response = self.call_llm(SYSTEM_PROMPT_PLANNER, user_query)
        # printing response
        self.logger.debug("System prompt:\n%s", SYSTEM_PROMPT_PLANNER)
        self.logger.info(LOGGER_PLANNER_INPUT,
                         USER_QUERY_DEFAULT, llm_response)
        if execute_functions:
            # call parser and functions
            parsed_dollar_syntax = self.parser.analyze_and_execute_dollar_syntax(
                llm_response)
            self.logger.info(LOGGER_PLANNER_RESPONSE, parsed_dollar_syntax)
        return llm_response


if __name__ == "__main__":
    # check if user prompt was given
    user_query = USER_QUERY_DEFAULT
    if len(sys.argv) > 1:
        user_query = sys.argv[1]

    # create conversational agent
    agent = Agent()
    # call the llm planner
    llm_response = agent.call_llm_planner(user_query)
