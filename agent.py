import sys
import logging
import os

from prompts import *

from parser import Dollarparser
from functioncalls import Agentfunctions
from software_environment import Software_environment
from groq import Groq
from dotenv import load_dotenv


class Agent():

    def __init__(self) -> None:
        load_dotenv()
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')

        # creating logger
        self.logger = logging.getLogger("tests")
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(logging.DEBUG)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(stdout)

        # initialize api and software env
        self.softwareenv = Software_environment()
        self.functionclass = Agentfunctions(self.softwareenv)
        self.parser = Dollarparser(self.functionclass)
        self.client = Groq(
            api_key=self.GROQ_API_KEY,
        )

    def call_llm_planner(self, model, user_query, execute_functions=True):
        chat_completion = self.client.chat.completions.create(
            messages=[
                # system prompt
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_PLANNER,

                },
                {
                    "role": "user",
                    "content": user_query,
                }
            ],
            model=model,
            temperature=0.0
        )
        llm_response = chat_completion.choices[0].message.content
        # printing response
        self.logger.debug("System prompt:\n%s", SYSTEM_PROMPT_PLANNER)
        self.logger.info("""
--------------------------\n
user input:
%s
--------------------------\n
output:
%s
\n--------------------------\n""", USER_QUERY_DEFAULT, llm_response)
        if execute_functions:
            # call parser and functions
            parsed_dollar_syntax = self.parser.analyze_and_execute_dollar_syntax(
                llm_response)
            self.logger.info("""
response: 
--------------------------\n      
%s
\n--------------------------\n""", parsed_dollar_syntax)
        return llm_response


if __name__ == "__main__":
    # check if user prompt was given
    user_query = USER_QUERY_DEFAULT
    if len(sys.argv) > 1:
        USER_QUERY_DEFAULT = sys.argv[1]

    # create conversational agent
    agent = Agent()
    # call the llm planner
    llm_response = agent.call_llm_planner(
        "llama3-70b-8192", user_query)
