import logging.config
import unittest
import csv
import os
import logging
import sys

from os import getcwd
from os.path import join
from agent import Agent
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

logger = logging.getLogger("tests")
stdout = logging.StreamHandler(stream=sys.stdout)
stdout.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
logger.addHandler(stdout)


class TestParser(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def setUp(self):
        # load groq
        self.client = Groq(
            api_key=GROQ_API_KEY,
        )
        self.agent = Agent()

    def test_expected_functions(self):
        # the models and files to test the planner on
        # Groq website says 30 requests per minute is the ratelimit
        # but it seems to be 10????
        models = ["llama3-70b-8192"]
        files = ["test_expectations_Cheetah.csv",
                 "test_expectations_Politician.csv"]

        for file in files[:1]:
            for model in models:
                self.helper_planner(file, model)
                # self.helper_scope(file, model)

    def helper_planner(self, file, model):
        logger.info("testing %s with %s", file, model)
        # open file
        with open(join(getcwd(), "testfiles", file), newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            correct_results = 0
            for i, row in enumerate(reader):
                # extract columns from csv
                prompt, expected_results, scope, layer, feature, label, criteria = row
                expected_results = set(expected_results.split(","))
                # prompt planner
                llm_response = self.agent.call_llm_planner(
                    model, prompt, execute_functions=False)
                # extract only the functions from the planner response
                detected = set([func for _, func,
                                _ in self.agent.parser.analyze_functions(llm_response)])
                # see if correct functions were called (order irrelevant)
                correct_result = detected == expected_results
                correct_results += correct_result
                # logging
                logger.debug("Analyzing prompt: %s", prompt)
                logger.debug("Expected functioncalls: %s", expected_results)
                logger.debug("Detected functioncalls: %s", detected)
                logger.debug("Correct Result?: %s\n", correct_result)
        logger.info(
            "%d/%d functions were correctly called from the planner.", correct_results, i+1)

    def helper_scope(self, file, model):
        with open(join(getcwd(), "testfiles", file), newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            correct_results = 0
            for i, row in enumerate(reader):
                # extract columns from csv
                prompt, expected_results, scope, layer, feature, label, criteria = row
                pred_scope = self.agent.functionclass.get_scope(prompt)
                correct_results += scope == pred_scope
                logger.debug("Analyzing prompt: %s", prompt)
                logger.debug("Expected scope: %s", scope)
                logger.debug("Detected scope: %s", pred_scope)
                logger.debug("Correct Result?: %s\n", scope == pred_scope)

        logger.info(
            "%d/%d scopes were correctly predicted.", correct_results, i+1)
