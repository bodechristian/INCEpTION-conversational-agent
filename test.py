import logging.config
import unittest
import os
import logging
import sys
import yaml

from os import getcwd
from os.path import join
from agent import Agent
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')


class TestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestParser, cls).setUpClass()

        # read and save yaml test files
        files = ["test_expectations_Cheetah.yaml",
                 "test_expectations_Politician.yaml"]
        cls.data = {}
        for f in files:
            with open(join(getcwd(), "testfiles", f)) as stream:
                try:
                    yml = yaml.safe_load(stream)
                    cls.data[f] = yml
                except yaml.YAMLError as exc:
                    print(exc)

        # set up Groq API client
        cls.client = Groq(
            api_key=GROQ_API_KEY,
        )
        cls.agent = Agent()

        # set up logger
        cls.logger = logging.getLogger("tests")
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(logging.DEBUG)
        cls.logger.setLevel(logging.DEBUG)
        cls.logger.addHandler(stdout)

    def test_expected_functions(self):
        # the models and files to test the planner on
        models = ["llama3-70b-8192"]
        files = ["test_expectations_Cheetah.yaml",
                 "test_expectations_Politician.yaml"]

        for file in files[:1]:
            for model in models:
                # self.helper_planner(file, model)
                self.helper_scope(file, model)

    def helper_planner(self, file, model):
        self.logger.info(
            "\nTesting correct planning on %s with model %s", file, model)
        correct_results = 0
        for i, testcase in enumerate(self.data[file]["testcases"]):
            # extract columns from yaml
            prompt = testcase["prompt"]
            expected_results = set(testcase["expectations"])

            # prompt planner
            llm_response = self.agent.call_llm_planner(
                prompt, execute_functions=False)

            # extract only the functions from the planner response
            detected = set([func for _, func,
                            _ in self.agent.parser.analyze_functions(llm_response)])

            # see if correct functions were called (order irrelevant)
            correct_result = detected == expected_results
            correct_results += correct_result

            # logging
            self.logger.debug("\nAnalyzing prompt: %s", prompt)
            self.logger.debug("Expected functioncalls: %s", expected_results)
            self.logger.debug("Detected functioncalls: %s", detected)
            self.logger.debug("Correct Result?: %s", correct_result)
        self.logger.info(
            "\n%d/%d functions were correctly called from the planner.", correct_results, i+1)

    def helper_scope(self, file, model):
        self.logger.info(
            "\nTesting scope detection on %s with model %s", file, model)
        correct_results = 0
        for i, testcase in list(enumerate(self.data[file]["testcases"]))[:4]:
            # extract columns from yaml
            prompt = testcase["prompt"]
            scope = testcase["scope"]

            # get prediction
            pred_scope = self.agent.functionclass.get_scope(prompt)

            # check prediction to expectation
            correct_results += scope == pred_scope

            # logging
            self.logger.debug("\nAnalyzing prompt: %s", prompt)
            self.logger.debug("Expected scope: %s", scope)
            self.logger.debug("Detected scope: %s", pred_scope)
            self.logger.debug("Correct Result?: %s", scope == pred_scope)
        self.logger.info(
            "\n%d/%d scopes were correctly predicted.", correct_results, i+1)


if __name__ == "__main__":
    unittest.main()
