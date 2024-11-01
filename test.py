import logging.config
import unittest
import os
import logging
import sys
import yaml
import json
import time

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

        cls.agent = Agent()

        # set up logger
        cls.logger = logging.getLogger("tests")
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(logging.DEBUG)
        cls.logger.setLevel(logging.DEBUG)
        cls.logger.addHandler(stdout)

        # set up logging to file
        cls.logged_data = {}

    def test_expected_functions(self):
        # the models and files to test the planner on
        models = ["llama3-70b-8192"]
        files = ["test_expectations_Cheetah.yaml",
                 "test_expectations_Politician.yaml"]

        all_logged_data = []
        for file in files:
            for model in models:
                # logging to file
                self.logged_data = {
                    "file": file,
                    "model": model,
                    "tests": []
                }

                self.helper_planner(file, model)
                self.helper_scope(file, model)
                self.helper_layer_and_feature(file, model)
                all_logged_data.append(self.logged_data)
        # Convert and write JSON object to file
        with open(join("test_logs", f"{time.strftime("%Y%m%d-%H%M%S")}.json"), "w") as outfile:
            json.dump(all_logged_data, outfile)

    def helper_planner(self, file, model):
        self.logger.info(
            "\nTesting correct planning on %s with model %s", file, model)
        correct_results = 0
        runs = []
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
            runs.append({
                "prompt": prompt,
                "expected": ", ".join(sorted(list(expected_results), key=str.lower)),
                "predicted": ", ".join(sorted(list(detected), key=str.lower)),
                "result": correct_result,
            })

            # logging
            self.logger.debug("\nAnalyzing prompt: %s", prompt)
            self.logger.debug("Expected functioncalls: %s", expected_results)
            self.logger.debug("Detected functioncalls: %s", detected)
            self.logger.debug("Correct Result?: %s", correct_result)
        self.logger.info(
            "\n%d/%d functions were correctly called from the planner.", correct_results, i+1)
        self.logged_data["tests"].append({
            "test_name": "planner",
            "correct": correct_results,
            "amount": i+1,
            "runs": runs,
        })

    def helper_scope(self, file, model):
        self.logger.info(
            "\nTesting scope detection on %s with model %s", file, model)
        correct_results = 0
        runs = []
        for i, testcase in list(enumerate(self.data[file]["testcases"])):
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

            runs.append({
                "prompt": prompt,
                "expected": scope,
                "predicted": pred_scope,
                "result": scope == pred_scope,
            })
        self.logger.info(
            "\n%d/%d scopes were correctly predicted.", correct_results, i+1)
        self.logged_data["tests"].append({
            "test_name": "scope",
            "correct": correct_results,
            "amount": i+1,
            "runs": runs,
        })

    def helper_layer_and_feature(self, file, model):
        self.logger.info(
            "\nTesting layer and feature detection on %s with model %s", file, model)
        correct_results = 0
        runs = []
        skipped_tests = 0
        for i, testcase in list(enumerate(self.data[file]["testcases"])):
            # extract columns from yaml
            prompt = testcase["prompt"]
            layer = testcase["layer"]
            feature = testcase["feature"]

            # if layer empty, that meanns that testcase does not use a layer
            # so skip testcase
            if layer is None or feature is None:
                skipped_tests += 1
                continue

            # get prediction
            pred_layer, pred_feature = self.agent.functionclass.get_layer_and_feature(
                original_user_query=prompt)
            # remove prefix from layer name
            pred_layer = pred_layer.split(".")[-1]

            # check prediction to expectation
            correct_result = (layer == pred_layer and feature == pred_feature)
            correct_results += correct_result

            # logging
            self.logger.debug("\nAnalyzing prompt: %s", prompt)
            self.logger.debug(
                "Expected layer and feature: %s, %s", layer, feature)
            self.logger.debug(
                "Detected layer and feature: %s, %s", pred_layer, pred_feature)
            self.logger.debug("Correct Result?: %s", correct_result)

            runs.append({
                "prompt": prompt,
                "expected": ", ".join((layer, feature)),
                "predicted": ", ".join((pred_layer, pred_feature)),
                "result": correct_result,
            })
        self.logger.info(
            "\n%d/%d layers and features were correctly predicted.", correct_results, i+1-skipped_tests)
        self.logged_data["tests"].append({
            "test_name": "layers and features",
            "correct": correct_results,
            "amount": i+1-skipped_tests,
            "runs": runs,
        })


if __name__ == "__main__":
    unittest.main()
