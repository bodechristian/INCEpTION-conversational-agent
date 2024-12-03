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
import utils


class EvaluateActObserve():
    def __init__(self, models=[], filenames=[]) -> None:

        # the models to evaluate. List of pairs of ('client', 'model')
        self.models = models
        # read and store yaml test files
        self.filenames = filenames
        self.test_input_files = {}
        for _filename in self.filenames:
            with open(join(getcwd(), "testfiles", _filename)) as f:
                try:
                    yml = yaml.safe_load(f)
                    self.test_input_files[_filename] = yml
                except yaml.YAMLError as exc:
                    print(exc)

        # set up logging evaluation results to file
        self.all_logged_data = []
        # each k:v pair is a file+model run, then appended to all_logged_data
        self.logged_data_per_run = {}

        # set up loggers
        self.logger = logging.getLogger("tests")
        stdout = logging.StreamHandler(stream=sys.stdout)
        stdout.setLevel(logging.DEBUG)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(stdout)

    def setup_agent(self, client, model):
        # clear loggers from agent to avoid double logging
        l = logging.getLogger("output")
        l.handlers.clear()
        l = logging.getLogger("functions")
        l.handlers.clear()
        # create new agent
        self.agent = Agent(client=client, model=model, toolcalling_functions=True, testing=True)

    def evaluate(self):
        for file in self.filenames:
            for client, model in self.models:
                self.setup_agent(client, model)
                # logging to file
                self.logged_data_per_run = {
                    "method": "Act and Observe",
                    "file": file,
                    "client": client,
                    "model": model,
                    "tests": []
                }

                self._eval_correctness_functions(file)
                # self._eval_scope(file)
                # self._eval_layer_and_feature(file)

                self.all_logged_data.append(self.logged_data_per_run)
        # Convert and write JSON object to file
        with open(join("test_logs", f"{time.strftime("%Y%m%d-%H%M%S")}.json"), "w") as outfile:
            json.dump(self.all_logged_data, outfile)

    def _eval_correctness_functions(self, file):
        self.logger.info("\nTesting correct functions on %s", file)
        start_time = time.time()
        correct_results = 0
        runs = []
        for i, testcase in enumerate(self.test_input_files[file]["testcases"]):
            # extract columns from yaml
            prompt = testcase["prompt"]
            expected_results = set(testcase["expectations"])

            # prompt planner
            start_time_testcase = time.time()
            detected_messages = self.agent.call_llm_toolcalling(user_query=prompt)
            time_testcase = time.time() - start_time_testcase

            # extract only the functions from the act/observe response
            detected = set(utils.get_toolcalls_from_messages(detected_messages))
            detected.add('respond')

            # see if correct functions were called (order irrelevant)
            correct_result = detected == expected_results
            correct_results += correct_result
            runs.append({
                "prompt": prompt,
                "expected": ", ".join(sorted(list(expected_results), key=str.lower)),
                "predicted": ", ".join(sorted(list(detected), key=str.lower)),
                "result": correct_result,
                "duration": time_testcase,
            })

            # logging
            self.logger.debug("\nAnalyzing prompt: %s", prompt)
            self.logger.debug("Expected functioncalls: %s", expected_results)
            self.logger.debug("Detected functioncalls: %s", detected)
            self.logger.debug("Correct Result?: %s", correct_result)
        self.logger.info(
            "\n%d/%d functions were correctly called from the planner.", correct_results, i+1)
        self.logged_data_per_run["tests"].append({
            "test_name": "called_functions",
            "correct": correct_results,
            "amount": i+1,
            "runs": runs,
            "duration": time.time() - start_time,
        })

    def _eval_scope(self, file):
        self.logger.info(
            "\nTesting scope detection on %s", file)
        start_time = time.time()
        correct_results = 0
        runs = []
        for i, testcase in list(enumerate(self.test_input_files[file]["testcases"])):
            # extract columns from yaml
            prompt = testcase["prompt"]
            scope = testcase["scope"]

            # get prediction
            start_time_testcase = time.time()
            pred_scope = self.agent.functionclass.get_scope(prompt)
            time_testcase = time.time() - start_time_testcase

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
                "duration": time_testcase,
            })
        self.logger.info(
            "\n%d/%d scopes were correctly predicted.", correct_results, i+1)
        self.logged_data_per_run["tests"].append({
            "test_name": "scope",
            "correct": correct_results,
            "amount": i+1,
            "runs": runs,
            "duration": time.time() - start_time,
        })

    def _eval_layer_and_feature(self, file):
        self.logger.info(
            "\nTesting layer and feature detection on %s", file)
        start_time = time.time()
        correct_results = 0
        runs = []
        skipped_tests = 0
        for i, testcase in list(enumerate(self.test_input_files[file]["testcases"])):
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
            start_time_testcase = time.time()
            pred_layer, pred_feature = self.agent.functionclass.get_layer_and_feature(
                original_user_query=prompt)
            time_testcase = time.time() - start_time_testcase
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
                "duration": time_testcase,
            })
        self.logger.info(
            "\n%d/%d layers and features were correctly predicted.", correct_results, i+1-skipped_tests)
        self.logged_data_per_run["tests"].append({
            "test_name": "layers and features",
            "correct": correct_results,
            "amount": i+1-skipped_tests,
            "runs": runs,
            "duration": time.time() - start_time
        })


if __name__ == "__main__":
    models = [("groq", "llama3-70b-8192"), ("cerebras", "llama3.1-70b")]
    filenames = ["test_expectations_Cheetah.yaml", "test_expectations_Politician.yaml"]

    EvaluateActObserve(models=models[1:2], filenames=filenames[:1]).evaluate()
