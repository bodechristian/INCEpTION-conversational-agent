import logging.config
import os
import logging
import sys
import yaml
import json
import utils
import time

from os import getcwd
from os.path import join
from agent import Agent


class EvaluatePlanner():
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
        self.agent = Agent(client=client, model=model)

    def evaluate(self):
        for file in self.filenames:
            for client, model in self.models:
                self.setup_agent(client, model)
                # switch open document to corresponding testfile
                self.agent.softwareenv.set_current_document_by_name(utils.remove_file_ending(file))

                # logging to file
                self.logged_data_per_run = {
                    "method": "Planner",
                    "file": file,
                    "client": client,
                    "model": model,
                    "tests": []
                }

                self._eval_end_to_end(file)
                # self._eval_correctness_functions(file)
                # self._eval_scope(file)
                # self._eval_layer_and_feature(file)

                self.all_logged_data.append(self.logged_data_per_run)
        # Convert and write JSON object to file
        with open(join("test_logs", f"{time.strftime("%Y%m%d-%H%M%S")}.json"), "w") as outfile:
            json.dump(self.all_logged_data, outfile)

    def _eval_end_to_end(self, file):
        self.logger.info("\nTesting end to end on %s", file)
        start_time = time.time()
        runs = []
        times_testcases = []
        nb_tokens_prompt = 0
        nb_tokens_completion = 0
        for i, testcase in enumerate(self.test_input_files[file]["testcases"]):
            # extract columns from yaml
            prompt = testcase["prompt"]

            # prompt planner
            start_time_testcase = time.time()
            llm_response = self.agent.call_llm_planner(prompt, execute_functions=False)

            # extract only the functions from the planner response
            funcs = self.agent.parser.analyze_functions(llm_response)
            detected = set([func for _, func, _ in funcs])
            result = self.agent.parser.call_functions(funcs)
            time_testcase = time.time() - start_time_testcase

            str_predicted_results = ", ".join(sorted(list(detected), key=str.lower))

            results_dict = {
                "prompt": prompt,
                "executed": str_predicted_results,
                "duration": time_testcase,
                "error": result == 'Unable to parse LLM response',
                "nb_tokens_prompt": self.agent.nb_tokens_prompt - nb_tokens_prompt,
                "nb_tokens_completion": self.agent.nb_tokens_completion - nb_tokens_completion,
            }
            nb_tokens_prompt = self.agent.nb_tokens_prompt
            nb_tokens_completion = self.agent.nb_tokens_completion

            times_testcases.append(time_testcase)
            runs.append(results_dict)
            # logging
            self.logger.debug(f"\n{results_dict}")
        self.logged_data_per_run["tests"].append({
            "test_name": "end-to-end test",
            "amount": i+1,
            "duration": time.time() - start_time,
            "average_testcase_duration": sum(times_testcases)/len(times_testcases),
            "runs": runs,
            "total_nb_tokens_prompt": self.agent.nb_tokens_prompt,
            "total_nb_tokens_completion": self.agent.nb_tokens_completion,
        })

    def _eval_correctness_functions(self, file):
        self.logger.info("\nTesting correct planning on %s", file)
        start_time = time.time()
        correct_results = 0
        runs = []
        for i, testcase in enumerate(self.test_input_files[file]["testcases"]):
            # extract columns from yaml
            prompt = testcase["prompt"]
            expected_results = set(testcase["expectations"])
            exclude = testcase.get('exclude', [])

            # prompt planner
            start_time_testcase = time.time()
            llm_response = self.agent.call_llm_planner(
                prompt, execute_functions=False)
            time_testcase = time.time() - start_time_testcase

            # extract only the functions from the planner response
            detected = set([func for _, func,
                            _ in self.agent.parser.analyze_functions(llm_response)])

            # see if AT LEAST correct functions were called (order here irrelevant)
            correct_result = (expected_results <= detected) and set(exclude).isdisjoint(detected)
            correct_results += correct_result
            str_expected_results = ", ".join(sorted(list(expected_results), key=str.lower))
            str_predicted_results = ", ".join(sorted(list(detected), key=str.lower))

            results_dict = {
                "prompt": prompt,
                "expected": str_expected_results,
                "exclude": exclude,
                "predicted": str_predicted_results,
                "result": correct_result,
                "duration": time_testcase,
            }
            runs.append(results_dict)
            # logging
            self.logger.debug(f"\n{results_dict}")
        self.logger.info("\n%d/%d functions were correctly called from the planner.", correct_results, i+1)
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
        self.logger.info("\n%d/%d scopes were correctly predicted.", correct_results, i+1)
        self.logged_data_per_run["tests"].append({
            "test_name": "scope",
            "correct": correct_results,
            "amount": i+1,
            "runs": runs,
            "duration": time.time() - start_time,
        })

    def _eval_layer_and_feature(self, file):
        self.logger.info("\nTesting layer and feature detection on %s", file)
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
            self.logger.debug("Expected layer and feature: %s, %s", layer, feature)
            self.logger.debug("Detected layer and feature: %s, %s", pred_layer, pred_feature)
            self.logger.debug("Correct Result?: %s", correct_result)

            runs.append({
                "prompt": prompt,
                "expected": ", ".join((layer, feature)),
                "predicted": ", ".join((pred_layer, pred_feature)),
                "result": correct_result,
                "duration": time_testcase,
            })
        self.logger.info("\n%d/%d layers and features were correctly predicted.", correct_results, i+1-skipped_tests)
        self.logged_data_per_run["tests"].append({
            "test_name": "layers and features",
            "correct": correct_results,
            "amount": i+1-skipped_tests,
            "runs": runs,
            "duration": time.time() - start_time
        })


if __name__ == "__main__":
    models = [("groq", "llama3-70b-8192"), ("cerebras", "llama3.1-70b")]
    # filenames = ["wikipedia_cheetah.yaml", "cleanedwikipedia_2016_pres_election.yaml"]
    filenames = os.listdir(os.path.join(os.getcwd(), 'testfiles'))

    EvaluatePlanner(models=models[1:2], filenames=filenames).evaluate()
