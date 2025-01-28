from argparse import ArgumentParser
import logging.config
import unittest
import os
import logging
import sys
import yaml
import json
import time

from agent import CLIENTMODELS
from os import getcwd
from os.path import join
from agent import Agent
import utils


class EvaluateActObserve():
    def __init__(self, client, model, filenames=[]) -> None:
        # the models to evaluate. List of pairs of ('client', 'modelname')
        self.client = client
        self.model = model

        # read and store yaml test files
        self.filenames = filenames
        self.outputfilename = f"{time.strftime("%Y%m%d-%H%M%S")}-oaa-{client}-{self.model}"
        self.test_input_files = {}
        for _filename in self.filenames:
            with open(join(getcwd(), "testfiles", _filename)) as f:
                try:
                    yml = yaml.safe_load(f)
                    self.test_input_files[_filename] = yml
                except yaml.YAMLError as exc:
                    print(exc)

        # initialize global values to keep track of
        self.classifications = {}  # storing True Postive, TN, FP, FN for each intent
        self.nb_prompts = 0
        self.nb_errors = 0
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
        fh = logging.FileHandler(join("test_logs", f"{self.outputfilename}.log"))
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    def setup_agent(self, client, model, testing=True):
        # create new agent
        # testing skips the long iterative classifying-spans step. can be used to just retrieve which functions are called

        # clear handlers first to avoid double logging when creating agents multiple times
        l = logging.getLogger('functions')
        l.handlers.clear()
        l = logging.getLogger('output')
        l.handlers.clear()
        fh = logging.FileHandler(join("test_logs", f"{self.outputfilename}.log"))
        fh.setLevel(logging.DEBUG)
        l.addHandler(fh)

        self.agent = Agent(client=client, model=model, toolcalling_functions=True, testing=testing, debug=True)

    def evaluate(self):
        for file in self.filenames:
            self.setup_agent(self.client, self.model)
            # switch open document to corresponding testfile
            self.agent.softwareenv.set_current_document_by_name(utils.remove_file_ending(file))

            # logging to file
            self.logged_data_per_run = {
                "method": "Act and Observe",
                "file": file,
                "client": self.client,
                "model": self.model,
                "tests": []
            }

            self._eval_end_to_end(file)
            # self._eval_correctness_functions(file)
            # self._eval_scope(file)
            # self._eval_layer_and_feature(file)

            self.all_logged_data.append(self.logged_data_per_run)
        # calculate recall and precision via Recall = tp / (tp+fn) and Precision = tp / (tp+fp)
        # f1scores are tuples (score, amount_of_acutal_occurences)
        f1scores = []
        for intent, vals_dict in self.classifications.items():
            tp = vals_dict["tp"]
            tn = vals_dict["tn"]
            fp = vals_dict["fp"]
            fn = vals_dict["fn"]
            recall = 0 if (tp+fn) == 0 else tp / (tp+fn)
            precision = 0 if (tp+fp) == 0 else tp / (tp+fp)
            self.classifications[intent]["recall"] = recall
            self.classifications[intent]["precision"] = precision
            f1score = (2*tp) / (2*tp + fp + fn)
            self.classifications[intent]["f1-score"] = f1score
            f1scores.append((f1score, tp+fn))
        f1score_macro = sum([score for score, _ in f1scores])/len(f1scores)
        f1score_micro = sum([score*nb for score, nb in f1scores])/sum([nb for _, nb in f1scores])
        error_rate = 0 if self.nb_errors == 0 else self.nb_errors / self.nb_prompts
        self.all_logged_data.append(
            {**self.classifications, "f1-score-macro": f1score_macro, "f1-score-micro": f1score_micro, "errorrate": error_rate,
             "total_tokens_prompt": self.agent.nb_tokens_prompt, "total_tokens_completion": self.agent.nb_tokens_completion,
             "total_tokens": self.agent.nb_tokens_prompt+self.agent.nb_tokens_completion})
        # Convert and write JSON object to file
        with open(join("test_logs", f"{self.outputfilename}.json"), "w") as outfile:
            json.dump(self.all_logged_data, outfile)

    def do_classifications(self, expected, detected):
        # make sure it exists already, otherwise add it
        for el in expected | detected:
            if not el in self.classifications:
                self.classifications[el] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            # add em up
            if el in expected:
                if el in detected:
                    self.classifications[el]['tp'] += 1
                else:
                    self.classifications[el]['fn'] += 1
            else:
                if el in detected:
                    self.classifications[el]['fp'] += 1
                else:
                    self.classifications[el]['tn'] += 1

    def _eval_end_to_end(self, file):
        self.logger.info("\nTesting end to end on %s", file)
        start_time = time.time()
        runs = []
        times_testcases = []
        nb_tokens_prompt = 0
        nb_tokens_completion = 0

        for i, testcase in list(enumerate(self.test_input_files[file]["testcases"])):
            # take prompt
            prompt = testcase["prompt"]

            # ask agent
            start_time_testcase = time.time()
            detected_messages = self.agent.call_llm_toolcalling(user_query=prompt)
            time_testcase = time.time() - start_time_testcase

            # check response
            _is_error = isinstance(detected_messages, utils.ParsingException)
            if _is_error:
                str_predicted_results = detected_messages.message
            else:
                # extract only the functions from the act/observe response
                detected = utils.get_toolcalls_from_messages(detected_messages)
                self.do_classifications(expected=set(testcase["expectations"]), detected=set(detected))
                str_predicted_results = ", ".join(detected)

            # log
            results_dict = {
                "prompt": prompt,
                "expected": ", ".join(testcase["expectations"]),
                "executed": str_predicted_results,
                "duration": time_testcase,
                "error": isinstance(detected_messages, utils.ParsingException),
                "nb_tokens_prompt": self.agent.nb_tokens_prompt - nb_tokens_prompt,
                "nb_tokens_completion": self.agent.nb_tokens_completion - nb_tokens_completion,
            }
            nb_tokens_prompt = self.agent.nb_tokens_prompt
            nb_tokens_completion = self.agent.nb_tokens_completion
            self.nb_prompts += 1
            if _is_error:
                self.nb_errors += 1

            times_testcases.append(time_testcase)
            runs.append(results_dict)
            # logging
            self.logger.debug(f"\n{results_dict}")
        self.logged_data_per_run["tests"].append({
            "test_name": "called_functions",
            "amount": i+1,
            "duration": time.time() - start_time,
            "average_testcase_duration": sum(times_testcases)/len(times_testcases),
            "runs": runs,
            "total_nb_tokens_prompt": self.agent.nb_tokens_prompt,
            "total_nb_tokens_completion": self.agent.nb_tokens_completion,
        })

    def _eval_correctness_functions(self, file):
        self.logger.info("\nTesting correct functions on %s", file)
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
            detected_messages = self.agent.call_llm_toolcalling(user_query=prompt)
            time_testcase = time.time() - start_time_testcase

            # extract only the functions from the act/observe response
            detected = set(utils.get_toolcalls_from_messages(detected_messages))
            detected.add('respond')

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
        self.logger.info("\nTesting scope detection on %s", file)
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
    parser = ArgumentParser()
    parser.add_argument("--client", type=str)
    args = parser.parse_args()

    filenames = os.listdir(os.path.join(os.getcwd(), 'testfiles'))

    client = args.client
    if not client is None and client in CLIENTMODELS:
        # do a specific client
        EvaluateActObserve(client=client, model=CLIENTMODELS[client], filenames=filenames).evaluate()
