from argparse import ArgumentParser
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
from agent import CLIENTMODELS, Agent


class EvaluatePlanner():
    def __init__(self, client: str, model: str, filenames=[], mode="planner") -> None:
        # the models to evaluate.
        self.client = client
        self.model = model
        self.mode = mode  # planner, sequential, sequential_no_tools

        # read and store yaml test files
        self.filenames = filenames
        self.outputfilename = f"{time.strftime("%Y%m%d-%H%M%S")}-{self.mode}-{self.client}-{self.model.split(":")[0]}"
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
        self.nb_nonerrored_prompts = 0
        self.nb_errors = 0
        self.nb_correct_layerandfeature = 0
        self.nb_incorrect_layerandfeature = 0
        self.nb_correct_scope = 0
        self.nb_incorrect_scope = 0
        self.valid_dags = 0
        self.nb_total_unused_funcs = 0
        self.completely_corrects = 0
        self.nb_tokens_prompt = 0
        self.nb_tokens_completion = 0
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
        # clear handlers first to avoid double logging when creating agents multiple times
        l = logging.getLogger('functions')
        l.handlers.clear()
        l = logging.getLogger('output')
        l.handlers.clear()
        fh = logging.FileHandler(join("test_logs", f"{self.outputfilename}.log"))
        fh.setLevel(logging.DEBUG)
        l.addHandler(fh)

        if self.mode == "planner":
            self.agent = Agent(mode='planner', client=client, model=model, testing=testing, debug=True)
        elif self.mode == "sequential":
            self.agent = Agent(mode='sequential', client=client, model=model,
                               toolcalling_functions=True, testing=testing, debug=True)
        elif self.mode == "sequential_no_tools":
            self.agent = Agent(mode='sequential', client=client, model=model,
                               toolcalling_functions=False, testing=testing, debug=True)

    def evaluate(self):
        for file in self.filenames:
            self.setup_agent(self.client, self.model)
            # switch open document to corresponding testfile
            self.agent.softwareenv.set_current_document_by_name(utils.remove_file_ending(file))

            # logging to file
            self.logged_data_per_run = {
                "method": "Planner",
                "file": file,
                "client": self.client,
                "model": self.model,
                "tests": []
            }

            if self.mode == "planner":
                self._evaluate_planner(file)
            elif self.mode == "sequential":
                self._evaluate_planner(file)
            elif self.mode == "sequential_no_tools":
                self._evaluate_planner(file)

            self.nb_tokens_prompt += self.agent.nb_tokens_prompt
            self.nb_tokens_completion += self.agent.nb_tokens_completion
            self.all_logged_data.append(self.logged_data_per_run)

        # error rates
        error_rate = 0 if self.nb_errors == 0 else self.nb_errors / self.nb_prompts
        correct_scope_rate = self.nb_correct_scope / \
            (self.nb_correct_scope+self.nb_incorrect_scope) if self.nb_correct_scope+self.nb_incorrect_scope > 0 else 0
        correct_layerandfeature_rate = self.nb_correct_layerandfeature / \
            (self.nb_correct_layerandfeature+self.nb_incorrect_layerandfeature) if self.nb_correct_layerandfeature + \
            self.nb_incorrect_layerandfeature > 0 else 0

        # log everything
        self.all_logged_data.append(
            {**self.classifications,
                "errorrate": error_rate,
                "correct scope rate": correct_scope_rate,
                "correct layer and feature rate": correct_layerandfeature_rate,
                "valid DAGs": self.valid_dags / self.nb_nonerrored_prompts,
                "avg_unused_funcs": self.nb_total_unused_funcs / self.nb_nonerrored_prompts,
                "completely corrects": self.completely_corrects / self.nb_prompts,
                "total_tokens_prompt": self.nb_tokens_prompt,
                "total_tokens_completion": self.nb_tokens_completion,
                "total_tokens": self.nb_tokens_prompt+self.nb_tokens_completion,
             })
        # Convert and write JSON object to file
        with open(join("test_logs", f"{self.outputfilename}.json"), "w") as outfile:
            json.dump(self.all_logged_data, outfile)

    def _evaluate_planner(self, file):
        self.logger.info("\nTesting end to end on %s", file)
        start_time = time.time()
        runs = []
        times_testcases = []
        nb_tokens_prompt = 0
        nb_tokens_completion = 0
        for i, testcase in list(enumerate(self.test_input_files[file]["testcases"])):
            # extract columns from yaml
            prompt = testcase["prompt"]
            expectations_dag = utils.load_dag(testcase["expectations"])

            # prompt planner
            start_time_testcase = time.time()
            llm_response = self.agent.call_llm_planner(prompt, execute_functions=False)

            # extract only the functions from the planner response
            funcs = self.agent.parser.analyze_functions(llm_response)
            detected_funcs = [func for _, func, _ in funcs]
            result = self.agent.parser.call_functions(funcs)
            time_testcase = time.time() - start_time_testcase

            # check if agent errored and called the correct settings
            _is_error = isinstance(result, utils.ParsingException)
            correct_layerandfeature = ""
            if testcase['layer'] and testcase['feature']:
                if 'layer' in self.agent.get_state() and 'feature' in self.agent.get_state():
                    pred_layer = self.agent.get_state()['layer'].split('.')[-1]  # webanno.custom.Animal -> Animal
                    pred_feature = self.agent.get_state()['feature']
                    correct_layerandfeature = (pred_feature == testcase['feature'] and pred_layer == testcase['layer'])
                    if correct_layerandfeature:
                        self.nb_correct_layerandfeature += 1
                    else:
                        self.nb_incorrect_layerandfeature += 1

            correct_scope = ""
            if testcase['scope']:
                # if scope is expected to have a value
                if 'scope' in self.agent.get_state():
                    # check if the agent state has set it
                    pred_scope = self.agent.get_state()['scope']
                    # and check if its predicted correctly
                    correct_scope = pred_scope == testcase['scope']
                    if correct_scope:
                        self.nb_correct_scope += 1
                    else:
                        self.nb_incorrect_scope += 1

            if not _is_error:
                dag_is_valid, nb_unused_funcs = utils.eval_functions_dag(detected_funcs, expectations_dag)
                self.nb_nonerrored_prompts += 1
                self.valid_dags += dag_is_valid
                self.nb_total_unused_funcs += nb_unused_funcs
            str_predicted_results = ", ".join(detected_funcs)

            # correct settings + correct intents
            completely_correct = (correct_scope == "" or correct_scope == True) and (
                correct_layerandfeature == "" or correct_layerandfeature == True) and (
                    not _is_error and (dag_is_valid and (nb_unused_funcs == 0)))
            self.completely_corrects += completely_correct

            # log
            results_dict = {
                "prompt": prompt,
                "expected": utils.flatten_dag(expectations_dag),
                "executed": str_predicted_results,
                "duration": time_testcase,
                "error": _is_error,
                "is_valid": dag_is_valid if not _is_error else "",
                "nb_overcalls": nb_unused_funcs if not _is_error else "",
                "correct_scope": correct_scope,
                "correct_layerandfeature": correct_layerandfeature,
                "completely correct": completely_correct,
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
            self.logger.info(result)
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

    def _evaluate_sequential(self, file):
        self.logger.info("\nTesting end to end on %s", file)
        start_time = time.time()
        runs = []
        times_testcases = []
        nb_tokens_prompt = 0
        nb_tokens_completion = 0

        for i, testcase in list(enumerate(self.test_input_files[file]["testcases"])):
            # take prompt
            prompt = testcase["prompt"]
            expectations_dag = utils.load_dag(testcase["expectations"])

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
                str_predicted_results = ", ".join(detected)
                dag_is_valid, nb_unused_funcs = utils.eval_functions_dag(detected, expectations_dag)
                self.valid_dags += dag_is_valid
                self.nb_total_unused_funcs += nb_unused_funcs
                self.nb_nonerrored_prompts += 1

            # log
            results_dict = {
                "prompt": prompt,
                "expected": utils.flatten_dag(expectations_dag),
                "executed": str_predicted_results,
                "duration": time_testcase,
                "error": isinstance(detected_messages, utils.ParsingException),
                "is_valid": dag_is_valid,
                "nb_overcalls": nb_unused_funcs,
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

    def _evaluate_sequential_no_tools(self, file):
        self.logger.info("\nTesting end to end on %s", file)
        start_time = time.time()
        runs = []
        times_testcases = []
        nb_tokens_prompt = 0
        nb_tokens_completion = 0

        for i, testcase in enumerate(self.test_input_files[file]["testcases"]):
            # take prompt
            prompt = testcase["prompt"]
            expectations_dag = utils.load_dag(testcase["expectations"])

            # ask agent
            start_time_testcase = time.time()
            detected_messages = self.agent.call_llm_sequential_no_tools(user_query=prompt)
            time_testcase = time.time() - start_time_testcase

            # check response
            _is_error = isinstance(detected_messages, utils.ParsingException)
            if _is_error:
                str_predicted_results = detected_messages.message
            else:
                # extract only the functions from the act/observe response
                detected = utils.get_toolcalls_from_messages(detected_messages)
                str_predicted_results = ", ".join(detected)
                dag_is_valid, nb_unused_funcs = utils.eval_functions_dag(detected, expectations_dag)
                self.valid_dags += dag_is_valid
                self.nb_total_unused_funcs += nb_unused_funcs
                self.nb_nonerrored_prompts += 1

            # log
            results_dict = {
                "prompt": prompt,
                "expected": utils.flatten_dag(expectations_dag),
                "executed": str_predicted_results,
                "duration": time_testcase,
                "error": isinstance(detected_messages, utils.ParsingException),
                "is_valid": dag_is_valid,
                "nb_overcalls": nb_unused_funcs,
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--client", type=str)
    parser.add_argument("--mode", type=str, nargs='?', default='planner')
    args = parser.parse_args()

    filenames = os.listdir(os.path.join(os.getcwd(), 'testfiles'))

    client = args.client
    if not client is None and client in CLIENTMODELS:
        # do a specific client
        EvaluatePlanner(client=client, model=CLIENTMODELS[client], filenames=filenames, mode=args.mode).evaluate()
