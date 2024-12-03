# INCEpTION Conversational Agent

This is a prototype conversational agent for the annotation platform INCEpTION developed at TU Darmstadt.
This agent allows users to interact with the software in natural language.

To use groq API or other APIs, add your API keys to your environments variables (e.g. as `GROQ_API_KEY=` or `CEREBRAS_API_KEY=`). By default a Cerebras key is required for the planner. The toolcalling method additionally requires a Groq key.

The agent can be used with  `python agent.py`. The flag `--toolcalling` will use an Act/Observe approach, the default uses a planner. Optionally the `--debug` flag will show intermediate inner steps.

Use `python evaluate_planner.py` or `python evaluate_act_observe.py` to run an evaluation of each approach.

## Visualization

Annotations are currently written to `temp.json` at the top level. To use the visualization (https://github.com/bodechristian/CAS-visualization), point it to that .json file.
