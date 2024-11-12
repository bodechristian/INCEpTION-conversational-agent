# INCEpTION Conversational Agent

This is a conversational agent for the annotation platform INCEpTION developed at TU Darmstadt.
This agent allows users to interact with the software in natural language.

To use groq API, add a .env with `GROQ_API_KEY="[YOUR_API_KEY]"` first. Or to use Cerebras add `CEREBRAS_API_KEY="[YOUR_API_KEY]"`.

Then use e.g. `python agent.py "tell me a joke"` to see the planner and the response.

## Visualization

Annotations are currently written to `temp.json` at the top level. To use the visualization (https://github.com/bodechristian/CAS-visualization), point it to that .json file.
