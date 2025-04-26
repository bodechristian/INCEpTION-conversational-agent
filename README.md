# INCEpTION Conversational Agent

This is a prototype conversational agent for the annotation platform INCEpTION developed at TU Darmstadt.
This agent allows users to interact with the software in natural language.

To use groq API or other APIs, add your API keys to your environments variables (e.g. as `GROQ_API_KEY=` or `CEREBRAS_API_KEY=`).

The agent can be used with  `python agent.py`. The flag `--toolcalling` will use an Act/Observe approach, the default uses a One-Shot Planner. Optionally the `--debug` flag will show intermediate inner steps.


## Visualization

Annotations are currently written to `temp.json` at the top level. To use the visualization (https://github.com/bodechristian/CAS-visualization), point it to that .json file.


## Evaluation

There are the commands used for evaluation

```
python .\evaluate.py --client [ukp/openai/groq/cerebras] --mode planner 
python .\evaluate.py --client [ukp/openai/groq/cerebras] --mode planner --hugginggpt
python .\evaluate.py --client [ukp/openai/groq/cerebras] --mode sequential
python .\evaluate.py --client [ukp/openai/groq/cerebras] --mode sequential --thoughts
python .\evaluate.py --client ukp --mode sequential_no_tools
python .\evaluate.py --client ukp --mode planner --parampred
```
