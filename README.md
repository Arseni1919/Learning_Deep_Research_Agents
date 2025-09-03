# 🧱 Learning Deep Research From Scratch 

## Scope Agent

Important tricks and ideas:
- The structure of the workflow is super simple: to clarify the request with user and to finish, or to write a research brief and to finish

The schema: 

![Screenshot 2025-09-03 at 14.22.06.png](pics/Screenshot%202025-09-03%20at%2014.22.06.png)

## Research Agent

Important tricks and ideas:
- To use `think_tool`, so that the agent can 
- To use the workflow structure that allows an agent to return to search as much as it needed (up to some predefined limit)
- To use very specific prompts that guide the execution of every LLM call. The prompts are very detailed and contain additional examples.

The schema:  
![Screenshot 2025-09-03 at 14.22.15.png](pics/Screenshot%202025-09-03%20at%2014.22.15.png)

## Research Agent MCP

Basically, it is the same as a regular research agent described earlier, but with async tool execution and MCP configuration.


## Research Supervisor

Important tricks and ideas:
- long context is bad
- 

![Screenshot 2025-09-03 at 15.38.33.png](pics/Screenshot%202025-09-03%20at%2015.38.33.png)


## Credits

- [github | deep_research_from_scratch](https://github.com/langchain-ai/deep_research_from_scratch/tree/main?tab=readme-ov-file)
- [anthropic | How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [anthropic | The "think" tool: Enabling Claude to stop and think in complex tool use situations](https://www.anthropic.com/engineering/claude-think-tool)