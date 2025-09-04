# app/agents.py
import asyncio
from app.prompts import wrap_agent_prompt

# Replace with your actual LLM call
async def call_llm(prompt: str) -> dict:
    # stub → integrate with OpenAI, Gemini, or your model
    return {"mock": "response"}

async def run_parallel_agents(journey_type: str, context: dict):
    agents = ["feedback", "issue", "sentiment", "competitor", "ideation"]
    tasks = []
    for agent in agents:
        prompt = wrap_agent_prompt(agent, journey_type, context)
        tasks.append(call_llm(prompt))
    results = await asyncio.gather(*tasks)
    return {agents[i]: results[i] for i in range(len(agents))}
