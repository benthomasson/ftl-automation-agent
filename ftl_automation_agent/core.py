from ftl_automation_agent.agents import CodeAgent
from smolagents import LiteLLMModel
import yaml
import importlib.resources


def create_model(model_id, context=8192, llm_api_base=None, enable_prompt_caching=False):
    model_kwargs = {
        "model_id": model_id,
        # num_ctx=context,
        "api_base": llm_api_base,
        "temperature": 0,
    }
    return LiteLLMModel(**model_kwargs)


def make_agent(tools, model, max_steps=10, enable_prompt_caching=False):
    prompt_templates = yaml.safe_load(
        importlib.resources.files("ftl_automation_agent.prompts").joinpath("code_agent.yaml").read_text()
    )
    agent = CodeAgent(
        tools=tools,
        model=model,
        verbosity_level=4,
        prompt_templates=prompt_templates,
        max_steps=max_steps,
        enable_prompt_caching=enable_prompt_caching,
    )
    return agent


def run_agent(tools, model, problem_statement, max_steps=10, enable_prompt_caching=False):
    agent = make_agent(tools, model, max_steps=max_steps, enable_prompt_caching=enable_prompt_caching)
    return agent.run(problem_statement, stream=True)


