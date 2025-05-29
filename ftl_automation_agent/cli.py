import asyncio
import os
import time
from threading import Thread

import click
import faster_than_light as ftl
import yaml
from rich.console import Console
from smolagents.agent_types import AgentText

from ftl_automation_agent.memory import ActionStep

from .codegen import (generate_explain_action_step, generate_explain_header,
                      generate_playbook_header, generate_playbook_task,
                      generate_python_header, generate_python_step_footer,
                      generate_python_step_header, generate_python_tool_call,
                      reformat_python)
from .core import create_model, run_agent
from .default_tools import TOOLS
from .prompts import SOLVE_PROBLEM
from .tools import get_tool, load_tools
from .util import resolve_modules_path_or_package

console = Console()


@click.command()
@click.option("--tools", "-t", multiple=True)
@click.option("--tools-files", "-f", multiple=True)
@click.option("--problem", "-p", default=None)
@click.option("--problem-file", "-pf", default=None)
@click.option("--system-design", "-s", prompt="What is the system design?")
@click.option("--model", "-m", default="ollama_chat/deepseek-r1:14b")
@click.option("--modules", "-M", default=['modules'], multiple=True)
@click.option("--inventory", default="inventory.yml")
@click.option("--extra-vars", multiple=True)
@click.option("--output", default="output-{time}.py")
@click.option("--explain", default="output-{time}.txt")
@click.option("--playbook", default="playbook-{time}.yml")
@click.option("--info", multiple=True)
@click.option("--user-input", default="user_input-{time}.yml")
@click.option("--llm-api-base", default=None)
def main(
    tools,
    tools_files,
    problem,
    problem_file,
    system_design,
    model,
    modules,
    inventory,
    extra_vars,
    output,
    explain,
    playbook,
    info,
    user_input,
    llm_api_base,
):

    """A agent that solves a problem given a system design and a set of tools"""
    start = time.time()
    output = output.format(time=start)
    explain = explain.format(time=start)
    user_input = user_input.format(time=start)
    playbook = playbook.format(time=start)
    tool_classes = {}
    tool_classes.update(TOOLS)
    for tf in tools_files:
        tool_classes.update(load_tools(tf))
    model = create_model(model, llm_api_base=llm_api_base)
    if not os.path.exists(inventory):
        with open(inventory, "w") as f:
            f.write(yaml.dump({}))
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    modules_resolved = []
    for modules_path_or_package in modules:
        modules_path = resolve_modules_path_or_package(modules_path_or_package)
        modules_resolved.append(modules_path)

    state = {
        "inventory": ftl.load_inventory(inventory),
        "modules": modules_resolved,
        "localhost": ftl.localhost,
        "user_input": {},
        "gate": None,
        "loop": loop,
        "gate_cache": {},
        "log": None,
        "console": console,
    }
    for extra_var in extra_vars:
        name, _, value = extra_var.partition("=")
        state[name] = value

    if problem_file is not None and problem is not None:
        raise Exception('problem and problem-file are mutually exclusive options')
    elif problem_file is not None and problem is None:
        with open(problem_file) as f:
            problem = f.read()
    elif problem is None:
        raise Exception('no problem')

    generate_python_header(
        output,
        system_design,
        problem,
        tools_files,
        tools,
        inventory,
        modules,
        extra_vars,
        user_input,
    )
    generate_explain_header(explain, system_design, problem)
    generate_playbook_header(playbook, system_design, problem)

    parts = []

    if info:
        parts.append("Use this addition information:")

    for i in info:
        with open(i) as f:
            parts.append(f.read())

    prompt = SOLVE_PROBLEM.format(
                problem=problem, system_design=system_design
            )
    if info:
        prompt = prompt + "\n".join(parts)
    try:
        for o in run_agent(
            tools=[get_tool(tool_classes, t, state) for t in tools],
            model=model,
            problem_statement=prompt,
        ):
            if isinstance(o, ActionStep):
                generate_explain_action_step(explain, o)
                generate_python_step_header(output, o)
                if o.tool_calls:
                    for call in o.tool_calls:
                        generate_python_tool_call(o, output, call)
                    generate_playbook_task(playbook, o)
                generate_python_step_footer(output, o)
            elif isinstance(o, AgentText):
                print(o.to_string())

    finally:
        if state['user_input']:
            with open(user_input, 'w') as f:
                f.write(yaml.dump(state['user_input']))
            print(f"Wrote {user_input}")
        else:
            with open(user_input, 'w') as f:
                f.write(yaml.dump({}))
            print(f"Wrote {user_input}")
        reformat_python(output)
