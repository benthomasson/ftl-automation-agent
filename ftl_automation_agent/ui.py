import os
import time

import click

from .core import create_model, make_agent
from .default_tools import TOOLS
from .tools import get_tool, load_tools
import faster_than_light as ftl
import gradio as gr
from functools import partial
import yaml
from rich.console import Console

from .codegen import (
    generate_python_header,
    reformat_python,
    add_lookup_plugins,
    generate_explain_header,
    generate_playbook_header,
)

from .util import resolve_modules_path_or_package

from ftl_automation_agent.util import Bunch
from ftl_automation_agent.Gradio_UI import stream_to_gradio

console = Console()


TASK_PROMPT = """

Use the user_input_tool to ask for additional information.
Use the complete() tool to signal that you are done

Do not use example data, ask the user for input using user_input_tool().
Do not ask for root passwords, user tokens, API keys, or secrets.  They will be provided to the tools directly.
Do not import the `os` package.
Do not import the `socket` package.
Do not use the open function.
Only use the tools provided.
Do not assume input values to the tools.  Ask the user.


This is a real scenario.  Use the tools provided or ask for assistance.

"""


def bot(context, prompt, messages, system_design, tools):

    if isinstance(prompt, dict):
        prompt = prompt['text']

    full_prompt = TASK_PROMPT + prompt
    agent = make_agent(
        tools=[get_tool(context.tool_classes, t, context.state) for t in tools],
        model=context.model,
    )
    generate_python_header(
        context.output,
        system_design,
        prompt,
        context.tools_files,
        tools,
        context.inventory,
        context.modules,
        context.extra_vars,
        context.user_input,
    )
    generate_explain_header(context.explain, system_design, full_prompt)
    generate_playbook_header(context.playbook, system_design, prompt)

    def update_code():
        nonlocal python_output, playbook_output
        with open(context.output) as f:
            python_output = f.read()
        with open(context.playbook) as f:
            playbook_output = f.read()

    python_output = ""
    playbook_output = ""

    update_code()

    # chat interface only needs the latest messages yielded
    messages = []
    messages.append(gr.ChatMessage(role="user", content=full_prompt))
    yield messages, python_output, playbook_output
    for msg in stream_to_gradio(
        agent, context, task=full_prompt, reset_agent_memory=False
    ):
        update_code()
        messages.append(msg)
        yield messages, python_output, playbook_output

    if context.state['user_input']:
        with open(context.user_input, 'w') as f:
            f.write(yaml.dump(context.state['user_input']))
        print(f"Wrote {context.user_input}")
    else:
        with open(context.user_input, 'w') as f:
            f.write(yaml.dump({}))
        print(f"Wrote {context.user_input}")
    reformat_python(context.output)
    add_lookup_plugins(context.playbook)
    update_code()

    yield messages, python_output, playbook_output


def launch(context, tool_classes, system_design, **kwargs):
    with gr.Blocks(fill_height=True) as demo:

        python_code = gr.Code(render=False, label="FTL Automation")
        playbook_code = gr.Code(render=False, label="Ansible playbook")
        with gr.Row():
            with gr.Column():
                system_design_field = gr.Textbox(system_design, label="System Design", render=False)
                tool_check_boxes = gr.CheckboxGroup(
                    choices=sorted(tool_classes), value=sorted(context.tools), label="Tools", render=False
                )

                chatbot = gr.Chatbot(
                    label="FTL Agent",
                    type="messages",
                    resizeable=True,
                    scale=1,
                )
                gr.ChatInterface(
                    fn=partial(bot, context),
                    type="messages",
                    chatbot=chatbot,
                    additional_inputs=[
                        system_design_field,
                        tool_check_boxes,
                    ],
                    additional_inputs_accordion=gr.Accordion(label="Additional Inputs", open=False, render=False),
                    additional_outputs=[python_code, playbook_code],
                    textbox=gr.MultimodalTextbox(file_types=['text_encoded'], value=context.problem),
                )

            with gr.Column():

                current_question_input = gr.Textbox(visible=False)

                @gr.render(inputs=current_question_input)
                def render_form(*args, **kwargs):
                    print('render_form')
                    print(args)
                    print(kwargs)
                    print(context.state['questions'])
                    print(context.state['user_input'])
                    if context.state["questions"]:
                        gr.Markdown("### Please answer the following questions:")
                        inputs = []
                        for q in context.state["questions"]:
                            if q in context.state["user_input"]:
                                inputs.append(gr.Textbox(label=q, value=context.state["user_input"][q], interactive=False))
                            else:
                                inputs.append(gr.Textbox(label=q, ))
                        answer_button = gr.Button("Submit")

                        def answer_questions(*args, **kwargs):
                            print(args)
                            print(kwargs)
                            for question, answer in zip(context.state["questions"], args):
                                context.state["user_input"][question] = answer

                        answer_button.click(
                            answer_questions,
                            inputs=inputs
                        )

                def update_questions():
                    return context.state["questions"]

                gr.Timer(1).tick(fn=update_questions, outputs=current_question_input)

                # python_code.render()
                # playbook_code.render()

        demo.launch(debug=True, **kwargs)


@click.command()
@click.option("--tools", "-t", multiple=True)
@click.option("--tools-files", "-f", multiple=True)
@click.option("--tools", "-t", multiple=True)
@click.option("--problem", "-p", default=None)
@click.option("--problem-file", "-pf", default=None)
@click.option("--system-design", "-s")
@click.option("--model", "-m", default="ollama_chat/deepseek-r1:14b")
@click.option("--modules", "-M", default=["modules"], multiple=True)
@click.option("--inventory", "-i", default="inventory.yml")
@click.option("--extra-vars", "-e", multiple=True)
@click.option("--output", default="output-{time}.py")
@click.option("--explain", default="output-{time}.txt")
@click.option("--playbook", default="playbook-{time}.yml")
@click.option("--info", multiple=True)
@click.option("--user-input", default="user_input-{time}.yml")
@click.option("--server-name", default="127.0.0.1")
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
    server_name,
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
        "loop": None,
        "gate_cache": None,
        "log": None,
        "console": console,
        "questions": [],
    }
    for extra_var in extra_vars:
        name, _, value = extra_var.partition("=")
        state[name] = value

    if problem_file is not None and problem is not None:
        raise Exception('problem and problem-file are mutually exclusive options')
    elif problem_file is not None and problem is None:
        with open(problem_file) as f:
            problem = f.read()

    context = Bunch(
        tool_classes=tool_classes,
        state=state,
        tools_files=tools_files,
        tools=tools,
        problem=problem,
        system_design=system_design,
        model=model,
        inventory=inventory,
        modules=modules_resolved,
        extra_vars=extra_vars,
        output=output,
        explain=explain,
        playbook=playbook,
        user_input=user_input,
    )

    launch(context, tool_classes, system_design, server_name=server_name)
