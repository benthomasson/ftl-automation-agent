import click

from .core import create_model, make_agent
from .default_tools import TOOLS
from .tools import get_tool, load_tools
import faster_than_light as ftl
import gradio as gr
from functools import partial

from .codegen import (
    generate_python_header,
    reformat_python,
    add_lookup_plugins,
    generate_explain_header,
    generate_playbook_header,
)

from ftl_agent.util import Bunch
from ftl_agent.Gradio_UI import stream_to_gradio


def bot(context, prompt, messages, system_design, tools):
    agent = make_agent(
        tools=[get_tool(context.tool_classes, t, context.state) for t in tools],
        model=context.model,
    )
    generate_python_header(
        context.python,
        system_design,
        prompt,
        context.tools_files,
        tools,
        context.inventory,
        context.modules,
        context.extra_vars,
    )
    generate_explain_header(context.explain, system_design, prompt)
    generate_playbook_header(context.playbook, system_design, prompt)

    def update_code():
        nonlocal python_output, playbook_output
        with open(context.python) as f:
            python_output = f.read()
        with open(context.playbook) as f:
            playbook_output = f.read()

    python_output = ""
    playbook_output = ""

    update_code()

    # chat interface only needs the latest messages yielded
    messages = []
    messages.append(gr.ChatMessage(role="user", content=prompt))
    yield messages, python_output, playbook_output
    for msg in stream_to_gradio(
        agent, context, task=prompt, reset_agent_memory=False
    ):
        update_code()
        messages.append(msg)
        yield messages, python_output, playbook_output

    reformat_python(context.python)
    add_lookup_plugins(context.playbook)
    update_code()
    yield messages, python_output, playbook_output


def launch(context, tool_classes, system_design, **kwargs):
    with gr.Blocks(fill_height=True) as demo:
        python_code = gr.Code(render=False, label="FTL Automation")
        playbook_code = gr.Code(render=False, label="Ansible playbook")
        with gr.Row():
            with gr.Column():
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
                        gr.Textbox(system_design, label="System Design"),
                        gr.CheckboxGroup(
                            choices=sorted(tool_classes), label="Tools"
                        ),
                    ],
                    additional_outputs=[python_code, playbook_code],
                )

            with gr.Column():
                python_code.render()
                playbook_code.render()

        demo.launch(debug=True, **kwargs)


@click.command()
@click.option("--tools-files", "-f", multiple=True)
@click.option("--tools", "-t", multiple=True)
@click.option("--system-design", "-s")
@click.option("--model", "-m", default="ollama_chat/deepseek-r1:14b")
@click.option("--inventory", "-i", default="inventory.yml")
@click.option("--modules", "-M", default=["modules"], multiple=True)
@click.option("--extra-vars", "-e", multiple=True)
@click.option("--python", "-o", default="output.py")
@click.option("--explain", "-o", default="output.txt")
@click.option("--playbook", default="playbook.yml")
@click.option("--server-name", default="127.0.0.1")
@click.option("--llm-api-base", default=None)
def main(
    tools_files,
    tools,
    system_design,
    model,
    inventory,
    modules,
    extra_vars,
    python,
    explain,
    playbook,
    server_name,
    llm_api_base,
):
    """A agent that solves a problem given a system design and a set of tools"""
    tool_classes = {}
    tool_classes.update(TOOLS)
    for tf in tools_files:
        tool_classes.update(load_tools(tf))
    model = create_model(model, llm_api_base=llm_api_base)
    state = {
        "inventory": ftl.load_inventory(inventory),
        "modules": modules,
        "localhost": ftl.localhost,
        "gate": None,
        "loop": None,
        "gate_cache": None,
    }
    for extra_var in extra_vars:
        name, _, value = extra_var.partition("=")
        state[name] = value

    context = Bunch(
        tool_classes=tool_classes,
        state=state,
        tools_files=tools_files,
        tools=tools,
        system_design=system_design,
        model=model,
        inventory=inventory,
        modules=modules,
        extra_vars=extra_vars,
        python=python,
        explain=explain,
        playbook=playbook,
    )

    launch(context, tool_classes, system_design, server_name=server_name)
