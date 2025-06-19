import os
import time

import click
import json

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

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import FastAPI, Depends, Request
from starlette.config import Config
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
import uvicorn

from pprint import pprint


console = Console()


TASK_PROMPT = """

Use the gradio_input_tool to ask for additional information.
Use the complete() tool to signal that you are done

Do not use example data, ask the user for input using gradio_input_tool().
Do not ask for root passwords, user tokens, API keys, or secrets.  They will be provided to the tools directly.
Do not import the `os` package.
Do not import the `socket` package.
Do not use the open function.
Only use the tools provided.
Do not assume input values to the tools.  Ask the user.


This is a real scenario.  Use the tools provided or ask for assistance.

"""


def load_session(sub):
    session_file_name = f"sessions/{sub}/session.json"
    if os.path.exists(session_file_name):
        with open(session_file_name) as f:
            data = json.loads(f.read())
        return data
    else:
        return {}


def save_session(sub, data):
    session_file_name = f"sessions/{sub}/session.json"
    os.makedirs(os.path.dirname(session_file_name), exist_ok=True)
    with open(session_file_name, "w") as f:
        f.write(json.dumps(data))


def bot(context, prompt, messages, system_design, tools):

    if isinstance(prompt, dict):
        prompt = prompt["text"]

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
        nonlocal python_output, playbook_output, inventory_text
        with open(context.output) as f:
            python_output = f.read()
        with open(context.playbook) as f:
            playbook_output = f.read()
        print(context.inventory)
        if os.path.exists(context.inventory):
            with open(context.inventory) as f:
                inventory_text = f.read()

    python_output = ""
    playbook_output = ""
    inventory_text = ""
    print(f"{context.inventory=}")
    if os.path.exists(context.inventory):
        with open(context.inventory) as f:
            inventory_text = f.read()

    update_code()

    # chat interface only needs the latest messages yielded
    messages = []
    # messages.append(gr.ChatMessage(role="user", content=full_prompt))
    yield messages, python_output, playbook_output, inventory_text
    for msg in stream_to_gradio(
        agent, context, task=full_prompt, reset_agent_memory=False
    ):
        update_code()
        messages.append(msg)
        yield messages, python_output, playbook_output, inventory_text

    if context.state["user_input"]:
        with open(context.user_input, "w") as f:
            f.write(yaml.dump(context.state["user_input"]))
        print(f"Wrote {context.user_input}")
    else:
        with open(context.user_input, "w") as f:
            f.write(yaml.dump({}))
        print(f"Wrote {context.user_input}")
    reformat_python(context.output)
    add_lookup_plugins(context.playbook)
    update_code()

    yield messages, python_output, playbook_output, inventory_text


def launch(context, tool_classes, system_design, **kwargs):

    app = FastAPI()

    # Replace these with your own OAuth settings
    GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
    GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
    ALLOWED_USERS = os.environ["ALLOWED_USERS"].split(",")

    config_data = {
        "GOOGLE_CLIENT_ID": GOOGLE_CLIENT_ID,
        "GOOGLE_CLIENT_SECRET": GOOGLE_CLIENT_SECRET,
    }
    starlette_config = Config(environ=config_data)
    oauth = OAuth(starlette_config)
    oauth.register(
        name="google",
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

    SECRET_KEY = os.environ["SECRET_KEY"]
    app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

    # Dependency to get the current user
    def get_user(request: Request):
        user = request.session.get("user")
        if user:
            return user["name"]
        return None

    @app.get("/")
    def public(user: dict = Depends(get_user)):
        if user:
            return RedirectResponse(url="/ftl")
        else:
            return RedirectResponse(url="/login-ftl")

    @app.route("/logout")
    async def logout(request: Request):
        request.session.pop("user", None)
        return RedirectResponse(url="/")

    @app.route("/login")
    async def login(request: Request):
        redirect_uri = request.url_for("auth")
        # If your app is running on https, you should ensure that the
        # `redirect_uri` is https, e.g. uncomment the following lines:
        #
        from urllib.parse import urlparse, urlunparse

        redirect_uri = urlunparse(urlparse(str(redirect_uri))._replace(scheme="https"))
        return await oauth.google.authorize_redirect(request, redirect_uri)

    @app.route("/auth")
    async def auth(request: Request):
        try:
            access_token = await oauth.google.authorize_access_token(request)
        except OAuthError:
            return RedirectResponse(url="/")
        pprint(dict(access_token)["userinfo"])
        if access_token["userinfo"]["email"] not in ALLOWED_USERS:
            return RedirectResponse(url="/")
        request.session["user"] = dict(access_token)["userinfo"]
        return RedirectResponse(url="/")

    with gr.Blocks() as login_demo:
        gr.Button("Login", link="/login")

    app = gr.mount_gradio_app(app, login_demo, path="/login-ftl")

    persistent_sessions = {}

    def initialize(request: gr.Request):
        pprint(request.request.session["user"])
        data = load_session(request.request.session["user"]["sub"])
        pprint(data)
        persistent_sessions[request.session_hash] = data
        return (
            f"Welcome, {request.username}!",
            data.get("title"),
            data.get("system_design"),
            data.get("user_input"),
        )

    def cleanup(request: gr.Request):
        print("cleanup")
        if request.session_hash in persistent_sessions:
            pprint(persistent_sessions[request.session_hash])
            save_session(
                request.request.session["user"]["sub"],
                persistent_sessions[request.session_hash],
            )
            del persistent_sessions[request.session_hash]

    def persist_title_input(request: gr.Request, title):
        persistent_sessions[request.session_hash]["title"] = title

    def persist_system_design(request: gr.Request, system_design):
        persistent_sessions[request.session_hash]["system_design"] = system_design

    def clear_session(request: gr.Request):
        print("clear_session")
        data = {"title": "Session", "system_design": ""}
        persistent_sessions[request.session_hash] = data
        context.state['questions'] = []
        context.state['user_input'] = {}
        return data["title"], data["system_design"], None

    with gr.Blocks(fill_height=True) as demo:

        with gr.Sidebar(position="left", open=False):
            with gr.Column(scale=1):
                title = gr.Textbox(
                    show_label=False,
                    value="New Session",
                    interactive=True,
                    submit_btn=False,
                    scale=0,
                )
                title.input(persist_title_input, inputs=[title])
                clear_session_btn = gr.Button("Clear", scale=0)
                gr.Button("New", scale=0)

        with gr.Sidebar(position="right", open=False):
            m = gr.Markdown("Welcome to Gradio!")
            gr.Button("Logout", link="/logout", scale=0)

        python_code = gr.Code(render=False, label="FTL Automation", language="python", visible=False)
        playbook_code = gr.Code(render=False, label="Ansible playbook", language="yaml", visible=False)
        inventory_text = gr.Code(render=False, label="Inventory", language="yaml", visible=False)
        with gr.Row():
            with gr.Column():
                system_design_field = gr.Textbox(
                    system_design, label="System Design", render=False
                )
                system_design_field.input(
                    persist_system_design, inputs=[system_design_field]
                )
                tool_check_boxes = gr.CheckboxGroup(
                    choices=sorted(tool_classes),
                    value=sorted(context.tools),
                    label="Tools",
                    render=False,
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
                    stop_btn=True,
                    additional_inputs=[
                        system_design_field,
                        tool_check_boxes,
                    ],
                    additional_inputs_accordion=gr.Accordion(
                        label="Additional Inputs", open=False, render=False
                    ),
                    additional_outputs=[python_code, playbook_code, inventory_text],
                    textbox=gr.MultimodalTextbox(
                        file_types=["text_encoded"], value=context.problem
                    ),
                )

            with gr.Column():

                current_question_input = gr.Textbox(visible=False)

                @gr.render(inputs=current_question_input)
                def render_form(request: gr.Request, *args, **kwargs):
                    print("render_form")
                    print(args)
                    print(kwargs)
                    if persistent_sessions[request.session_hash].get('user_input'):
                        context.state["questions"] = []
                        for question, answer in persistent_sessions[request.session_hash]['user_input'].items():
                            context.state["questions"].append(question)
                            context.state["user_input"][question] = answer
                    print(context.state["questions"])
                    print(context.state["user_input"])
                    if context.state["questions"]:
                        gr.Markdown("### Please answer the following questions:")
                        inputs = []
                        for q in context.state["questions"]:
                            if q in context.state["user_input"]:
                                inputs.append(
                                    gr.Textbox(
                                        label=q,
                                        value=context.state["user_input"][q],
                                        interactive=False,
                                    )
                                )
                            else:
                                inputs.append(
                                    gr.Textbox(
                                        label=q,
                                    )
                                )
                        answer_button = gr.Button("Submit")

                        def answer_questions(request: gr.Request, *args, **kwargs):
                            print(args)
                            print(kwargs)
                            persistent_sessions[request.session_hash]["user_input"] = {}
                            for question, answer in zip(
                                context.state["questions"], args
                            ):
                                context.state["user_input"][question] = answer
                                persistent_sessions[request.session_hash]["user_input"][
                                    question
                                ] = answer

                        answer_button.click(answer_questions, inputs=inputs)

                def update_questions():
                    return context.state["questions"]

                def update_inventory():
                    inventory_text = ""
                    if os.path.exists(context.inventory):
                        with open(context.inventory) as f:
                            inventory_text = f.read()
                    return inventory_text

                gr.Timer(1).tick(fn=update_questions, outputs=current_question_input)
                gr.Timer(1).tick(fn=update_inventory, outputs=inventory_text)

                python_code.render()
                playbook_code.render()
                inventory_text.render()

        clear_session_btn.click(
            clear_session, inputs=None, outputs=[title, system_design_field, current_question_input]
        )
        demo.load(initialize, inputs=None, outputs=[m, title, system_design_field, current_question_input])
        demo.unload(cleanup)

    app = gr.mount_gradio_app(app, demo, path="/ftl", auth_dependency=get_user)

    uvicorn.run(app, host="0.0.0.0", port=7860)


@click.command()
@click.option("--tools", "-t", multiple=True)
@click.option("--tools-files", "-f", default=["ftl_tools.tools"], multiple=True)
@click.option("--tools", "-t", multiple=True)
@click.option("--problem", "-p", default=None)
@click.option("--problem-file", "-pf", default=None)
@click.option("--system-design", "-s")
@click.option("--model", "-m", default="ollama_chat/deepseek-r1:14b")
@click.option("--modules", "-M", default=["ftl_modules"], multiple=True)
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
        raise Exception("problem and problem-file are mutually exclusive options")
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
