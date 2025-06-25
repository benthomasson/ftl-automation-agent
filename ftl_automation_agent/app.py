import os
import time

import click
import json
import shutil
import glob


from collections import defaultdict
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
Use local paths for all source paths for files. They are relative to the local workspace.
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


def launch(model, tool_classes, modules):

    app = FastAPI()

    persistent_sessions = defaultdict(dict)
    user_contexts = defaultdict(dict)

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

    def bot(request, prompt, messages, system_design, tools):

        context = user_contexts[request.session_hash]

        if isinstance(prompt, dict):
            prompt = prompt["text"]

        full_prompt = TASK_PROMPT + prompt + "\nThen call complete() when done."
        agent = make_agent(
            tools=[get_tool(tool_classes, t, context.state) for t in tools],
            model=model,
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

        messages = []
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

    def initialize(request: gr.Request):
        pprint(request.request.session["user"])
        data = load_session(request.request.session["user"]["sub"])
        pprint(data)
        persistent_sessions[request.session_hash] = data
        user_contexts[request.session_hash] = Bunch()
        return (
            f"Welcome, {request.username}!",
            data.get("title"),
            data.get("system_design"),
            data.get("user_input"),
            data.get("chat"),
            data.get("python_code"),
            data.get("playbook_code"),
            data.get("inventory_text"),
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

    def persist_tool_check_boxes(request: gr.Request, tool_check_boxes):
        persistent_sessions[request.session_hash]["tool_check_boxes"] = tool_check_boxes

    def persist_chat(request: gr.Request, chat):
        persistent_sessions[request.session_hash]["chat"] = chat

    def persist_python_code(request: gr.Request, python_code):
        persistent_sessions[request.session_hash]["python_code"] = python_code

    def persist_playbook_code(request: gr.Request, playbook_code):
        persistent_sessions[request.session_hash]["playbook_code"] = playbook_code

    def persist_inventory_text(request: gr.Request, inventory_text):
        persistent_sessions[request.session_hash]["inventory_text"] = inventory_text

    def clear_session(request: gr.Request):
        print("clear_session")
        data = {"title": "Session", "system_design": ""}
        persistent_sessions[request.session_hash] = data
        context = user_contexts[request.session_hash]
        context.state["questions"] = []
        context.state["user_input"] = {}
        return data["title"], data["system_design"], None, None, None, None, None

    def render_left_bar():

        with gr.Sidebar(position="left", open=False):
            with gr.Column(scale=1):
                title = gr.Textbox(
                    show_label=False,
                    value="New Session",
                    interactive=True,
                    submit_btn=False,
                    scale=0,
                )
                title.change(persist_title_input, inputs=[title])
                clear_session_btn = gr.Button("Clear", scale=0)
                gr.Button(
                    "New",
                    scale=0,
                    variant="primary",
                    icon=gr.utils.get_icon_path("plus.svg"),
                    visible=False,
                )

        return title, clear_session_btn

    def render_right_bar():

        with gr.Sidebar(position="right", open=False):
            welcome = gr.Markdown("Welcome to Gradio!")
            gr.Button("Logout", link="/logout", scale=0)

        return welcome

    def upload_file(request: gr.Request, files):
        for file_name in files:
            print("upload", file_name)
            shutil.copy(file_name, user_contexts[request.session_hash].workspace)

    def delete_file(
        request: gr.Request, deleted_file: gr.DeletedFileData, *args, **kwargs
    ):
        print("delete", deleted_file, deleted_file.file.path)
        print(args)
        print(kwargs)
        workspace_file_name = os.path.join(
            user_contexts[request.session_hash].workspace,
            os.path.basename(deleted_file.file.path),
        )
        if os.path.exists(workspace_file_name):
            os.unlink(workspace_file_name)

    def clear_file(request: gr.Request, *args, **kwargs):
        print("clear")
        print(args)
        print(kwargs)
        shutil.rmtree(user_contexts[request.session_hash].workspace)
        os.makedirs(user_contexts[request.session_hash].workspace, exist_ok=True)

    def render_workspace():

        with gr.Tab("Workspace"):
            workspace_files = gr.Files()
            workspace_files.upload(upload_file, inputs=[workspace_files])
            workspace_files.delete(delete_file, inputs=[workspace_files])
            workspace_files.clear(clear_file, inputs=[workspace_files])

        return workspace_files

    with gr.Blocks(fill_height=True) as demo:

        title, clear_session_btn = render_left_bar()
        welcome = render_right_bar()

        with gr.Tab("Agent"):

            python_code = gr.Code(
                render=False, label="FTL Automation", language="python", visible=True
            )
            python_code.change(persist_python_code, inputs=[python_code])
            playbook_code = gr.Code(
                render=False, label="Ansible playbook", language="yaml", visible=True
            )
            playbook_code.change(persist_playbook_code, inputs=[playbook_code])
            inventory_text = gr.Code(
                render=False, label="Inventory", language="yaml", visible=True
            )
            inventory_text.change(persist_inventory_text, inputs=[inventory_text])
            with gr.Row():
                with gr.Column():
                    system_design_field = gr.Textbox(
                        label="System Design", render=False
                    )
                    system_design_field.change(
                        persist_system_design, inputs=[system_design_field]
                    )
                    tool_check_boxes = gr.CheckboxGroup(
                        choices=sorted(tool_classes),
                        label="Tools",
                        render=False,
                    )

                    tool_check_boxes.change(
                        persist_tool_check_boxes, inputs=[tool_check_boxes]
                    )

                    chatbot = gr.Chatbot(
                        label="FTL Agent",
                        type="messages",
                        resizeable=True,
                        scale=1,
                    )
                    chatbot.change(persist_chat, inputs=[chatbot])
                    gr.ChatInterface(
                        fn=bot,
                        type="messages",
                        chatbot=chatbot,
                        additional_inputs=[
                            system_design_field,
                            tool_check_boxes,
                        ],
                        additional_inputs_accordion=gr.Accordion(
                            label="Additional Inputs", open=False, render=False
                        ),
                        additional_outputs=[python_code, playbook_code, inventory_text],
                        textbox=gr.MultimodalTextbox(
                            file_types=[".conf"],
                            stop_btn=True,
                        ),
                        save_history=False,
                    )

                with gr.Column():

                    current_question_input = gr.Textbox(visible=False)

                    @gr.render(inputs=current_question_input)
                    def render_form(request: gr.Request, *args, **kwargs):

                        context = user_contexts[request.session_hash]
                        print("render_form")
                        print(args)
                        print(kwargs)
                        if persistent_sessions[request.session_hash].get("user_input"):
                            if (
                                "questions"
                                not in context.state
                            ):
                                context.state[
                                    "questions"
                                ] = []
                            for question, answer in persistent_sessions[
                                request.session_hash
                            ]["user_input"].items():
                                if (
                                    question
                                    not in context.state[
                                        "questions"
                                    ]
                                ):
                                    context.state[
                                        "questions"
                                    ].append(question)
                                context.state["user_input"][
                                    question
                                ] = answer
                        print(context.state["questions"])
                        print(context.state["user_input"])
                        if context.state["questions"]:
                            gr.Markdown("### Please answer the following questions:")
                            inputs = []
                            for q in context.state[
                                "questions"
                            ]:
                                if (
                                    q
                                    in context.state[
                                        "user_input"
                                    ]
                                ):
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
                            answer_button = gr.Button("Submit", scale=0)
                            clear_button = gr.Button("Clear", scale=0)

                            def answer_questions(request: gr.Request, *args, **kwargs):
                                print(args)
                                print(kwargs)
                                persistent_sessions[request.session_hash][
                                    "user_input"
                                ] = {}
                                for question, answer in zip(
                                    context.state[
                                        "questions"
                                    ],
                                    args,
                                ):
                                    user_contexts[request.session_hash].state[
                                        "user_input"
                                    ][question] = answer
                                    persistent_sessions[request.session_hash][
                                        "user_input"
                                    ][question] = answer

                            def clear_questions(request: gr.Request, *args, **kwargs):
                                persistent_sessions[request.session_hash][
                                    "user_input"
                                ] = {}
                                user_contexts[request.session_hash].state[
                                    "questions"
                                ] = []

                            answer_button.click(answer_questions, inputs=inputs)

                            clear_button.click(clear_questions, inputs=inputs)

                    def update_questions(request: gr.Request):
                        return user_contexts[request.session_hash].state["questions"]

                    def update_inventory(request: gr.Request):
                        inventory_text = ""
                        if os.path.exists(
                            user_contexts[request.session_hash].inventory
                        ):
                            with open(
                                user_contexts[request.session_hash].inventory
                            ) as f:
                                inventory_text = f.read()
                        return inventory_text

                    gr.Timer(1).tick(
                        fn=update_questions, outputs=current_question_input
                    )
                    gr.Timer(1).tick(fn=update_inventory, outputs=inventory_text)

                    # python_code.render()
                    playbook_code.render()
                    inventory_text.render()

        workspace_files = render_workspace()

        clear_session_btn.click(
            clear_session,
            inputs=None,
            outputs=[
                title,
                system_design_field,
                current_question_input,
                chatbot,
                python_code,
                playbook_code,
                inventory_text,
            ],
        )
        demo.load(
            initialize,
            inputs=None,
            outputs=[
                welcome,
                title,
                system_design_field,
                current_question_input,
                chatbot,
                python_code,
                playbook_code,
                inventory_text,
            ],
        )
        demo.unload(cleanup)

    app = gr.mount_gradio_app(app, demo, path="/ftl", auth_dependency=get_user)

    uvicorn.run(app, host="0.0.0.0", port=7860)


@click.command()
@click.option("--tools-files", "-f", default=["ftl_tools.tools"], multiple=True)
@click.option("--model", "-m", default="ollama_chat/deepseek-r1:14b")
@click.option("--modules", "-M", default=["ftl_modules"], multiple=True)
@click.option("--llm-api-base", default=None)
def main(
    tools_files,
    model,
    modules,
    llm_api_base,
):
    """A agent that solves a problem given a system design and a set of tools"""

    tool_classes = {}
    tool_classes.update(TOOLS)
    for tf in tools_files:
        tool_classes.update(load_tools(tf))
    model = create_model(model, llm_api_base=llm_api_base)
    modules_resolved = []
    for modules_path_or_package in modules:
        modules_path = resolve_modules_path_or_package(modules_path_or_package)
        modules_resolved.append(modules_path)

    launch(model, tool_classes, modules)
