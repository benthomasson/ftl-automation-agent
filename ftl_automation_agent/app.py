import os

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
from ftl_automation_agent.Gradio_UI import agent_stream_to_gradio, planning_stream_to_gradio

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import FastAPI, Depends, Request
from starlette.config import Config
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
import uvicorn

from pprint import pprint

from collections import UserDict


console = Console()


TASK_PROMPT = """
You are a task running agent.

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

PLANNING_PROMPT = """

You are a planning agent. You create plans for other agents to execute.
You are working as a consultant engineer with a customer who wants to deploy
some software.  Work collaboratively with them to plan how to deploy this
system.  You are given some tools to help in your planning.

Use the planning_input_tool to ask for additional information.
Use the complete() tool to signal that you are done

"""


def load_sessions(sub):
    session_file_name = os.path.join("/sessions", sub, "sessions.json")
    if os.path.exists(session_file_name):
        with open(session_file_name) as f:
            data = json.loads(f.read())
        return data
    else:
        return {}


def load_session(sub, i):
    session_file_name = os.path.join("/sessions", sub, str(i), "session.json")
    if os.path.exists(session_file_name):
        with open(session_file_name) as f:
            data = json.loads(f.read())
        return data
    else:
        return {}


def save_session(sub, i, data):
    session_file_name = os.path.join("/sessions", sub, str(i), "session.json")
    os.makedirs(os.path.dirname(session_file_name), exist_ok=True)
    with open(session_file_name, "w") as f:
        f.write(json.dumps(data))


def save_sessions(sub, data):
    session_file_name = os.path.join("/sessions", sub, "sessions.json")
    os.makedirs(os.path.dirname(session_file_name), exist_ok=True)
    with open(session_file_name, "w") as f:
        f.write(json.dumps(data))


class SecretsView(UserDict):

    def __init__(self, secrets):
        self.secrets = secrets

    def __getitem__(self, key):
        for k, v in self.secrets:
            if key == k:
                return v
        raise KeyError(key)


def launch(model, tool_classes, tools_files, modules_resolved, modules):

    app = FastAPI()

    current_sessions = defaultdict(dict)
    session_histories = defaultdict(dict)
    persistent_sessions = defaultdict(dict)
    user_contexts = dict()

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

    def agent_bot(
        prompt, messages, playbook_name, system_design, tools, request: gr.Request
    ):
        print(f"{prompt=}")
        print(f"{messages=}")
        print(f"{playbook_name=}")
        print(f"{system_design=}")
        print(f"{tools=}")
        print(f"{request=}")

        context = user_contexts[request.session_hash]

        if isinstance(prompt, dict):
            prompt = prompt["text"]

        full_prompt = TASK_PROMPT + prompt + "\nThen call complete() when done."
        agent = make_agent(
            tools=[get_tool(tool_classes, t, context.state) for t in tools],
            model=model,
        )

        playbook_prefix, _ = os.path.splitext(playbook_name)
        context.output = os.path.join(context.outputs, f"{playbook_prefix}.py")
        context.explain = os.path.join(context.outputs, f"{playbook_prefix}.txt")
        context.playbook = os.path.join(context.outputs, f"{playbook_prefix}.yml")
        context.user_input = os.path.join(
            context.outputs, f"{playbook_prefix}-user_input.yml"
        )
        user_input_file = os.path.basename(context.user_input)
        inventory_file = os.path.basename(context.inventory)
        generate_python_header(
            context.output,
            system_design,
            prompt,
            tools_files,
            tools,
            inventory_file,
            modules,
            {},  # context.extra_vars,
            user_input_file,
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
        for msg in agent_stream_to_gradio(
            agent, context, task=full_prompt, prompt=prompt, reset_agent_memory=False
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

    def planning_bot(prompt, messages, request: gr.Request):

        print(f"{prompt=}")
        print(f"{messages=}")
        print(f"{request=}")


        context = user_contexts[request.session_hash]

        if isinstance(prompt, dict):
            prompt = prompt["text"]

        full_prompt = PLANNING_PROMPT + prompt + "\nCall complete() when the customer is satified with the plan."

        tools = ['complete', 'impossible', 'planning_input_tool']

        agent = make_agent(
            tools=[get_tool(tool_classes, t, context.state) for t in tools],
            model=model,
        )

        for msg in planning_stream_to_gradio(
            agent, context, task=full_prompt, reset_agent_memory=False
        ):
            messages.append(msg)
            yield messages

        yield messages

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

    with gr.Blocks(title="Welcome to FTL") as login_demo:
        gr.Button("Login", link="/login")

    app = gr.mount_gradio_app(app, login_demo, path="/login-ftl")

    def get_workspace_files(request: gr.Request):
        sub = request.request.session["user"]["sub"]
        current_session = current_sessions[request.session_hash]
        workspace = os.path.join("/workspace", sub, str(current_session))
        workspace_files = glob.glob(os.path.join(workspace, "*"))
        return workspace_files

    def get_output_files(request: gr.Request):
        sub = request.request.session["user"]["sub"]
        current_session = current_sessions[request.session_hash]
        outputs = os.path.join("/outputs", sub, str(current_session))
        output_files = glob.glob(os.path.join(outputs, "*"))
        return output_files

    def initialize(request: gr.Request):
        pprint(request.request.session["user"])
        sub = request.request.session["user"]["sub"]
        sessions_data = load_sessions(sub)
        current_session = sessions_data.get("current_session", 0)
        current_sessions[request.session_hash] = current_session
        sessions = session_histories[request.session_hash] = sessions_data.get(
            "session_history", []
        )
        data = load_session(sub, current_session)
        if len(sessions) == 0:
            sessions.append([data.get("title", f"Session {current_session}")])
        return start_session(request)

    def start_session(request: gr.Request):
        current_session = current_sessions[request.session_hash]
        sub = request.request.session["user"]["sub"]
        sessions_data = load_sessions(sub)
        sessions = session_histories[request.session_hash] = sessions_data.get(
            "session_history", []
        )
        data = load_session(sub, current_session)
        if "secrets" not in data:
            data["secrets"] = []
        pprint(data)
        persistent_sessions[request.session_hash] = data
        workspace = os.path.join("/workspace", sub, str(current_session))
        outputs = os.path.join("/outputs", sub, str(current_session))
        os.makedirs(workspace, exist_ok=True)
        os.makedirs(outputs, exist_ok=True)
        inventory = os.path.join(workspace, "inventory.yml")
        if not os.path.exists(inventory):
            with open(inventory, "w") as f:
                f.write(yaml.dump({}))

        workspace_files = glob.glob(os.path.join(workspace, "*"))
        output_files = glob.glob(os.path.join(outputs, "*"))
        secrets = data.get("secrets", [])
        state = {
            "inventory": ftl.load_inventory(inventory),
            "inventory_file": inventory,
            "localhost": ftl.localhost,
            "modules": modules_resolved,
            "user_input": {},
            "gate": None,
            "loop": None,
            "gate_cache": None,
            "log": None,
            "console": console,
            "questions": [],
            "workspace": workspace,
            "secrets": SecretsView(secrets),
        }
        user_contexts[request.session_hash] = Bunch(
            state=state,
            inventory=inventory,
            outputs=outputs,
            tool_classes=tool_classes,
            workspace=workspace,
        )
        return (
            gr.Dataset(
                samples=sessions,
            ),
            f"Welcome, {request.username}!",
            data.get("title", f"Session {current_session}"),
            data.get("system_design"),
            data.get("user_input"),
            data.get("chat"),
            data.get("python_code"),
            data.get("playbook_code"),
            data.get("playbook_name"),
            data.get("inventory_text"),
            data.get("tool_check_boxes"),
            workspace_files,
            secrets,
            output_files,
        )

    def persist_all(request: gr.Request):
        print("persist_all")
        if request.session_hash in persistent_sessions:
            # pprint(persistent_sessions[request.session_hash])
            current_session = current_sessions.get(request.session_hash, 0)
            print(f"{current_session=}")
            save_session(
                request.request.session["user"]["sub"],
                current_session,
                persistent_sessions[request.session_hash],
            )
            sessions_data = {
                "current_session": current_session,
                "session_history": session_histories.get(request.session_hash, []),
            }
            save_sessions(
                request.request.session["user"]["sub"],
                sessions_data,
            )

    def cleanup(request: gr.Request):
        print("cleanup")
        persist_all(request)
        if request.session_hash in persistent_sessions:
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

    def persist_playbook_name(request: gr.Request, playbook_name):
        persistent_sessions[request.session_hash]["playbook_name"] = playbook_name

    def persist_inventory_text(request: gr.Request, inventory_text):
        persistent_sessions[request.session_hash]["inventory_text"] = inventory_text

    def get_sessions(request: gr.Request):
        return session_histories.get(request.session_hash, [])

    def new_session(sessions, title, system_design_field, request: gr.Request):

        persist_all(request)
        # current_session = current_sessions.get(request.session_hash, 0)
        sessions = session_histories.get(request.session_hash, [])

        next_session = len(sessions)
        current_sessions[request.session_hash] = next_session
        (
            title,
            system_design_field,
            current_question_input,
            agent_chatbot,
            python_code,
            playbook_code,
            playbook_name,
            inventory_text,
            current_secrets,
        ) = clear_session("", request)
        title = f"Session {next_session}"
        sessions.append([title])
        persist_all(request)
        return [
            gr.Dataset(
                samples=sessions,
            ),
            title,
            system_design_field,
            current_question_input,
            agent_chatbot,
            python_code,
            playbook_code,
            playbook_name,
            inventory_text,
            current_secrets,
        ]

    def clear_session(title, request: gr.Request):
        print("clear_session")
        secrets = persistent_sessions[request.session_hash].get("secrets", [])
        data = {"title": title, "system_design": "", "secrets": secrets}
        persistent_sessions[request.session_hash] = data
        context = user_contexts[request.session_hash]
        context.state["questions"] = []
        context.state["user_input"] = {}
        with open(context.inventory, "w") as f:
            f.write(json.dumps({}))
        persist_all(request)
        return (
            data["title"],
            data["system_design"],
            None,
            None,
            None,
            None,
            "playbook.yml",
            None,
            secrets,
        )

    def select_session(event: gr.SelectData, request: gr.Request):
        print(event)
        print(event.index)
        print(event.value)
        print(event.row_value)
        print(event.col_value)
        print(event.selected)
        persist_all(request)
        current_sessions[request.session_hash] = event.index
        return start_session(request)

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
                clear_session_btn = gr.Button("🗑️ Clear session", scale=0)
                new_session_btn = gr.Button(
                    "New session",
                    scale=0,
                    variant="primary",
                    icon=gr.utils.get_icon_path("plus.svg"),
                    visible=True,
                )
                session_list = gr.Dataset(
                    samples=[],
                    components=[gr.Textbox(visible=False)],
                    show_label=False,
                    layout="table",
                    type="tuple",
                    samples_per_page=100,
                )

        return title, clear_session_btn, new_session_btn, session_list

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

        workspace = gr.Tab("Workspace")

        with workspace:
            workspace_files = gr.Files()
            workspace_files.upload(upload_file, inputs=[workspace_files])
            workspace_files.delete(delete_file, inputs=[workspace_files])
            workspace_files.clear(clear_file, inputs=[workspace_files])

        def workspace_selected(workspace_files, request: gr.Request):
            workspace_files = get_workspace_files(request)
            return workspace_files

        workspace.select(
            workspace_selected, inputs=[workspace_files], outputs=[workspace_files]
        )

        return workspace_files

    def render_agent():

        with gr.Tab("Agent"):

            python_code = gr.Code(
                render=False, label="FTL Automation", language="python", visible=True
            )
            playbook_name = gr.Textbox(
                label="Playbook Name:", value="playbook.yml", render=False
            )
            playbook_name.change(persist_playbook_name, inputs=[playbook_name])
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

                    agent_chatbot = gr.Chatbot(
                        label="FTL Agent",
                        type="messages",
                        resizeable=True,
                        scale=1,
                    )
                    agent_chatbot.change(persist_chat, inputs=[agent_chatbot])

                    gr.ChatInterface(
                        fn=agent_bot,
                        type="messages",
                        chatbot=agent_chatbot,
                        additional_inputs=[
                            playbook_name,
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

                        if request.session_hash not in user_contexts:
                            return

                        context = user_contexts[request.session_hash]
                        print("render_form")
                        print(args)
                        print(kwargs)
                        if persistent_sessions[request.session_hash].get("user_input"):
                            if "questions" not in context.state:
                                context.state["questions"] = []
                            for question, answer in persistent_sessions[
                                request.session_hash
                            ]["user_input"].items():
                                if question not in context.state["questions"]:
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
                            answer_button = gr.Button("Submit", scale=0)
                            clear_button = gr.Button("Clear", scale=0)

                            def answer_questions(request: gr.Request, *args, **kwargs):
                                print(args)
                                print(kwargs)
                                persistent_sessions[request.session_hash][
                                    "user_input"
                                ] = {}
                                for question, answer in zip(
                                    context.state["questions"],
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

                    python_code.render()
                    # playbook_name.render()
                    playbook_code.render()
                    inventory_text.render()
        return (
            system_design_field,
            current_question_input,
            agent_chatbot,
            python_code,
            playbook_code,
            playbook_name,
            tool_check_boxes,
            inventory_text,
        )

    def render_automation():
        automation = gr.Tab("Automation")

        with automation:
            output_files = gr.Files(interactive=False)

        def automation_selected(output_files, request: gr.Request):
            output_files = get_output_files(request)
            return output_files

        automation.select(
            automation_selected, inputs=[output_files], outputs=[output_files]
        )

        return output_files

    def render_topology():
        with gr.Tab("Topology"):
            pass

    def render_documents():
        with gr.Tab("Documents"):
            pass

    def render_planning():
        with gr.Tab("Planning"):
            with gr.Column():

                planning_chatbot = gr.Chatbot(
                    placeholder="<strong>FTL Planning</strong><br>What do you want to build today?",
                    label="FTL Planning",
                    type="messages",
                    resizeable=True,
                    scale=1,
                )
                gr.ChatInterface(
                    fn=planning_bot,
                    type="messages",
                    chatbot=planning_chatbot,
                    textbox=gr.MultimodalTextbox(
                        stop_btn=True,
                    ),
                    save_history=False,
                )

            with gr.Column():

                planning_question_input = gr.Textbox(visible=False)


    def render_secrets():

        def persist_secret(secret_state, key_text, value_text, request: gr.Request):
            print(f"{secret_state=} {key_text=} {value_text=}")
            persistent_sessions[request.session_hash]["secrets"][secret_state] = [
                key_text,
                value_text,
            ]
            persist_all(request)

        with gr.Tab("Secrets"):
            with gr.Row():
                with gr.Column(scale=2):
                    current_secrets = gr.Textbox(visible=False)

                    @gr.render(inputs=current_secrets)
                    def render_current_secrets(user_msg, request: gr.Request):
                        for i, (key, value) in enumerate(
                            persistent_sessions[request.session_hash].get("secrets", [])
                        ):
                            with gr.Row():
                                secret_state = gr.State(i)
                                key_text = gr.Textbox(
                                    type="text", value=key, show_label=False
                                )
                                value_text = gr.Textbox(
                                    type="password", value=value, show_label=False
                                )
                                key_text.change(
                                    persist_secret,
                                    inputs=[secret_state, key_text, value_text],
                                )
                                value_text.change(
                                    persist_secret,
                                    inputs=[secret_state, key_text, value_text],
                                )

                    with gr.Row():
                        add_btn = gr.Button("➕ Add New Pair", variant="primary")
                        clear_all_btn = gr.Button("🗑️Clear All", variant="secondary")

            def add_secret(request: gr.Request):
                print("add_secret")
                # if "secrets" not in persistent_sessions[request.session_hash]:
                #    persistent_sessions[request.session_hash]["secrets"] = []
                persistent_sessions[request.session_hash]["secrets"].append(["", ""])
                print(persistent_sessions[request.session_hash]["secrets"])
                return gr.update(
                    value=persistent_sessions[request.session_hash]["secrets"]
                )

            def clear_secrets(request: gr.Request):
                print("add_secret")
                # if "secrets" not in persistent_sessions[request.session_hash]:
                #    persistent_sessions[request.session_hash]["secrets"] = []
                persistent_sessions[request.session_hash]["secrets"].clear()
                print(persistent_sessions[request.session_hash]["secrets"])
                return gr.update(value=[])

            add_btn.click(add_secret, inputs=None, outputs=[current_secrets])
            clear_all_btn.click(clear_secrets, inputs=None, outputs=[current_secrets])

        return current_secrets

    def title_input(title, request: gr.Request):

        sessions = get_sessions(request)
        current_session = current_sessions[request.session_hash]

        if len(sessions) == 0:
            return gr.Dataset(
                samples=sessions,
            )

        sessions[current_session][0] = title

        return gr.Dataset(
            samples=sessions,
        )

    with gr.Blocks(fill_height=True, title="FTL") as demo:

        title, clear_session_btn, new_session_btn, session_list = render_left_bar()
        title.input(title_input, inputs=[title], outputs=[session_list])

        welcome = render_right_bar()

        render_planning()
        (
            system_design_field,
            current_question_input,
            agent_chatbot,
            python_code,
            playbook_code,
            playbook_name,
            tool_check_boxes,
            inventory_text,
        ) = render_agent()

        workspace_files = render_workspace()
        output_files = render_automation()
        render_topology()
        render_documents()
        current_secrets = render_secrets()

        session_list.select(
            select_session,
            inputs=None,
            outputs=[
                session_list,
                welcome,
                title,
                system_design_field,
                current_question_input,
                agent_chatbot,
                python_code,
                playbook_code,
                playbook_name,
                inventory_text,
                tool_check_boxes,
                workspace_files,
                current_secrets,
                output_files,
            ],
            show_api=False,
            queue=False,
            show_progress="hidden",
        )

        clear_session_btn.click(
            clear_session,
            inputs=[title],
            outputs=[
                title,
                system_design_field,
                current_question_input,
                agent_chatbot,
                python_code,
                playbook_code,
                playbook_name,
                inventory_text,
                current_secrets,
            ],
            show_api=False,
            queue=False,
            show_progress="hidden",
        )

        new_session_btn.click(
            new_session,
            inputs=[
                session_list,
                title,
                system_design_field,
            ],
            outputs=[
                session_list,
                title,
                system_design_field,
                current_question_input,
                agent_chatbot,
                python_code,
                playbook_code,
                playbook_name,
                inventory_text,
                current_secrets,
            ],
            show_api=False,
            queue=False,
            show_progress="hidden",
        )
        demo.load(
            initialize,
            inputs=None,
            outputs=[
                session_list,
                welcome,
                title,
                system_design_field,
                current_question_input,
                agent_chatbot,
                python_code,
                playbook_code,
                playbook_name,
                inventory_text,
                tool_check_boxes,
                workspace_files,
                current_secrets,
                output_files,
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

    launch(model, tool_classes, tools_files, modules_resolved, modules)
