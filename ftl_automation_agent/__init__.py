import asyncio
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from threading import Thread
from typing import Dict

import yaml
from faster_than_light import load_inventory, localhost
from faster_than_light.gate import build_ftl_gate, use_gate
from faster_than_light.ref import Ref
from smolagents.tools import Tool

from ftl_automation_agent.local_python_executor import FinalAnswerException

from .default_tools import TOOLS
from .tools import get_tool, load_tools
from .util import resolve_modules_path_or_package
from .secret import Secret

from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from textual.widgets import RichLog
from rich.console import Console
console = Console()

dependencies = [
    "ftl_module_utils @ git+https://github.com/benthomasson/ftl_module_utils@main",
    "ftl_collections @ git+https://github.com/benthomasson/ftl-collections@main",
]


class Tools(object):
    def __init__(self, tools: dict[str, Tool]):
        self.__dict__.update(tools)


@dataclass
class FTL:
    tools: Tools
    inventory: Dict
    host: Ref
    console: Console
    progress: Progress
    log: RichLog
    state: Dict


@contextmanager
def automation(tools_files, tools, inventory, modules, user_input=None, log=None, sync=True, secrets=None):
    tool_classes = {}
    tool_classes.update(TOOLS)
    for tf in tools_files:
        tool_classes.update(load_tools(tf))
    gate_cache = {}
    if sync:
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()
    else:
        loop = None
    tool_modules = []
    for tool_name, tool in tool_classes.items():
        if tool_name not in tools:
            continue
        if hasattr(tool, "module"):
            if module := getattr(tool, "module"):
                tool_modules.append(module)
    if user_input is not None:
        with open(user_input) as f:
            user_input = yaml.safe_load(f.read())
    else:
        user_input = {}
    if not os.path.exists(inventory):
        with open(inventory, "w") as f:
            f.write(yaml.dump({}))

    modules_resolved = []
    for modules_path_or_package in modules:
        modules_path = resolve_modules_path_or_package(modules_path_or_package)
        modules_resolved.append(modules_path)

    state = {
        "inventory": load_inventory(inventory),
        "modules": modules_resolved,
        "localhost": localhost,
        "gate_cache": gate_cache,
        "loop": loop,
        "gate": partial(
            use_gate,
            *build_ftl_gate(
                modules=tool_modules,
                module_dirs=modules_resolved,
                interpreter="/usr/bin/python3",
                dependencies=dependencies,
            ),
        ),
        "user_input": user_input,
        "log": log,
        "console": console,
        "questions": [],
        "secrets": {},
    }

    for secret in secrets:
        state["secrets"][secret] = Secret(os.environ[secret])

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
    ) as progress:
        ftl = FTL(
            tools=Tools({name: get_tool(tool_classes, name, state) for name in tools}),
            inventory=inventory,
            host=Ref(None, "host"),
            console=console,
            progress=progress,
            log=log,
            state=state,
        )
        try:
            yield ftl
        except FinalAnswerException as e:
            console.print(e)
