from contextlib import contextmanager

from threading import Thread
import asyncio

from smolagents.tools import Tool
from .tools import load_tools, get_tool
from .default_tools import TOOLS
from dataclasses import dataclass
from faster_than_light import load_inventory
from ftl_agent.local_python_executor import FinalAnswerException
from faster_than_light import localhost
from faster_than_light.gate import build_ftl_gate, use_gate
from functools import partial

dependencies = [
"ftl_module_utils @ git+https://github.com/benthomasson/ftl_module_utils@main"
]


class Tools(object):
    def __init__(self, tools: dict[str, Tool]):
        self.__dict__.update(tools)


@dataclass
class FTL:
    tools: Tools


@contextmanager
def automation(tools_files, tools, inventory, modules, **kwargs):
    tool_classes = {}
    tool_classes.update(TOOLS)
    gate_cache = {}
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    tool_modules = tools[:]
    if "complete" in tool_modules:
        tool_modules.remove("complete")
    state = {
        "inventory": load_inventory(inventory),
        "modules": modules,
        "localhost": localhost,
        "gate_cache": gate_cache,
        "loop": loop,
        "gate": partial(
            use_gate,
            *build_ftl_gate(
                modules=tool_modules,
                module_dirs=modules,
                interpreter="/usr/bin/python3",
                dependencies=dependencies,
            )
        ),
    }
    state.update(kwargs)
    for tf in tools_files:
        tool_classes.update(load_tools(tf))
    ftl = FTL(
        tools=Tools({name: get_tool(tool_classes, name, state) for name in tools})
    )
    try:
        yield ftl
    except FinalAnswerException as e:
        print(e)
