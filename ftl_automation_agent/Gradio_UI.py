#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Originally from First_agent_template/Gradio_UI.py.
# Modifications for ftl-automation-agent
#


import gradio as gr
import re
from .codegen import (
    generate_python_tool_call,
    generate_explain_action_step,
    generate_playbook_task,
)
from typing import Optional

from smolagents.agent_types import (
    AgentAudio,
    AgentImage,
    AgentText,
    handle_agent_output_types,
)
from smolagents.memory import MemoryStep
from ftl_automation_agent.memory import ActionStep


def pull_messages_from_step(
    step_log: MemoryStep,
):
    """Extract ChatMessage objects from agent steps with proper nesting"""

    print(f"1. {type(step_log)=}")
    if isinstance(step_log, ActionStep):
        # Output the step number
        step_number = (
            f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        )
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            print("model_output")
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove think tokens
            model_output = re.sub(r"<think>", "", model_output)  # handles <think>
            model_output = re.sub(r"</think>", "", model_output)  # handles </think>
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(
                r"```\s*<end_code>", "```", model_output
            )  # handles ```<end_code>
            model_output = re.sub(
                r"<end_code>\s*```", "```", model_output
            )  # handles <end_code>```
            model_output = re.sub(
                r"```\s*\n\s*<end_code>", "```", model_output
            )  # handles ```\n<end_code>
            model_output = model_output.strip()
            print(f"{model_output=}")
            yield gr.ChatMessage(role="assistant", content=model_output)

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            print("tool_calls")
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Tool call becomes the parent message with timing info
            # First we will handle arguments based on type
            args = first_tool_call.arguments
            print(f"2. {type(args)=}")
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(
                    r"```.*?\n", "", content
                )  # Remove existing code blocks
                content = re.sub(
                    r"\s*<end_code>\s*", "", content
                )  # Remove end_code tags
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            parent_message_tool = gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}",
                    "id": parent_id,
                    "status": "pending",
                },
            )
            yield parent_message_tool

            # Nesting execution logs under the tool call if they exist
            if hasattr(step_log, "observations") and (
                step_log.observations is not None and step_log.observations.strip()
            ):  # Only yield execution logs if there's actual content
                log_content = step_log.observations.strip()
                if log_content:
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    yield gr.ChatMessage(
                        role="assistant",
                        content=f"{log_content}",
                        metadata={
                            "title": "📝 Execution Logs",
                            "parent_id": parent_id,
                            "status": "done",
                        },
                    )

            # Nesting any errors under the tool call
            if hasattr(step_log, "error") and step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant",
                    content=str(step_log.error),
                    metadata={
                        "title": "💥 Error",
                        "parent_id": parent_id,
                        "status": "done",
                    },
                )

            # Update parent message metadata to done status without yielding a new message
            parent_message_tool.metadata["status"] = "done"

        # Handle standalone errors but not from tool calls
        elif hasattr(step_log, "error") and step_log.error is not None:
            print("error")
            yield gr.ChatMessage(
                role="assistant",
                content=str(step_log.error),
                metadata={"title": "💥 Error"},
            )

        # Calculate duration and token information
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(
            step_log, "output_token_count"
        ):
            print("input_token_count")
            token_str = f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            step_footnote += token_str
        if hasattr(step_log, "duration"):
            print("duration")
            step_duration = (
                f" | Duration: {round(float(step_log.duration), 2)}"
                if step_log.duration
                else None
            )
            step_footnote += step_duration
        step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
        yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
        yield gr.ChatMessage(role="assistant", content="-----")


def stream_to_gradio(
    agent,
    context,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""

    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(
        task, stream=True, reset=reset_agent_memory, additional_args=additional_args
    ):
        if isinstance(step_log, ActionStep):
            generate_explain_action_step(context.explain, step_log)
            if step_log.tool_calls:
                for call in step_log.tool_calls:
                    generate_python_tool_call(step_log, context.output, call)
            generate_playbook_task(context.playbook, step_log)
        # Track tokens if model provides them
        if hasattr(agent.model, "last_input_token_count"):
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            print(f"3. {type(step_log)=}")
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        for message in pull_messages_from_step(
            step_log,
        ):
            yield message

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    print(f"4. {type(final_answer)=}")
    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(
            role="assistant", content=f"**Final answer:** {str(final_answer)}"
        )


__all__ = ["stream_to_gradio"]
