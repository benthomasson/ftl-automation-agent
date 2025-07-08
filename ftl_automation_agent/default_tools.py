from smolagents.tools import Tool
from ftl_automation_agent.tools import get_json_schema
from ftl_automation_agent.local_python_executor import FinalAnswerException

from rich.prompt import Prompt
import time


class Complete(Tool):
    name = "complete"
    module = None

    def __init__(self, state, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def forward(self, message: str = "Task was completed"):
        """
        Mark the solution as complete.

        Args:
            message: A completion message
        """

        raise FinalAnswerException(message)

    description, inputs, output_type = get_json_schema(forward)


class Impossible(Tool):
    name = "impossible"
    module = None

    def __init__(self, state, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def forward(self, message: str = "Task was impossible"):
        """
        Mark the solution as impossible

        Args:
            message: A message explaining why the task was impossible
        """

        raise FinalAnswerException(message)

    description, inputs, output_type = get_json_schema(forward)


class UserInputTool(Tool):
    name = "user_input_tool"
    description = "Asks for user's input on a specific question"
    inputs = {
        "question": {"type": "string", "description": "The question to ask the user"}
    }
    output_type = "string"
    module = None

    def __init__(self, state, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def forward(self, question):
        if question not in self.state["user_input"]:
            user_input = Prompt.ask(f"{question} => Type your answer here:")
            self.state["user_input"][question] = user_input
        return self.state["user_input"][question]


class InputTool(Tool):
    name = "input_tool"
    description = "Asks for user's input on a specific question"
    inputs = {
        "question": {"type": "string", "description": "The question to ask the user"}
    }
    output_type = "string"
    module = None

    def __init__(self, state, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def forward(self, question):
        if question not in self.state["user_input"]:
            user_input = Prompt.ask(f"{question} => Type your answer here:")
            self.state["user_input"][question] = user_input
        return self.state["user_input"][question]


class GradioUserInputTool(Tool):
    name = "gradio_input_tool"
    description = "Asks for user's input on a specific question"
    inputs = {
        "question": {"type": "string", "description": "The question to ask the user"}
    }
    output_type = "string"
    module = None

    def __init__(self, state, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def forward(self, question):
        if question not in self.state["questions"]:
            self.state["questions"].append(question)
        while question not in self.state["user_input"]:
            print(f'Waiting on user input for question: {question}')
            time.sleep(1)
        return self.state["user_input"][question]


class PlanningUserInputTool(Tool):
    name = "planning_input_tool"
    description = "Asks for user's input on a specific question"
    inputs = {
        "question": {"type": "string", "description": "The question to ask the user"}
    }
    output_type = "string"
    module = None

    def __init__(self, state, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def forward(self, question):
        if question not in self.state["planning_questions"]:
            self.state["planning_questions"].append(question)
        while question not in self.state["planning_input"]:
            print(f'Waiting on user input for question: {question}')
            time.sleep(1)
        return self.state["planning_input"][question]


class ApprovalTool(Tool):
    name = "approval_tool"
    description = "Asks the user for approval of the plan"
    inputs = {
        "question": {"type": "string", "description": "The question to ask the user's approval for"}
    }
    output_type = "string"
    module = None

    def __init__(self, state, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def forward(self, question):

        if not self.state.get("plan"):
            raise Exception("Not plan found.  Use the submit_plan_tool to send a plan to the user")
        if question not in self.state["planning_approval_questions"]:
            self.state["planning_approval_questions"].append(question)
        while question not in self.state["planning_approvals"]:
            print(f'Waiting on user input for planning approval question: {question}')
            time.sleep(1)
        return self.state["planning_approvals"][question]


class SubmitPlanTool(Tool):
    name = "submit_plan_tool"
    description = "Sends the plan to the user. The plan should be in markdown format."
    inputs = {
        "plan": {"type": "string", "description": "The plan to send to the user."}
    }
    output_type = "string"
    module = None

    def __init__(self, state, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def forward(self, plan):
        self.state["plan"] = plan


TOOLS = {
    "complete": Complete,
    "impossible": Impossible,
    "user_input_tool": UserInputTool,
    "input_tool": InputTool,
    "gradio_input_tool": GradioUserInputTool,
    "planning_input_tool": PlanningUserInputTool,
    "approval_tool": ApprovalTool,
    "submit_plan_tool": SubmitPlanTool,
}
