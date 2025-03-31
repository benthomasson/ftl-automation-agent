from smolagents.tools import Tool
from ftl_automation_agent.tools import get_json_schema
from ftl_automation_agent.local_python_executor import FinalAnswerException


class Complete(Tool):
    name = "complete"

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
    name = "user_input"
    description = "Asks for user's input on a specific question"
    inputs = {
        "question": {"type": "string", "description": "The question to ask the user"}
    }
    output_type = "string"

    def __init__(self, state, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def forward(self, question):
        if question not in self.state["user_input"]:
            user_input = input(f"{question} => Type your answer here:")
            self.state["user_input"][question] = user_input
        return self.state["user_input"][question]


class InputTool(Tool):
    name = "input"
    description = "Asks for user's input on a specific question"
    inputs = {
        "question": {"type": "string", "description": "The question to ask the user"}
    }
    output_type = "string"

    def __init__(self, state, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def forward(self, question):
        if question not in self.state["user_input"]:
            user_input = input(f"{question} => Type your answer here:")
            self.state["user_input"][question] = user_input
        return self.state["user_input"][question]


TOOLS = {
    "complete": Complete,
    "impossible": Impossible,
    "user_input": UserInputTool,
    "input": InputTool,
}
