import os
import yaml


def generate_python_header(
    output, system_design, problem, tools_files, tools, inventory, modules, extra_vars
):

    with open(output, "w") as f:
        f.write("#!/usr/bin/env python3\n")
        f.write(f'"""\nSystem Design: {system_design}\n')
        if problem:
            f.write(f'Problem:{problem}\n')
        f.write('"""\n')
        f.write("import ftl_automation_agent\n")
        f.write("import os\n\n\n")
        f.write("with ftl_automation_agent.automation(\n")
        f.write(f"tools_files={tools_files},\n")
        f.write(f"tools={tools},\n")
        f.write(f"inventory='{inventory}',\n")
        f.write(f"modules={modules},\n")
        for e in extra_vars:
            e, _, _ = e.partition("=")
            f.write(f"{e.lower()} = os.environ['{e.upper()}'],\n")
        f.write(") as ftl:\n\n")

        for t in tools:
            f.write(f"    {t} = ftl.tools.{t}\n")

        f.write("\n")


def generate_python_tool_call(output, call):
    with open(output, "a") as f:
        f.write("\n    ")
        f.write("\n    ".join(call.arguments.strip().split("\n")))
        f.write("\n")


def reformat_python(output):
    os.system("black " + output)


def generate_explain_header(explain, system_design, problem):
    with open(explain, "w") as f:
        f.write(f"System design: {system_design}\n\n")
        if problem:
            f.write("Problem: {problem}\n\n")


def generate_explain_action_step(explain, o):
    if o.model_output:
        with open(explain, "a") as f:
            f.write(f"Step {o.step_number:2d} ")
            f.write("-" * 100)
            f.write("\n\n")
            f.write(o.model_output)
            f.write("\n\n")


def generate_playbook_header(playbook, system_design, problem):
    with open(playbook, "w") as f:
        if not problem:
            problem = system_design
        header = {"name": problem, "hosts": "all", "gather_facts": False, "tasks": []}
        f.write(yaml.dump([header]))


def generate_playbook_task(playbook, o):
    if not o.trace:
        return
    with open(playbook, "r") as f:
        data = yaml.safe_load(f.read())
    for fn in o.trace:
        name = fn['func_name']
        if name == "complete":
            continue
        kwargs = fn['kwargs']
        data[0]["tasks"].append({name: kwargs})
    with open(playbook, "w") as f:
        f.write(yaml.dump(data))


def add_lookup_plugins(playbook):
    """
    This needs a better solution.
    A second pass isn't terrible, but I need a way to map additional
    arguments.
    """
    with open(playbook, "r") as f:
        data = yaml.safe_load(f.read())

    for play in data:
        for task in play.get('tasks', []):
            if 'authorized_key' in task:
                args = task['authorized_key']
                if 'key' in args and 'lookup' not in args['key']:
                    args['key'] = '{{ lookup("file", "' + args['key'] + '" ) }}'
            if 'slack' in task:
                args = task['slack']
                args['token'] = "{{ lookup('ansible.builtin.env', 'SLACK_TOKEN') }}"
            if 'discord' in task:
                args = task['discord']
                args['webhook_token'] = "{{ lookup('ansible.builtin.env', 'DISCORD_TOKEN') }}"
                args['webhook_id'] = "{{ lookup('ansible.builtin.env', 'DISCORD_CHANNEL') }}"

    with open(playbook, "w") as f:
        f.write(yaml.dump(data))

