"""
Tests how a we can create a reference to host args by using dot notation.
"""


def get_host_path(ref):

    path = []
    while ref._parent is not None:
        path.append(ref._name)
        ref = ref._parent

    return path[::-1]


def get_host_value(host, path):

    value = host
    for part in path:
        value = value[part]

    return value



class Ref(object):

    def __init__(self, parent, name):
        self._parent = parent
        self._name = name

    def __getattr__(self, name):
        ref = Ref(self, name)
        setattr(self, name, ref)
        return ref


class FTL(object):

    def __init__(self, inventory):
        self.inventory = inventory

    @property
    def host(self):
        return Ref(None, "host")


def test_1():

    inventory_data = {
        "all": {
            "hosts": {
                "localhost": {
                    "ansible_host": "localhost",
                    "ansible_connection": "local",
                    "ansible_python_interpreter": "/usr/bin/python3",
                    "a": {"b": {"c": {"name": "foo"}}},
                }
            }
        }
    }

    ftl = FTL(inventory_data)

    r = ftl.host.ansible_host
    path = get_host_path(r)
    assert path == ["ansible_host"]
    v = get_host_value(inventory_data["all"]["hosts"]["localhost"], path)
    assert v == "localhost"

    r = ftl.host.a.b.c.name
    path = get_host_path(r)
    assert path == ["a", "b", "c", "name"]
    v = get_host_value(inventory_data["all"]["hosts"]["localhost"], path)
    assert v == "foo"
