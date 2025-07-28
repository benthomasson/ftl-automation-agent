

class Secret(object):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "***SECRET REDACTED***"

    def __str__(self):
        return self.value

