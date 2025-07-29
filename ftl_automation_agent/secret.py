import logging

logger = logging.getLogger("ftl_automation.secrets")


class Secret(object):
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def __repr__(self):
        return "***SECRET REDACTED***"

    def __str__(self):
        logger.info("Secret %s was accessed", self._name)
        return self._value
