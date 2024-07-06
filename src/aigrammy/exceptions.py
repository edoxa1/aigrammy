class NoGptPromptSpecifiedException(Exception):
    """ Raise for cases where no prompt was specified for ChatGPT"""


class FailedToConvertBinaryToBase64(Exception):
    """ Raise for cases where conversion to b64 was failed """
