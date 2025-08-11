FORMAT = {}


_ALLOWED_KEYS = ["annotation", "_id"]


def name(name=""):
    def wrapper(func):
        if name in FORMAT and FORMAT[name] != func:
            raise KeyError("Duplicated name")
        FORMAT[name] = func
        return func
    return wrapper


def format(item):
    return {
        "image": [],
        "prompt": "",
        "response": "",
        "answer": ""
    }
    

@name("default")
def default_format(item):
    return {
        "image": item["images"],
        "prompt": item["conversations"][0]["value"].replace("<image>", '').strip(),
        "response": item["result"],
        "answer": item["chosen"]["value"],
        **{k: item[k] for k in _ALLOWED_KEYS if k in item}
    }
    

def get_format(name):
    return FORMAT[name]