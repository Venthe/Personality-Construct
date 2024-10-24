def system_message(content):
    return {"role": "system", "content": content}


def assistant_message(content, eos=True):
    return {"role": "assistant", "content": content, "eos": eos}


def user_message(content):
    return {"role": "user", "content": content}
