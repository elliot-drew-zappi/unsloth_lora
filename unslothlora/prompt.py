llama3_chatml_system = "<|start_header_id|>{}<|end_header_id|>{}<|eot_id|>"

def formatting_prompt_func(messages, prompt_format = llama3_chatml):
    # single conversation scope
    formatted_messages = []
    for message in messages:
        formatted_message = prompt_format.format(message['role'], message['content'])
        formatted_messages.append(formatted_message)
    
    return "".join(formatted_messages)

