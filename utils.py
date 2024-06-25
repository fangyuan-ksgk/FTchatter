# Test with Maria conversation
# Test with the served Finetune checkpoints
from openai import OpenAI

def formattting_query_prompt_func_with_sys(prompt, sys_prompt,
                                           tokenizer,
                                           completion = "####Dummy-Answer"):
    """ 
    Formatting response according to specific llama3 chat template
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion}
    ]
    format_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    query_prompt = format_prompt.split(completion)[0]
    return query_prompt

def formatting_query_prompt(message_history, 
                             sys_prompt,
                             tokenizer,
                             completion = "####Dummy-Answer"):
    """ 
    Formatting response according to specific llama3 chat template
    """
    messages = [{"role":"system", "content":sys_prompt}]
    messages.extend(message_history)
    messages.extend([{"role": "assistant", "content": completion}])
    format_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    query_prompt = format_prompt.split(completion)[0]
    return query_prompt


def detect_incomplete_issue(response):
    """
    Check if the response ends with common sentence-ending punctuation or emojis.
    If not, the response is considered incomplete.
    """
    import re
    
    # Define patterns for sentence-ending punctuation and emojis
    punctuation_pattern = r'[.?!]+'
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    # Strip whitespace from the end of the response
    response = response.strip()
    
    # Check if the response ends with punctuation or emoji
    is_complete = (response.endswith(('...', '.', '?', '!')) or
                   re.search(punctuation_pattern, response[-3:]) is not None or
                   emoji_pattern.search(response[-2:]) is not None)
    
    return not is_complete

def get_response_from_finetune_checkpoint(format_prompt, do_print=True, temperature=0.0, max_tokens=512):
    """
    - Using vLLM to serve the fine-tuned Llama3 checkpoint
    - v6 full precision checkpoint is adopted here
    """
    # Serving bit of the client
    client = OpenAI(api_key="EMPTY", base_url="http://43.218.77.178:8000/v1")    
    model_name = "Ksgk-fy/ecoach_philippine_v7_merge"
    # Streaming bit of the client
    stream = client.completions.create(
                model=model_name,
                prompt=format_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=[],
                stream=True,
                extra_body={
                    "repetition_penalty": 1.1,
                    "length_penalty": 1.0,
                    "min_tokens": 0,
                },
            )

    if do_print:
        print("Maria: ", end="")
    response_text = ""
    for response in stream:
        txt = response.choices[0].text
        if txt == "\n":
            continue
        if do_print:
            print(txt, end="")
        response_text += txt
    response_text += "\n"
    print("")
    return response_text

def patch_incomplete_response(format_prompt, initial_response, do_print=True):
    """
    Patch incomplete responses by generating with a higher temperature
    Untill the sentence is completed
    """
    if detect_incomplete_issue(initial_response):
        # Continue generation with higher temperature
        continuation = get_response_from_finetune_checkpoint(
            format_prompt + initial_response,
            do_print=do_print,
            temperature=0.7,
            max_tokens=100  # Adjust as needed
        )
        return initial_response.strip() + continuation.strip()
    return initial_response


def get_response_with_patch(format_prompt, do_print=True):
    """
    Get the response with patching
    """
    initial_response = get_response_from_finetune_checkpoint(format_prompt, do_print=do_print)
    patched_response = patch_incomplete_response(format_prompt, initial_response, do_print=do_print)
    return patched_response

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Ksgk-fy/genius_v2_merge")