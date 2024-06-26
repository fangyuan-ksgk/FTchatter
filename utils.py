# Test with Maria conversation
# Test with the served Finetune checkpoints
from openai import OpenAI
import random
import warnings
from patch import *
warnings.filterwarnings("ignore", category=FutureWarning)

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
                stop=["<|im_end|>"],
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
    if do_print:
        print("")
    return response_text

def patch_incomplete_response(format_prompt, initial_response, do_print=True):
    """
    Patch incomplete responses by generating with a higher temperature
    Untill the sentence is completed
    """
    if detect_incomplete_issue(initial_response):
        # print("--- Incomplete Detected ---")
        # Continue generation with higher temperature
        continuation = get_response_from_finetune_checkpoint(
            format_prompt + initial_response,
            do_print=do_print,
            temperature=0.7,
            max_tokens=100  # Adjust as needed
        )
        return initial_response.strip() + continuation.strip()
    return initial_response


def continue_from_prefix(format_prompt, prefix, do_print):
    continuation = get_response_from_finetune_checkpoint(
        format_prompt + prefix,
        do_print=do_print,
        temperature=0.7,
        max_tokens=120  # Adjust as needed
    )
    return prefix.strip() + continuation.strip()


# def get_response_with_patch(format_prompt, do_print=True):
#     """
#     Get the response with patching
#     """
#     initial_response = get_response_from_finetune_checkpoint(format_prompt, do_print=do_print)
#     patched_response = patch_incomplete_response(format_prompt, initial_response, do_print=do_print)
#     return patched_response

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Ksgk-fy/genius_v2_merge")


####################################
# OOC is Multi-Hop Reasoning Issue #
####################################
# Enhancing OOC Handling in Training Data would help strengthen the Association towards Non-OOC Behavior 



ok_responses = [
    "Hello",
    "Hi, how are you today?",
    "Hi there",
    "Pull it together Alex, you are here to sell insurance right? 😆"
]


from semantic_router import Route
from semantic_router.encoders import CohereEncoder
from semantic_router.layer import RouteLayer

# Initialize the encoder and routes
encoder = CohereEncoder()

ooc_query_route = Route(
    name="OOC_Query",
    utterances=ooc_queries
)

ooc_response_route = Route(
    name="OOC_Response",
    utterances=ooc_responses
)

ok_response_route = Route(
    name="OK_Response",
    utterances=ok_responses
)

routes = [ooc_query_route, ooc_response_route, ok_response_route]
rl = RouteLayer(routes=routes, encoder=encoder)

def detect_ooc_query(utterance):
    """
    Detect if the query is out-of-character (OOC)
    """
    return rl(utterance).name == "OOC_Query"

# def detect_ooc_response(utterance):
#     """
#     Detect if the response is out-of-character (OOC)
#     """
#     return rl(utterance).name == "OOC_Response"

def handle_ooc_query(query):
    """
    Handle OOC queries
    """
    return "Are you ok? You are the sales here bro..."


######################
# OOC Patching Model #
######################

import torch 
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the OOC-Patching Model from HuggingFace repo
repo_id = "Ksgk-fy/ooc_patch_v1"

# Load the tokenizer
patch_tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Load the model
patch_model = AutoModelForSequenceClassification.from_pretrained(repo_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
patch_model = patch_model.to(device)
label_map = {0: "Agent", 1: "Customer"}


def predict_em(sample_text, model = patch_model, tokenizer = patch_tokenizer, threshold_agent=0, threshold_customer=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tokenize the sample text
    encoded = tokenizer(sample_text, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_confidence = torch.nn.functional.softmax(logits, dim=1)[0][predicted_class].item()

    # Map the predicted class to the corresponding label
    label_map = {0: "Agent", 1: "Customer"}
    thres_map = {0: threshold_agent, 1: threshold_customer}
    if predicted_confidence < thres_map[predicted_class]:
        return "Not Sure", 0
    else:
        predicted_label = label_map[predicted_class]
        return predicted_label, predicted_confidence
    

def detect_ooc_response(utterance):
    """
    Detect if the response is out-of-character (OOC)
    """
    return predict_em(utterance, patch_model, patch_tokenizer, 0.8, 0.)[0] == "Agent"



def regenerate_response(format_prompt, initial_response, do_print=True, max_attempts=3):
    """
    Regenerate response if OOC is detected
    - Random Choice of OOC Patch (Prefix which has low-propability of being OOC response)
    - Continue generation from the OOC Patch
    """
    

    # for _ in range(max_attempts):
    if not detect_ooc_response(initial_response):
        return initial_response
    
    initial_prefix = random.choice(ooc_patch)
    initial_response = continue_from_prefix(format_prompt, initial_prefix, do_print=False)
    if do_print:
        print("Maria: ", initial_response)
    
    return initial_response  # Return the last attempt even if it's still OOC



def get_response_with_patch(format_prompt, do_print=True, max_attempts=3):
    """
    Get the response with patching for OOC and incomplete responses
    """
    initial_response = get_response_from_finetune_checkpoint(format_prompt, do_print=False)
    
    # First, handle OOC responses
    patched_response = regenerate_response(format_prompt, initial_response, do_print=False)
    
    # Then, handle incomplete responses
    final_response = patch_incomplete_response(format_prompt, patched_response, do_print=False)
    
    # Print only the final response if do_print is True
    if do_print:
        print("Maria: ", final_response)
    
    return final_response
