from utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
format_prompt = formatting_query_prompt([], "", tokenizer)


cap_rounds = 3 # Maximum number of conversation rounds stored in memory
message_history_base = []
message_history_ft = []

sys_prompt = "Roleplay as a Phillippino named Maria."
message_history_gpt4o = [{"role":"system", "content":sys_prompt}]

i = 0
while i<10:
    user_input = input("You: ")
    print(f"\nAgent: {user_input}")
    message_history_base.append({"role": "user", "content": user_input})
    message_history_ft.append({"role": "user", "content": user_input})
    message_history_gpt4o.append({"role": "user", "content": user_input})
    
    format_prompt_base = formatting_query_prompt(message_history_base, sys_prompt, tokenizer)
    format_prompt_ft = formatting_query_prompt(message_history_ft, sys_prompt, tokenizer)
    # Get response from GPT-4o
    
    print("------")
    response_gpt4o = get_oai_response(message_history_gpt4o)
    print("GPT-4o: ", response_gpt4o)
    print("------")
    response_base = get_response_from_base(format_prompt_base, do_print=True)
    print("------")
    response_ft = get_response_without_patch(format_prompt_ft, do_print=True)

    message_history_base.append({"role":"assistant", "content": response_base})
    message_history_ft.append({"role":"assistant", "content":response_ft})

    message_history_base = message_history_base[-(2*cap_rounds):] # Cap on conversation history stored within memory
    message_history_ft = message_history_ft[-(2*cap_rounds):] # Cap on conversation history stored within memory
    i+=1