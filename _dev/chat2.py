from ctransformers import AutoModelForCausalLM

# llm = AutoModelForCausalLM.from_pretrained("juanjgit/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
)


def get_prompt(instruction: str, history=None) -> str:
    system = "You are a chatbot that gives helpful answers."
    if history:
        return f"### System:\n{system}\n\n### User: Use the following pieces of context to answer the question at the end: {history.replace('#', '')}. This is the end of the context. Now answer this question. Give a short answer: \n{instruction}\n\n### Response:\n"
    else:
        return f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"


prompt = "Which is the largest city in India?"

history = prompt
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
    history += word

print()
prompt = get_prompt("Which is the second largest?", history=history)
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
