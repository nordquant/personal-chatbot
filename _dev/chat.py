from ctransformers import AutoModelForCausalLM


def gguf():
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    # llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral")
    # llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf")
    # llm = AutoModelForCausalLM.from_pretrained("TheBloke/TinyLlama-1.1B-intermediate-step-480k-1T-GGUF", model_file="tinyllama-1.1b-intermediate-step-480k-1t.Q4_K_M.gguf")
    llm = AutoModelForCausalLM.from_pretrained(
        "juanjgit/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )

    # f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    system = "You are an AI assistant that follows instruction extremely well. Help as much as you can. Give short answers."
    instruction = "Which is the biggest city of India?"
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"

    for i in llm(prompt, stream=True):
        print(i)


gguf()


def pipe():
    import torch
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
    )

    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        },
    ]
    # streamer = TextStreamer(tokenizer)

    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(prompt)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, top_k=1)
    print(outputs[0]["generated_text"])
