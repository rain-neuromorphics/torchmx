import gc
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from torchmx.config import MXConfig, QAttentionConfig, QLinearConfig
from torchmx.quant_api import quantize_llm_
from torchmx.utils import set_seed

# This set's all the random seeds to a fixed value. For reproducibility.
set_seed(95134)


def cleanup_memory() -> None:
    """Run GC and clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(model_name: str):
    print(f"Loading model {model_name}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|eot_id|>"]})
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # This ensures the model is loaded in BFloat16
        attn_implementation="eager",
        device_map=device,
    )
    print("Model loaded successfully.")
    print(f"Model before quantization:\n{model}")
    input(
        "Look at the GPU memory and note it down. After quantization, it will drop! Press Enter to continue..."
    )
    print("Quantizing model using torchmx...")

    qattention_config = QAttentionConfig(
        projection_config=QLinearConfig(
            weights_config=MXConfig(
                elem_dtype_name="int8",
            ),
            activations_config=MXConfig(
                elem_dtype_name="int8",
            ),
        ),
        query_config=MXConfig(
            elem_dtype_name="int8",
        ),
        key_config=MXConfig(
            elem_dtype_name="int8",
        ),
        value_config=MXConfig(
            elem_dtype_name="int8",
        ),
        attention_weights_config=MXConfig(
            elem_dtype_name="int8",
        ),
    )

    qmlp_config = QLinearConfig(
        weights_config=MXConfig(
            elem_dtype_name="int8",
        ),
        activations_config=MXConfig(
            elem_dtype_name="int8",
        ),
    )
    # Quantize the LLM module with torchmx. If you use quantize_linear_, it will only
    # replace the nn.Linear layer. Thus the attention mechanism will NOT be quantized.
    # quantize_llm_ will replace the LlamaAttention layer itself thereby quantizing the
    # attention mechanism too If you are interested, try out quantize_linear_ as below.
    # quantize_linear_(model=model, qconfig=qmlp_config)
    quantize_llm_(
        model=model, qattention_config=qattention_config, qmlp_config=qmlp_config
    )
    print(f"Model after quantization:\n{model}")
    print("Quantization complete.")
    print("Cleaning up GPU memory...")
    # This is needed to clean up the GPU memory after quantization.
    cleanup_memory()

    # Run inference on the model
    model.eval()
    print("Compiling model using torch.compile() ...")
    print("Model compiled successfully.")
    # if we do model = torch.compile(model=model, backend=rain_backend), it will
    # run on our hardware.
    model = torch.compile(model=model, backend="inductor")
    print("Starting Chat... Ctrl+C to exit the chat.")
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() == "exit":
                print("\nExiting chat...")
                break

            with torch.inference_mode():
                chat_template = [
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant. Be concise in your answers and avoid unnecessary details.",
                    },
                    {"role": "user", "content": user_input},
                ]
                prompt = tokenizer.apply_chat_template(
                    chat_template, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                # Set up the streamer
                streamer = TextIteratorStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True
                )
                generation_kwargs = dict(
                    inputs,
                    streamer=streamer,
                    max_new_tokens=1000,
                    pad_token_id=tokenizer.eos_token_id,
                )
                thread = Thread(target=model.generate, kwargs=generation_kwargs)

                thread.start()
                print("\nModel: ", end="", flush=True)
                for new_text in streamer:
                    print(new_text, end="", flush=True)
                thread.join()
                cleanup_memory()
                print("\n-----------------------------------------------\n")
        except KeyboardInterrupt:
            print("Chat ended! Exiting...")
            break


if __name__ == "__main__":
    # Set the model name here
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = "meta-llama/Llama-2-7b-hf"
    main(model_name=model_name)
