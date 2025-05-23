"""A simple example to quantize Llama model downloaded from HF using TorchMX and show
the generated AtenIR graph.
"""

import argparse
import os

import torch
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified
from torch.fx import GraphModule
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchmx.config import MXConfig, QAttentionConfig, QLinearConfig
from torchmx.quant_api import quantize_llm_
from torchmx.utils import get_logger

logger = get_logger("llama_example")

OUTPUT_FILE = None


def toy_backend(gm: GraphModule, sample_inputs):
    if not OUTPUT_FILE:
        raise ValueError("Please provide an output file to write the AtenIR graph")

    def my_compiler(gm: GraphModule, sample_inputs):
        # <implement your compiler here>
        logger.info("AOTAutograd produced a fx Graph in Aten IR.")
        printable_graph = gm.print_readable(print_output=False)
        # append to the file since, this is called multiple times if there are graph breaks
        with open(OUTPUT_FILE, "a") as f:
            f.write(printable_graph)
        logger.info(f"AtenIR Graph saved to {OUTPUT_FILE}")
        return gm.forward

    # Invoke AOTAutograd
    return aot_module_simplified(gm, sample_inputs, fw_compiler=my_compiler)


def get_causal_model_and_tokenizer(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    model_kwargs: dict = {
        "attn_implementation": "eager",
    },
    tokenizer_kwargs: dict = {},
):
    """
    Get model and tokenizer from model name.

    Args:
        model_name (str): Model name.
        torch_dtype (torch.dtype, optional): Torch dtype to be used. Defaults to torch.bfloat16.
        model_kwargs (dict, optional): Model kwargs to be passed to transformers.AutoModelForCausalLM. Defaults to {
            "attn_implementation": "eager",
        }.
        tokenizer_kwargs (dict, optional): Tokenizer kwargs to be passed to transformers.AutoTokenizer. Defaults to {}.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Model and Tokenizer.
    """
    logger.info(f"Loading the causal model and it's tokenizer for: {model_name}")
    logger.info(f"Using torch_dtype: {torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    return model, tokenizer


def main():
    # Model name
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
    logger.info(f"Device: {DEVICE}")
    model_name = "meta-llama/Llama-2-7b-hf"
    # Model and tokenizer
    model, tokenizer = get_causal_model_and_tokenizer(
        model_name, torch_dtype=torch.bfloat16
    )
    model = model.to(DEVICE)
    # Model config
    model_config = model.config
    logger.info(f"Model config: {model_config}")
    logger.info(f"Model before quantization:\n{model}")
    logger.info("----------------------------------------")
    # Quantization config
    # Quantize the model
    qattention_config = QAttentionConfig(
        projection_config=QLinearConfig(
            weights_config=MXConfig(
                elem_dtype_name="float6_e3m2",
                block_size=32,
            ),
            activations_config=MXConfig(
                elem_dtype_name="float8_e4m3",
                block_size=16,
            ),
        ),
        query_config=MXConfig(
            elem_dtype_name="float8_e4m3",
        ),
        key_config=MXConfig(
            elem_dtype_name="float4_e2m1",
        ),
        value_config=MXConfig(
            elem_dtype_name="float4_e2m1",
        ),
        attention_weights_config=MXConfig(
            elem_dtype_name="float6_e2m3",
        ),
    )

    qmlp_config = QLinearConfig(
        weights_config=MXConfig(
            elem_dtype_name="float6_e2m3",
            block_size=32,
        ),
        activations_config=MXConfig(
            elem_dtype_name="float8_e4m3",
            block_size=32,
        ),
    )
    quantize_llm_(
        model=model, qattention_config=qattention_config, qmlp_config=qmlp_config
    )
    logger.info(f"Model after quantization:\n{model}")

    prompt = ["Welcome to TorchMX! In the mission to make LLMs go Brrrrrrrrr"]
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    explanation = torch._dynamo.explain(model.forward)(**inputs)
    logger.info(f"Number of Graph Breaks in model: {explanation.graph_break_count}")
    logger.info(f"Number of Graphs in model: {explanation.graph_count}")
    for reason in explanation.break_reasons:
        logger.info(f"Graph Break Reason:\n{reason}")
    logger.info("-------------------------------------")

    torch._dynamo.reset()
    logger.info("Compiling the model...")

    # FullGraph is a much stricter mode that requires the entire graph to be compiled
    # Right now graph breaks due to logging.warning in transformers
    # Potential fix: https://github.com/pytorch/pytorch/pull/139403
    model.forward = torch.compile(
        model=model.forward, backend=toy_backend, fullgraph=True
    )
    with torch.inference_mode():
        # triggers compilation of forward graph on the first run
        _ = model.generate(**inputs, max_new_tokens=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default="llama_example_atenir_gen.py",
        help="File to write the generated AtenIR. Default: llama_example_atenir_gen.py",
    )
    args = parser.parse_args()
    OUTPUT_FILE = os.path.abspath(args.output_file)
    if os.path.exists(OUTPUT_FILE):
        logger.warning(f"Removing existing output_file: {OUTPUT_FILE}")
        os.remove(OUTPUT_FILE)

    logger.info(f"AtenIR Graph will be saved to: {OUTPUT_FILE}")
    main()
