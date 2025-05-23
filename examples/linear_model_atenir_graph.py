"""A simple example that quantizes a linear model using TorchMX's MXTensor and compiles
it using AOTAutograd.
"""

import argparse

import torch
import torch._dynamo
import torch.nn as nn
import torchao
from torch._functorch.aot_autograd import aot_module_simplified
from torch.fx import GraphModule
from tqdm import tqdm

from torchmx import dtypes
from torchmx.config import MXConfig, QLinearConfig
from torchmx.quant_api import mx_dynamic_activation_mx_weights, quantize_linear_
from torchmx.utils import get_logger

logger = get_logger("linear_model")


def quantize_model_using_module_(
    model: torch.nn.Module,
    weight_elem_dtype: dtypes.DType = dtypes.float6_e3m2,
    weight_block_size: int = 32,
    activation_elem_dtype: dtypes.DType = dtypes.float8_e4m3,
    activation_block_size: int = 32,
):
    """Quantizes the model using TorchMX's MXLinear layer inplace.

    Args:
        model (torch.nn.Module): Model to quantize
        weight_elem_dtype (types.DType, optional): Weight element dtype. Defaults to torch.float6_e3m2.
        weight_block_size (int, optional): Weight block size. Defaults to 32.
        activation_elem_dtype (types.DType, optional): Activation element dtype. Defaults to torch.float8_e4m3fn.
        activation_block_size (int, optional): Activation block size. Defaults to 32.
    """
    logger.info("Quantizing model using TorchMX MXInferenceLinear")
    qconfig = QLinearConfig(
        weights_config=MXConfig(
            elem_dtype_name=weight_elem_dtype.name, block_size=weight_block_size
        ),
        activations_config=MXConfig(
            elem_dtype_name=activation_elem_dtype.name, block_size=activation_block_size
        ),
    )
    quantize_linear_(model, qconfig=qconfig)
    logger.info(f"Model quantized successfully. Quantized model:\n{model}")
    logger.info("-------------------------------------")


def quantize_model_using_tensor_(
    model: torch.nn.Module,
    weight_elem_dtype: dtypes.DType = dtypes.float6_e3m2,
    weight_block_size: int = 32,
    activation_elem_dtype: dtypes.DType = dtypes.float8_e4m3,
    activation_block_size: int = 32,
):
    """In place quantization of the model with MXFP Dynamic quantization for activations
    and MXFP quantization for weights.

    Args:
        moddel (torch.nn.Module): Model to quantize
        weight_elem_dtype (dtypes.DType, optional): Weight element dtype. Defaults to dtypes.float6_e3m2.
        weight_block_size (int, optional): Weight block size. Defaults to 32.
        activation_elem_dtype (dtypes.DType, optional): Activation element dtype. Defaults to torch.float8_e4m3fn.
        activation_block_size (int, optional): Activation block size. Defaults to 32.
    """
    logger.info("Quantizing model using torchao.quantize_")
    if not torchao.utils.TORCH_VERSION_AT_LEAST_2_5:
        error = (
            "Your torch version : {torch.__version__} is < 2.5.0 consider upgrading!"
        )
        logger.error(error)
        raise ImportError(error)

    # We use TorchAO's stable quantization API to quantize the model
    torchao.quantization.quantize_(
        model,
        mx_dynamic_activation_mx_weights(
            weight_elem_dtype=weight_elem_dtype,
            weight_block_size=weight_block_size,
            activation_elem_dtype=activation_elem_dtype,
            activation_block_size=activation_block_size,
        ),
    )
    logger.info(f"Model quantized successfully. Quantized model:\n{model}")
    logger.info("-------------------------------------")


def toy_backend(gm: GraphModule, sample_inputs):
    def my_compiler(gm: GraphModule, sample_inputs):
        # <implement your compiler here>
        logger.info("=====================================")
        logger.info("\nAOTAutograd produced a fx Graph in Aten IR.")
        logger.info("\nFX Graph in Tabular format:\n")
        gm.graph.print_tabular()
        logger.info("=====================================")
        logger.info("\nFX Graph in Readable format:\n")
        gm.print_readable(colored=True)
        logger.info("=====================================")
        # # Draw the graph
        # g = FxGraphDrawer(gm, name="mxlinear_model")
        # g.get_dot_graph().write_svg("mxlinear_model.svg")
        # logger.info("Graph is saved as mxlinear_model.svg")
        return gm.forward

    # Invoke AOTAutograd
    return aot_module_simplified(gm, sample_inputs, fw_compiler=my_compiler)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int = 32,
        out_features: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self._hidden_size = 128
        self.fc1 = nn.Linear(
            in_features=in_features, out_features=self._hidden_size, dtype=dtype
        )
        self.fc2 = nn.Linear(
            in_features=self._hidden_size, out_features=out_features, dtype=dtype
        )

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        return x


def main(
    quantize: str,
    activation_elem_dtype: dtypes.DType,
    activation_block_size: int,
    weight_elem_dtype: dtypes.DType,
    weight_block_size: int,
    run_count: int = 10000,
):
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"TorchAO version: {torchao.__version__}")

    IN_FEATURES = 64
    OUT_FEATURES = 256
    BATCH_SIZE = 8
    MODEL_DTYPE = torch.bfloat16
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
    logger.info(f"Device: {DEVICE}")

    model = MLP(
        in_features=IN_FEATURES, out_features=OUT_FEATURES, dtype=MODEL_DTYPE
    ).to(DEVICE)
    logger.info(f"A simple MLP model:\n{model}")
    logger.info("-------------------------------------")
    if quantize == "tensor":
        quantize_model_using_tensor_(
            model,
            weight_elem_dtype=weight_elem_dtype,
            weight_block_size=weight_block_size,
            activation_elem_dtype=activation_elem_dtype,
            activation_block_size=activation_block_size,
        )
    elif quantize == "module":
        quantize_model_using_module_(
            model,
            weight_elem_dtype=weight_elem_dtype,
            weight_block_size=weight_block_size,
            activation_elem_dtype=activation_elem_dtype,
            activation_block_size=activation_block_size,
        )
    elif quantize.lower() == "none":
        logger.info("Skipping quantization")
    else:
        raise ValueError(
            f"Invalid quantization option: {quantize}. Choose `module` or `tensor`."
        )
    input_tensor = torch.randn(BATCH_SIZE, IN_FEATURES, dtype=MODEL_DTYPE).to(DEVICE)
    # Let's see if there are graph breaks
    explanation = torch._dynamo.explain(model)(input_tensor)
    logger.info(f"Dynamo Explanation of the model:\n{explanation}")
    logger.info("-------------------------------------")

    torch._dynamo.reset()
    compiled_model = torch.compile(
        model=model, backend=toy_backend, dynamic=True, fullgraph=True
    )

    # # Use this context manager to print only the forward graph. If you remove this, it will
    # # print the backward graph as well.
    with torch.inference_mode():
        # Triggers compilation of forward graph on the first run
        for _ in tqdm(
            range(run_count), total=run_count, desc="Running compiled model: "
        ):
            inputs = torch.randn(BATCH_SIZE, IN_FEATURES, dtype=MODEL_DTYPE).to(DEVICE)
            _ = compiled_model(inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MX Quantize a PyTorch linear model using TorchMX")
    parser.add_argument(
        "--quantize",
        "-q",
        default="module",
        type=str,
        choices=("tensor", "module", "none"),
        help="""Quantize the model to MX formats. Choices are
                `tensor` to quantize the model using torchao.quantize_, `module` to
                quantize using TorchMX's MXInferenceLinear . `none` to skip quantization.
                Default is `module_train`.""",
    )
    parser.add_argument(
        "--activation_elem_dtype",
        "-ae",
        type=str,
        default="float8_e4m3",
        choices=tuple(dtypes.STR_TO_SUPPORTED_ELEM_DTYPE.keys()),
        help="Activation element dtype for quantization. Default is float8_e4m3.",
    )
    parser.add_argument(
        "--activation_block_size",
        "-ab",
        type=int,
        default=32,
        help="Activation block size for quantization. Default is 32.",
    )
    parser.add_argument(
        "--weight_elem_dtype",
        "-we",
        type=str,
        default="float6_e3m2",
        choices=tuple(dtypes.STR_TO_SUPPORTED_ELEM_DTYPE.keys()),
        help="Weight element dtype for quantization. Default is fp6_e3m2.",
    )
    parser.add_argument(
        "--weight_block_size",
        "-wb",
        type=int,
        default=32,
        help="Weight block size for quantization. Default is 32.",
    )
    parser.add_argument(
        "--run_count",
        "-rc",
        type=int,
        default=10000,
        help="Number of times to run the compiled model. Default is 10000.",
    )
    args = parser.parse_args()
    logger.info("Given arguments:")
    logger.info(args)
    main(
        quantize=args.quantize,
        activation_elem_dtype=dtypes.STR_TO_SUPPORTED_ELEM_DTYPE[
            args.activation_elem_dtype
        ],
        activation_block_size=args.activation_block_size,
        weight_elem_dtype=dtypes.STR_TO_SUPPORTED_ELEM_DTYPE[args.weight_elem_dtype],
        weight_block_size=args.weight_block_size,
        run_count=args.run_count,
    )
