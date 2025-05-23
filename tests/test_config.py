import pytest

from torchmx import dtypes
from torchmx.config import MXConfig, QAttentionConfig, QLinearConfig


@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("qkv_quantize", [True, False])
def test_load_from_dict(elem_dtype: dtypes.DType, qkv_quantize: bool):
    # testing for QAttentionConfig, recursivesly tests for others too
    if qkv_quantize:
        expected_qconfig = QAttentionConfig(
            projection_config=QLinearConfig(
                weights_config=MXConfig(
                    elem_dtype_name=elem_dtype.name,
                    block_size=32,
                ),
                activations_config=MXConfig(
                    elem_dtype_name=elem_dtype.name,
                    block_size=32,
                ),
            ),
            query_config=MXConfig(
                elem_dtype_name=elem_dtype.name,
                block_size=32,
            ),
            key_config=MXConfig(
                elem_dtype_name=elem_dtype.name,
                block_size=32,
            ),
            value_config=MXConfig(
                elem_dtype_name=elem_dtype.name,
                block_size=32,
            ),
            attention_weights_config=MXConfig(
                elem_dtype_name=elem_dtype.name,
                block_size=32,
            ),
        )

        qconfig = QAttentionConfig.load_from_dict(
            {
                "projection_config": {
                    "weights_config": {
                        "elem_dtype_name": elem_dtype.name,
                        "block_size": 32,
                    },
                    "activations_config": {
                        "elem_dtype_name": elem_dtype.name,
                        "block_size": 32,
                    },
                },
                "query_config": {
                    "elem_dtype_name": elem_dtype.name,
                    "block_size": 32,
                },
                "key_config": {
                    "elem_dtype_name": elem_dtype.name,
                    "block_size": 32,
                },
                "value_config": {
                    "elem_dtype_name": elem_dtype.name,
                    "block_size": 32,
                },
                "attention_weights_config": {
                    "elem_dtype_name": elem_dtype.name,
                    "block_size": 32,
                },
            }
        )
    else:
        expected_qconfig = QAttentionConfig(
            projection_config=QLinearConfig(
                weights_config=MXConfig(
                    elem_dtype_name=elem_dtype.name,
                    block_size=32,
                ),
                activations_config=MXConfig(
                    elem_dtype_name=elem_dtype.name,
                    block_size=32,
                ),
            )
        )

        qconfig = QAttentionConfig.load_from_dict(
            {
                "projection_config": {
                    "weights_config": {
                        "elem_dtype_name": elem_dtype.name,
                        "block_size": 32,
                    },
                    "activations_config": {
                        "elem_dtype_name": elem_dtype.name,
                        "block_size": 32,
                    },
                }
            }
        )

    assert qconfig == expected_qconfig


@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("qkv_quantize", [True, False])
def test_to_dict(elem_dtype: dtypes.DType, qkv_quantize: bool):
    # testing it for QAttentionConfig, recursivesly tests for others too
    if qkv_quantize:
        qconfig = QAttentionConfig(
            projection_config=QLinearConfig(
                weights_config=MXConfig(
                    elem_dtype_name=elem_dtype.name,
                    block_size=32,
                ),
                activations_config=MXConfig(
                    elem_dtype_name=elem_dtype.name,
                    block_size=32,
                ),
            ),
            query_config=MXConfig(
                elem_dtype_name=elem_dtype.name,
                block_size=32,
            ),
            key_config=MXConfig(
                elem_dtype_name=elem_dtype.name,
                block_size=32,
            ),
            value_config=MXConfig(
                elem_dtype_name=elem_dtype.name,
                block_size=32,
            ),
            attention_weights_config=MXConfig(
                elem_dtype_name=elem_dtype.name,
                block_size=32,
            ),
        )

        expected_dict = {
            "projection_config": {
                "weights_config": {
                    "elem_dtype_name": elem_dtype.name,
                    "block_size": 32,
                },
                "activations_config": {
                    "elem_dtype_name": elem_dtype.name,
                    "block_size": 32,
                },
            },
            "query_config": {
                "elem_dtype_name": elem_dtype.name,
                "block_size": 32,
            },
            "key_config": {
                "elem_dtype_name": elem_dtype.name,
                "block_size": 32,
            },
            "value_config": {
                "elem_dtype_name": elem_dtype.name,
                "block_size": 32,
            },
            "attention_weights_config": {
                "elem_dtype_name": elem_dtype.name,
                "block_size": 32,
            },
        }

    else:
        qconfig = QAttentionConfig(
            projection_config=QLinearConfig(
                weights_config=MXConfig(
                    elem_dtype_name=elem_dtype.name,
                    block_size=32,
                ),
                activations_config=MXConfig(
                    elem_dtype_name=elem_dtype.name,
                    block_size=32,
                ),
            )
        )

        expected_dict = {
            "projection_config": {
                "weights_config": {
                    "elem_dtype_name": elem_dtype.name,
                    "block_size": 32,
                },
                "activations_config": {
                    "elem_dtype_name": elem_dtype.name,
                    "block_size": 32,
                },
            }
        }

    assert qconfig.to_dict() == expected_dict
