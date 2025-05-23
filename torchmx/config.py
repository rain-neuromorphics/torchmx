from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from . import dtypes


class _BaseConfig(ABC):
    @classmethod
    @abstractmethod
    def load_from_dict(cls, config_dict: dict) -> Any:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass


@dataclass(frozen=True)
class MXConfig(_BaseConfig):
    """
    Configuration class for MX Quantization

    Args:
        elem_dtype_name (str): The name of the element dtype. Look at the `name` \
              attribute in dtypes.py for supported strings.
        block_size (int): The block size. Default 32

    Note:
        Pass either elem_dtype or elem_dtype_name and not both.

    Methods:
        __post_init__(): Validates the configuration parameters after initialization.
    """

    elem_dtype_name: str
    block_size: int = 32

    def __post_init__(self):
        if self.elem_dtype_name not in dtypes.STR_TO_SUPPORTED_ELEM_DTYPE:
            raise ValueError(
                f"Unsupported element dtype name: {self.elem_dtype_name}. "
                f"Supported names are: {tuple(dtypes.STR_TO_SUPPORTED_ELEM_DTYPE.keys())}"
            )
        if self.block_size < 1:
            raise ValueError(f"Block size must be at least 1, got {self.block_size}")

    @property
    def elem_dtype(self) -> dtypes.DType:
        """
        Get the DType object corresponding to elem_dtype_name.

        Returns:
            dtypes.DType: The corresponding dtypes.DType object
        """
        return dtypes.STR_TO_SUPPORTED_ELEM_DTYPE[self.elem_dtype_name]

    @classmethod
    def load_from_dict(cls, config_dict: dict) -> "MXConfig":
        """
        Load the configuration from a dictionary.

        Args:
            config_dict (dict): The configuration dictionary.

        Returns:
            MXConfig: The configuration object.
        """
        return cls(**config_dict)

    def __eq__(self, other: "MXConfig") -> bool:
        if not isinstance(other, MXConfig):
            return False
        return all(
            (
                self.elem_dtype_name == other.elem_dtype_name,
                self.block_size == other.block_size,
            )
        )

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        return {
            "elem_dtype_name": self.elem_dtype_name,
            "block_size": self.block_size,
        }


@dataclass(frozen=True)
class QLinearConfig(_BaseConfig):
    """Linear layer Quantization Configuration

    Args:
        weights_config (MXConfig): Configuration for the weights
        activations_config (MXConfig): Configuration for the activations
    """

    weights_config: MXConfig
    activations_config: MXConfig

    @classmethod
    def load_from_dict(cls, config_dict: dict) -> "QLinearConfig":
        """
        Load the configuration from a dictionary.

        Args:
            config_dict (dict): The configuration dictionary.

        Returns:
            QLinearConfig: The configuration object.
        """
        return cls(
            weights_config=MXConfig.load_from_dict(config_dict["weights_config"]),
            activations_config=MXConfig.load_from_dict(
                config_dict["activations_config"]
            ),
        )

    def __eq__(self, other: "QLinearConfig") -> bool:
        if not isinstance(other, QLinearConfig):
            return False
        return all(
            (
                self.weights_config == other.weights_config,
                self.activations_config == other.activations_config,
            )
        )

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        return {
            "weights_config": self.weights_config.to_dict(),
            "activations_config": self.activations_config.to_dict(),
        }


@dataclass(frozen=True)
class QAttentionConfig(_BaseConfig):
    """Attention layer Quantization Configuration

    Args:
        projection_config (QLinearConfig): Configuration for the projection layers. Generally q,k,v,o projection layers.
        query_config (Optional[MXConfig]): Configuration for the query tensor. Default None
        key_config (Optional[MXConfig]): Configuration for the key tensor. Default None
        value_config (Optional[MXConfig]): Configuration for the value tensor. Default None
        attention_weights_config (Optional[MXConfig]): Configuration for the attention weights which is the output of torch.matmul(q,k.T) operation. Default None
    """

    projection_config: QLinearConfig
    query_config: Optional[MXConfig] = None
    key_config: Optional[MXConfig] = None
    value_config: Optional[MXConfig] = None
    attention_weights_config: Optional[MXConfig] = None

    @property
    def is_qkv_quantization_enabled(self) -> bool:
        """
        Check if q,k,v and attention_weights quantization is enabled.

        Returns:
            bool: True if q,k,v and attention_weights quantization is enabled else False
        """
        return all(
            (
                self.query_config,
                self.key_config,
                self.value_config,
                self.attention_weights_config,
            )
        )

    def __post_init__(self):
        _together_configs = (
            (
                self.query_config,
                self.key_config,
                self.value_config,
                self.attention_weights_config,
            ),
        )
        if any(_together_configs):
            assert all(
                _together_configs
            ), "Either all or none of the q,k,v and attention_weights config must be provided"

    @classmethod
    def load_from_dict(cls, config_dict: dict) -> "QAttentionConfig":
        """
        Load the configuration from a dictionary.

        Args:
            config_dict (dict): The configuration dictionary.

        Returns:
            QAttentionConfig: The configuration object.
        """
        if not config_dict.get("query_config", None):
            return cls(
                projection_config=QLinearConfig.load_from_dict(
                    config_dict["projection_config"]
                )
            )
        # If query_config is present, then all the other configs must be present
        return cls(
            projection_config=QLinearConfig.load_from_dict(
                config_dict["projection_config"]
            ),
            query_config=MXConfig.load_from_dict(config_dict["query_config"]),
            key_config=MXConfig.load_from_dict(config_dict["key_config"]),
            value_config=MXConfig.load_from_dict(config_dict["value_config"]),
            attention_weights_config=MXConfig.load_from_dict(
                config_dict["attention_weights_config"]
            ),
        )

    def __eq__(self, other: "QAttentionConfig") -> bool:
        if not isinstance(other, QAttentionConfig):
            return False
        return all(
            (
                self.projection_config == other.projection_config,
                self.query_config == other.query_config,
                self.key_config == other.key_config,
                self.value_config == other.value_config,
                self.attention_weights_config == other.attention_weights_config,
            )
        )

    def to_dict(self):
        """
        Convert the configuration to a dictionary.

        Returns:
            dict: The configuration dictionary.
        """
        result = {}
        result["projection_config"] = self.projection_config.to_dict()
        if (
            self.query_config
            or self.key_config
            or self.value_config
            or self.attention_weights_config
        ):
            result["query_config"] = self.query_config.to_dict()
            result["key_config"] = self.key_config.to_dict()
            result["value_config"] = self.value_config.to_dict()
            result["attention_weights_config"] = self.attention_weights_config.to_dict()
        return result
