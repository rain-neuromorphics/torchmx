"""Environment variables for the torchmx package."""

import os

# Verbosity level of the log
TORCHMX_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# If set, the logs from this package will be written to the file
# in addition to the console
TORCHMX_LOG_FILE = os.getenv("LOG_FILE", None)


# If set to True, the hardware quantization will be done in exact mode.
# If set to False, the MX quantization is simulated by higher precision
# followed by a normalization step.
MX_EXACT_QUANTIZATION = os.getenv("MX_HARDWARE_EXACT_QUANTIZATION", "False")
