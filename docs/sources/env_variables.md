# Environment Variables for the `torchmx` Package

This document describes the environment variables available for configuring the `torchmx` package.

---

## Logging Configuration

### TORCHMX_LOG_LEVEL

- Description: Controls the verbosity level of logs.
- Default Value: `"INFO"`
- Usage:

  ```bash
  export TORCHMX_LOG_LEVEL="INFO"
  ```

### TORCHMX_LOG_FILE

- Description: If set, logs from this package will be written to the specified file in addition to the console.
- Default Value: `None`
- Usage:

  ```bash
  export TORCHMX_LOG_FILE="/path/to/logfile.log"
  ```

---

## Hardware and Computation Settings

### MX_HARDWARE_EXACT_QUANTIZATION

- Description: If set to `True`, hardware quantization will be performed in exact mode.
- Default Value: `"False"`
- Usage:

  ```bash
  export MX_HARDWARE_EXACT_QUANTIZATION="False"
  ```

---
