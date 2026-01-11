"""
ReGLU (Rectified Gated Linear Unit) activation.

Matches CUDA reglu kernel in reglu.cu.
"""

import numpy as np
from .primitives import sra_rne_tte_s32_np


def reglu_i8(gate: np.ndarray, value: np.ndarray) -> np.ndarray:
    """
    ReGLU activation: ReLU(gate) * value

    Both inputs are Q0.7 (int8 / 128 = real).
    Product is Q0.14, shift by 7 to get Q0.7 output.

    Args:
        gate: Gate tensor as int8 (Q0.7)
        value: Value tensor as int8 (Q0.7)

    Returns:
        Output tensor as int8 (Q0.7)
    """
    gate_i32 = gate.astype(np.int32)
    value_i32 = value.astype(np.int32)

    # ReLU on gate
    gate_relu = np.maximum(gate_i32, 0)

    # Product and shift with ties-to-even rounding
    product = gate_relu * value_i32
    result = sra_rne_tte_s32_np(product, 7)

    return np.clip(result, -127, 127).astype(np.int8)


# Test cases
if __name__ == "__main__":
    # Test basic ReGLU
    gate = np.array([64, -64, 127, 0], dtype=np.int8)  # 0.5, -0.5, ~1, 0
    value = np.array([64, 64, 64, 64], dtype=np.int8)  # 0.5, 0.5, 0.5, 0.5

    result = reglu_i8(gate, value)
    print(f"gate: {gate}")
    print(f"value: {value}")
    print(f"result: {result}")

    # Expected:
    # [0]: relu(64) * 64 = 64 * 64 = 4096, >> 7 = 32
    # [1]: relu(-64) * 64 = 0 * 64 = 0
    # [2]: relu(127) * 64 = 127 * 64 = 8128, >> 7 = 63
    # [3]: relu(0) * 64 = 0 * 64 = 0

    assert result[0] == 32, f"Expected 32, got {result[0]}"
    assert result[1] == 0, f"Expected 0, got {result[1]}"
    assert result[2] == 63 or result[2] == 64, f"Expected ~63, got {result[2]}"
    assert result[3] == 0, f"Expected 0, got {result[3]}"

    print("All ReGLU tests passed!")
