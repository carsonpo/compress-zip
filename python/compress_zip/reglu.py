"""
ReGLU (Rectified Gated Linear Unit) activation.

Matches CUDA reglu kernel in reglu.cu.
"""

import numpy as np
from .primitives import sra_rne_tte_s32, clamp_i8


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
    assert gate.shape == value.shape, "Gate and value must have same shape"

    gate_i32 = gate.astype(np.int32)
    value_i32 = value.astype(np.int32)

    output = np.zeros_like(gate, dtype=np.int8)

    # Flatten for iteration
    gate_flat = gate_i32.flatten()
    value_flat = value_i32.flatten()
    output_flat = output.flatten()

    for i in range(len(gate_flat)):
        g = gate_flat[i]
        v = value_flat[i]

        # ReLU on gate
        if g <= 0:
            output_flat[i] = 0
        else:
            # g * v is Q0.14, shift by 7 to get Q0.7
            product = g * v
            result = sra_rne_tte_s32(product, 7)
            output_flat[i] = clamp_i8(result)

    return output_flat.reshape(gate.shape)


def reglu_i8_vectorized(gate: np.ndarray, value: np.ndarray) -> np.ndarray:
    """
    Vectorized ReGLU (faster, but may have minor rounding differences).

    For bit-exact results, use reglu_i8.
    """
    gate_i32 = gate.astype(np.int32)
    value_i32 = value.astype(np.int32)

    # ReLU on gate
    gate_relu = np.maximum(gate_i32, 0)

    # Product and shift
    product = gate_relu * value_i32

    # Simple rounding (not ties-to-even, but close enough for most cases)
    # For bit-exact, need element-wise sra_rne_tte_s32
    result = (product + 64) >> 7  # Add 0.5 in Q0.7 for rounding

    return np.clip(result, -128, 127).astype(np.int8)


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
