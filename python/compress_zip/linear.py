"""
Integer linear (GEMM) layer.

Matches CUDA int8 GEMM implementation.
"""

import numpy as np


def linear_i8_to_i32(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """
    Int8 linear layer returning int32 accumulators.

    Computes y = x @ weight.T + bias

    Args:
        x: Input tensor of shape [in_features] or [batch, in_features] as int8
        weight: Weight tensor of shape [out_features, in_features] as int8
        bias: Optional bias tensor of shape [out_features] as int32

    Returns:
        Output tensor of shape [out_features] or [batch, out_features] as int32 accumulators
    """
    squeeze = False
    if x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze = True

    batch_size, in_features = x.shape
    out_features, weight_in = weight.shape

    assert weight_in == in_features, f"Weight in_features {weight_in} != x in_features {in_features}"

    # Convert to int32 for accumulation
    x_i32 = x.astype(np.int32)
    weight_i32 = weight.astype(np.int32)

    # Matrix multiply
    output = np.zeros((batch_size, out_features), dtype=np.int32)

    for b in range(batch_size):
        for o in range(out_features):
            acc = 0
            for i in range(in_features):
                acc += x_i32[b, i] * weight_i32[o, i]
            output[b, o] = acc

    # Add bias if provided
    if bias is not None:
        output += bias.astype(np.int32)

    if squeeze:
        return output[0]
    return output


def linear_i8_to_i32_vectorized(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """
    Vectorized int8 linear layer (faster, bit-exact for integer ops).
    """
    squeeze = False
    if x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze = True

    # Convert to int32 for accumulation
    x_i32 = x.astype(np.int32)
    weight_i32 = weight.astype(np.int32)

    # Matrix multiply: [batch, in] @ [out, in].T = [batch, out]
    output = x_i32 @ weight_i32.T

    # Add bias if provided
    if bias is not None:
        output += bias.astype(np.int32)

    if squeeze:
        return output[0]
    return output


# Test cases
if __name__ == "__main__":
    # Test basic linear
    x = np.array([[64, 64]], dtype=np.int8)  # [1, 2]
    weight = np.array([
        [64, 64],   # out[0] = 64*64 + 64*64 = 8192
        [32, 32],   # out[1] = 64*32 + 64*32 = 4096
    ], dtype=np.int8)  # [2, 2]

    result = linear_i8_to_i32(x, weight)
    print(f"x: {x}")
    print(f"weight: {weight}")
    print(f"result: {result}")

    assert result[0, 0] == 8192, f"Expected 8192, got {result[0, 0]}"
    assert result[0, 1] == 4096, f"Expected 4096, got {result[0, 1]}"

    # Test with bias
    bias = np.array([100, 200], dtype=np.int32)
    result_with_bias = linear_i8_to_i32(x, weight, bias)
    assert result_with_bias[0, 0] == 8292, f"Expected 8292, got {result_with_bias[0, 0]}"
    assert result_with_bias[0, 1] == 4296, f"Expected 4296, got {result_with_bias[0, 1]}"

    # Verify vectorized matches
    result_vec = linear_i8_to_i32_vectorized(x, weight)
    assert np.array_equal(result, result_vec), "Vectorized doesn't match scalar"

    print("All linear tests passed!")
