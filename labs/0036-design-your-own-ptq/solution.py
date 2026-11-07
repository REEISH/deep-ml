# --- cell 1 ---
import numpy as np
# numpy only


def ptq(weights, calib_X):
    """
    Design your own post-training quantization (PTQ) scheme.

    You receive FP32 weights of an MLP trained on real MNIST (Google/TF
    mnist.npz), plus a calibration batch of real flattened digit pixels
    in [0, 1]. Return dequantized weights after your PTQ round-trip.

    Args:
        weights: dict[str, np.ndarray]
            Keys: "W1","b1","W2","b2","W3","b3"
            Shapes: W1 (128,784), b1 (128,), W2 (64,128), b2 (64,), W3 (10,64), b3 (10,)
            Layout: each Wi is (out_features, in_features)
        calib_X: np.ndarray, shape (n_calib, 784)
            Real MNIST train pixels flattened, scaled to [0, 1].

    Returns:
        dequant_weights: dict[str, np.ndarray]
            Same keys/shapes, float dtype — low-bit reconstructions, not FP32 copies.
    """
    # TODO: quantize then dequantize
    dequant_weights = {}
    W_bits = 4
    b_bits = 8
    qmax_W = (2 ** W_bits) - 1
    qmax_b = (2 ** (b_bits - 1)) - 1
    for key, tensor in weights.items():
        if key.startswith("W"):
            min_vals = np.min(tensor, axis=1, keepdims=True)
            max_vals = np.max(tensor, axis=1, keepdims=True)
            scale = (max_vals - min_vals) / qmax_W
            scale = np.where(scale == 0, 1.0, scale)
            zero_point = np.round(-min_vals / scale)
            q_tensor = np.clip(np.round(tensor / scale) + zero_point, 0, qmax_W)
            dequant_weights[key] = (q_tensor - zero_point) * scale
        elif key.startswith("b"):
            max_abs = np.max(np.abs(tensor))
            scale = max_abs / qmax_b if max_abs > 0 else 1.0
            q_tensor = np.clip(np.round(tensor / scale), -qmax_b, qmax_b)
            dequant_weights[key] = q_tensor * scale
    return dequant_weights