# Part 4: Creating the Position-wise Feed-Forward Layer

Welcome to Part 4 of our NumPy Transformer series! After implementing Scaled Dot-Product Attention (Part 2) and Multi-Head Attention (Part 3), we now turn to another essential component of the Transformer block: the **Position-wise Feed-Forward Network (FFN)**.

## 4.1 What is a Position-wise Feed-Forward Network?

The Position-wise Feed-Forward Network is a relatively simple component found in each block of the Transformer's encoder and decoder. Its role is to introduce non-linearity and further process the output of the attention sub-layer.

Key characteristics:

1.  **Position-wise:** This means the FFN is applied to each position (e.g., each token or word representation in a sequence) independently and identically. The same network (with the same learned weights) is used for every position, but it doesn't share information across different positions within the same FFN application.
2.  **Fully Connected:** It consists of two linear transformations with a non-linear activation function in between.
3.  **Fixed Structure:** The structure is consistent across all positions and all Transformer blocks (though the weights are learned separately for each block).

## 4.2 Mathematical Formulation

The FFN takes an input `x` (which is typically the output of the preceding Multi-Head Attention sub-layer, after residual connection and layer normalization) and transforms it as follows:

`FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2`

Let's break this down:

1.  **First Linear Transformation:**
    *   `x @ W_1 + b_1`
    *   `x` is the input to the FFN for a specific position, with dimension `d_model` (the model's hidden size).
    *   `W_1` is the weight matrix of the first linear layer, with shape `(d_model, d_ff)`.
    *   `b_1` is the bias vector of the first linear layer, with shape `(d_ff,)`.
    *   `d_ff` is the inner-layer dimensionality, often referred to as the feed-forward dimension. Typically, `d_ff` is larger than `d_model` (e.g., `d_ff = 4 * d_model` as in the original Transformer paper).
    *   The output of this layer has dimension `d_ff`.

2.  **Non-linear Activation (ReLU):**
    *   `max(0, ...)`
    *   A Rectified Linear Unit (ReLU) activation function is applied element-wise to the output of the first linear layer.
    *   `ReLU(z) = max(0, z)`
    *   This introduces non-linearity, allowing the model to learn more complex functions.

3.  **Second Linear Transformation:**
    *   `... @ W_2 + b_2`
    *   The output of the ReLU activation (dimension `d_ff`) is then passed through a second linear layer.
    *   `W_2` is the weight matrix of the second linear layer, with shape `(d_ff, d_model)`.
    *   `b_2` is the bias vector of the second linear layer, with shape `(d_model,)`.
    *   The output of this layer, and thus the output of the FFN, has dimension `d_model`. This brings the representation back to the model's main hidden size, allowing it to be used by subsequent layers or blocks.

When processing a sequence of tokens (e.g., shape `(batch_size, seq_len, d_model)`), this entire FFN operation is applied to each `d_model`-dimensional vector at every position in the sequence independently.

## 4.3 NumPy Implementation

We'll implement the Position-wise Feed-Forward Network as a class.

```python
import numpy as np

class PositionwiseFeedForward:
    def __init__(self, d_model, d_ff, seed=None):
        """
        Initialize the Position-wise Feed-Forward Network.

        Args:
            d_model (int): Dimension of the input and output.
            d_ff (int): Dimension of the inner layer (feed-forward dimension).
            seed (int, optional): Random seed for weight initialization.
        """
        self.d_model = d_model
        self.d_ff = d_ff

        if seed is not None:
            np.random.seed(seed)

        # Weight matrix for the first linear transformation (d_model -> d_ff)
        # Glorot/Xavier initialization: scale by sqrt(6 / (fan_in + fan_out))
        # For simplicity, using randn here. In practice, better initialization is key.
        limit_w1 = np.sqrt(6.0 / (d_model + d_ff))
        self.W_1 = np.random.uniform(-limit_w1, limit_w1, (d_model, d_ff))
        self.b_1 = np.zeros(d_ff)

        # Weight matrix for the second linear transformation (d_ff -> d_model)
        limit_w2 = np.sqrt(6.0 / (d_ff + d_model))
        self.W_2 = np.random.uniform(-limit_w2, limit_w2, (d_ff, d_model))
        self.b_2 = np.zeros(d_model)

    def relu(self, x):
        """Rectified Linear Unit activation function."""
        return np.maximum(0, x)

    def forward(self, x):
        """
        Perform the forward pass for the Position-wise Feed-Forward Network.

        Args:
            x (np.ndarray): Input tensor, shape (batch_size, seq_len, d_model).

        Returns:
            np.ndarray: Output tensor, shape (batch_size, seq_len, d_model).
        """
        # Input x: (batch_size, seq_len, d_model)

        # First linear transformation
        # x @ W_1: (batch_size, seq_len, d_model) @ (d_model, d_ff) -> (batch_size, seq_len, d_ff)
        # b_1 is broadcasted: (d_ff,) -> (1, 1, d_ff)
        hidden_output = np.matmul(x, self.W_1) + self.b_1

        # ReLU activation
        activated_output = self.relu(hidden_output)

        # Second linear transformation
        # activated_output @ W_2: (batch_size, seq_len, d_ff) @ (d_ff, d_model) -> (batch_size, seq_len, d_model)
        # b_2 is broadcasted: (d_model,) -> (1, 1, d_model)
        output = np.matmul(activated_output, self.W_2) + self.b_2

        return output

```

**Explanation of the Code:**

1.  **`__init__(self, d_model, d_ff, seed=None)`:**
    *   Stores `d_model` (input/output dimension) and `d_ff` (inner feed-forward dimension).
    *   Initializes weight matrices `W_1` (shape `(d_model, d_ff)`) and `W_2` (shape `(d_ff, d_model)`), and bias vectors `b_1` (shape `(d_ff,)`) and `b_2` (shape `(d_model,)`).
    *   For weight initialization, I've used a simple uniform distribution scaled by a factor derived from Glorot/Xavier initialization (`np.sqrt(6.0 / (fan_in + fan_out))`). Biases are initialized to zeros. In a full deep learning framework, these initializations are often handled automatically with more options.

2.  **`relu(self, x)`:**
    *   A simple implementation of the ReLU activation function using `np.maximum(0, x)`.

3.  **`forward(self, x)`:**
    *   Takes an input `x` of shape `(batch_size, seq_len, d_model)`.
    *   **First Linear Layer:** Computes `np.matmul(x, self.W_1) + self.b_1`. NumPy's broadcasting handles adding `b_1` (shape `(d_ff,)`) to the result of the matrix multiplication (shape `(batch_size, seq_len, d_ff)`).
    *   **ReLU Activation:** Applies the `relu` function to the output of the first layer.
    *   **Second Linear Layer:** Computes `np.matmul(activated_output, self.W_2) + self.b_2`. Again, `b_2` (shape `(d_model,)`) is broadcast correctly.
    *   Returns the final `output` of shape `(batch_size, seq_len, d_model)`.

## 4.4 Simple Input/Output Examples

Let's test our `PositionwiseFeedForward` class.

```python
# Example Usage of PositionwiseFeedForward

# Parameters
batch_size_ex = 2
seq_len_ex = 3    # Sequence length
d_model_ex = 8    # Model dimension (input/output of FFN)
d_ff_ex = 32      # Inner feed-forward dimension (e.g., 4 * d_model)

# Initialize PositionwiseFeedForward layer
pffn = PositionwiseFeedForward(d_model=d_model_ex, d_ff=d_ff_ex, seed=77)

# Dummy input data (e.g., output from a Multi-Head Attention layer)
np.random.seed(102)
input_tensor = np.random.rand(batch_size_ex, seq_len_ex, d_model_ex)

print("--- Position-wise Feed-Forward Network Example ---")
print(f"Input tensor shape: {input_tensor.shape}")

# Forward pass
output_pffn = pffn.forward(input_tensor)

print(f"\nOutput PFFN shape: {output_pffn.shape}")

# Verify output shape: (batch_size, seq_len, d_model)
assert output_pffn.shape == (batch_size_ex, seq_len_ex, d_model_ex)

print("\nSample of Input Tensor (first batch, first token, first 4 features):\n", input_tensor[0, 0, :4])
print("\nSample of Output PFFN (first batch, first token, first 4 features):\n", output_pffn[0, 0, :4])

# Check if the computation is position-wise
# If we process two different positions from the same batch item separately,
# using only their respective d_model vectors, the result for that part should be the same.

# Process position 0 of batch 0
input_pos0_batch0 = input_tensor[0:1, 0:1, :] # Shape (1, 1, d_model)
output_pos0_batch0_isolated = pffn.forward(input_pos0_batch0)

print("\nVerifying position-wise nature:")
print("Output for [0,0] from full batch:", output_pffn[0,0,:4])
print("Output for [0,0] processed isolatedly:", output_pos0_batch0_isolated[0,0,:4])

# They should be very close (potential minor floating point differences if any, but should be identical here)
assert np.allclose(output_pffn[0,0,:], output_pos0_batch0_isolated[0,0,:])
print("Position-wise check successful.")
```

**Running the Example:**

If you save the Python code (the `PositionwiseFeedForward` class and the example usage) into a `.py` file and run it, you will see the shapes of the input and output tensors. The example also includes a small check to conceptually verify the "position-wise" nature of the computation by processing a single position vector and comparing it to the corresponding part of the full batch output.

## 4.5 Key Takeaways

*   The Position-wise Feed-Forward Network consists of two linear transformations with a ReLU activation in between.
*   It is applied independently to each position in the input sequence.
*   It introduces non-linearity and allows for further processing of features after the attention mechanism.
*   The input and output dimensions are typically `d_model`, while the inner dimension `d_ff` is larger.

## 4.6 What's Next?

So far, we have implemented:
1.  Scaled Dot-Product Attention
2.  Multi-Head Attention
3.  Position-wise Feed-Forward Network

Before we can assemble these into a full Transformer Encoder block, we need one more crucial piece: **Positional Encoding**. Since Transformers don't inherently process sequences in order (unlike RNNs), we need to explicitly provide information about the position of each token in the sequence. This will be the focus of Part 5.

Stay tuned!
