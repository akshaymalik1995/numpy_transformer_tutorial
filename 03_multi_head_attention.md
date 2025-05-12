# Part 3: Building Multi-Head Attention from Scratch

Welcome to Part 3 of our Transformer tutorial! In the previous part, we implemented Scaled Dot-Product Attention. Now, we'll build upon that to create **Multi-Head Attention**, a key component that enhances the Transformer's ability to focus on different parts of the input sequence in different ways.

## 3.1 Why Multi-Head Attention?

Single Scaled Dot-Product Attention allows the model to focus on different parts of the sequence based on the (Query, Key, Value) interactions. However, it might be beneficial for the model to attend to different types of information or different "representation subspaces" simultaneously.

Multi-Head Attention achieves this by running the Scaled Dot-Product Attention mechanism multiple times in parallel, each with different, learned linear projections of the Queries, Keys, and Values. This allows each "head" to learn to focus on different aspects of the input.

**Benefits:**

1.  **Diverse Representations:** It allows the model to jointly attend to information from different representation subspaces at different positions. A single attention head might average away some important information, but multiple heads can capture more nuanced relationships.
2.  **Stabilized Learning:** Averaging or concatenating the outputs of multiple heads can lead to a more stable and robust attention mechanism.
3.  **Increased Model Capacity:** With multiple sets of projection weights, the model has more parameters and potentially a greater capacity to learn complex patterns.

## 3.2 Mathematical Formulation

Multi-Head Attention consists of several attention heads. For each head `i`, the input Queries (Q), Keys (K), and Values (V) are linearly projected using learned weight matrices:

*   `Q_i = Q @ W_i^Q`
*   `K_i = K @ W_i^K`
*   `V_i = V @ W_i^V`

Where `W_i^Q`, `W_i^K`, and `W_i^V` are the weight matrices for the `i`-th head.

Then, Scaled Dot-Product Attention is applied to each projected set of (Q_i, K_i, V_i) to get the output for that head:

`head_i = Attention(Q_i, K_i, V_i)`

The outputs of all the heads are then concatenated and passed through a final linear projection layer with weight matrix `W^O`:

`MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) @ W^O`

**Dimensions:**

*   Let `d_model` be the input dimension of Q, K, V (e.g., the embedding dimension).
*   Let `h` be the number of attention heads.
*   The dimensions of the projected queries, keys, and values for each head are typically `d_k = d_model / h` and `d_v = d_model / h`. This ensures that the total computation is similar to a single attention head with full `d_model` dimensions.
    *   `W_i^Q` has shape `(d_model, d_k)`
    *   `W_i^K` has shape `(d_model, d_k)`
    *   `W_i^V` has shape `(d_model, d_v)`
*   After concatenation, `Concat(head_1, ..., head_h)` will have shape `(batch_size, seq_len_q, h * d_v)`.
*   The output projection matrix `W^O` has shape `(h * d_v, d_model)`, so the final output of Multi-Head Attention has shape `(batch_size, seq_len_q, d_model)`, matching the input query dimension.

## 3.3 NumPy Implementation

We'll implement Multi-Head Attention as a class to manage the weight matrices and the forward pass.

First, let's re-include the `softmax` and `scaled_dot_product_attention` functions from Part 2, as our `MultiHeadAttention` class will depend on them.

```python
import numpy as np

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calculate the attention weights and the output of the attention mechanism.
    Args:
        Q (np.ndarray): Query matrix, shape (..., seq_len_q, d_k)
        K (np.ndarray): Key matrix, shape (..., seq_len_k, d_k)
        V (np.ndarray): Value matrix, shape (..., seq_len_v, d_v) where seq_len_k == seq_len_v
        mask (np.ndarray, optional): Mask to apply to the attention scores. Defaults to None.
                                     Shape should be broadcastable to (..., seq_len_q, seq_len_k).
                                     0 for attend, 1 for mask.
    Returns:
        tuple: (output, attention_weights)
            - output (np.ndarray): The output of the attention mechanism, shape (..., seq_len_q, d_v)
            - attention_weights (np.ndarray): The attention weights, shape (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        # Ensure mask is correctly broadcast. Mask shape (..., seq_len_q, seq_len_k)
        # Add a new axis for num_heads if Q, K, V are (batch, num_heads, seq, dim_k) and mask is (batch, seq_q, seq_k)
        # This is typically handled before calling scaled_dot_product_attention if mask is head-agnostic.
        # Or mask can be (batch, num_heads, seq_q, seq_k)
        scores = np.where(mask == 0, scores, -1e9)
    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output, attention_weights

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, seed=None):
        """
        Initialize the MultiHeadAttention layer.

        Args:
            d_model (int): Total dimension of the model.
            num_heads (int): Number of attention heads.
            seed (int, optional): Random seed for weight initialization for reproducibility.
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of keys/queries per head
        self.d_v = d_model // num_heads # Dimension of values per head

        if seed is not None:
            np.random.seed(seed)

        # Weight matrices for input projections (Q, K, V) for all heads combined initially
        # These will be reshaped or split to represent per-head weights.
        # W_q, W_k, W_v transform input (batch, seq_len, d_model) to (batch, seq_len, d_model)
        # which is then reshaped to (batch, seq_len, num_heads, d_k/d_v)
        # and transposed to (batch, num_heads, seq_len, d_k/d_v)
        self.W_q = np.random.randn(d_model, d_model) # Projects input Q to Q' for all heads
        self.W_k = np.random.randn(d_model, d_model) # Projects input K to K' for all heads
        self.W_v = np.random.randn(d_model, d_model) # Projects input V to V' for all heads

        # Weight matrix for output projection
        self.W_o = np.random.randn(d_model, d_model) # Projects concatenated head outputs back to d_model

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k or d_v).
        Then reshape to (batch_size, num_heads, seq_len, d_k or d_v).

        Args:
            x (np.ndarray): Input tensor, shape (batch_size, seq_len, d_model)
            batch_size (int): Batch size.

        Returns:
            np.ndarray: Reshaped tensor, shape (batch_size, num_heads, seq_len, d_k or d_v)
        """
        seq_len = x.shape[1]
        # Reshape to (batch_size, seq_len, num_heads, depth_per_head)
        x = x.reshape(batch_size, seq_len, self.num_heads, -1) # -1 infers d_k or d_v
        # Transpose to (batch_size, num_heads, seq_len, depth_per_head)
        return x.transpose(0, 2, 1, 3)

    def forward(self, Q_in, K_in, V_in, mask=None):
        """
        Perform the forward pass for Multi-Head Attention.

        Args:
            Q_in (np.ndarray): Query tensor, shape (batch_size, seq_len_q, d_model)
            K_in (np.ndarray): Key tensor, shape (batch_size, seq_len_k, d_model)
            V_in (np.ndarray): Value tensor, shape (batch_size, seq_len_v, d_model) (seq_len_k == seq_len_v)
            mask (np.ndarray, optional): Mask to apply. Shape (batch_size, seq_len_q, seq_len_k).
                                         It will be expanded for heads.

        Returns:
            tuple: (output, attention_weights)
                - output (np.ndarray): Final output, shape (batch_size, seq_len_q, d_model)
                - attention_weights (np.ndarray): Attention weights from scaled dot-product attention,
                                                  shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = Q_in.shape[0]
        seq_len_q = Q_in.shape[1]
        # seq_len_k = K_in.shape[1] # (and V)

        # 1. Linear Projections
        # Q_in: (batch_size, seq_len_q, d_model) @ W_q: (d_model, d_model) -> (batch_size, seq_len_q, d_model)
        Q_proj = np.matmul(Q_in, self.W_q)
        K_proj = np.matmul(K_in, self.W_k)
        V_proj = np.matmul(V_in, self.W_v)

        # 2. Split heads
        # Resulting shape: (batch_size, num_heads, seq_len_q, d_k)
        Q_split = self.split_heads(Q_proj, batch_size)
        # Resulting shape: (batch_size, num_heads, seq_len_k, d_k)
        K_split = self.split_heads(K_proj, batch_size)
        # Resulting shape: (batch_size, num_heads, seq_len_v, d_v)
        V_split = self.split_heads(V_proj, batch_size)

        # 3. Scaled Dot-Product Attention for each head
        # Q_split: (batch_size, num_heads, seq_len_q, d_k)
        # K_split: (batch_size, num_heads, seq_len_k, d_k)
        # V_split: (batch_size, num_heads, seq_len_v, d_v)
        # Mask needs to be (batch_size, 1, seq_len_q, seq_len_k) to broadcast across heads
        # or (batch_size, num_heads, seq_len_q, seq_len_k) if head-specific mask
        if mask is not None:
            # Expand mask for broadcasting over heads: (batch_size, seq_len_q, seq_len_k) -> (batch_size, 1, seq_len_q, seq_len_k)
            # This assumes the same mask is applied to all heads.
            mask_expanded = np.expand_dims(mask, axis=1) # Add head dimension
        else:
            mask_expanded = None

        # attention_output shape: (batch_size, num_heads, seq_len_q, d_v)
        # attention_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_output, attention_weights = scaled_dot_product_attention(Q_split, K_split, V_split, mask=mask_expanded)

        # 4. Concatenate heads and apply final linear projection
        # Transpose attention_output to (batch_size, seq_len_q, num_heads, d_v)
        attention_output_transposed = attention_output.transpose(0, 2, 1, 3)
        # Reshape to (batch_size, seq_len_q, d_model) because d_model = num_heads * d_v
        concat_attention = attention_output_transposed.reshape(batch_size, seq_len_q, self.d_model)

        # Final linear projection: (batch_size, seq_len_q, d_model) @ W_o: (d_model, d_model) -> (batch_size, seq_len_q, d_model)
        output = np.matmul(concat_attention, self.W_o)

        return output, attention_weights

```

**Explanation of the Code:**

1.  **`__init__(self, d_model, num_heads, seed=None)`:**
    *   Initializes dimensions `d_model`, `num_heads`, `d_k` (depth of key/query per head), and `d_v` (depth of value per head).
    *   `d_model` must be divisible by `num_heads`.
    *   Initializes weight matrices `W_q`, `W_k`, `W_v` for projecting Q, K, V. Instead of creating `num_heads` separate small matrices, we create larger matrices of shape `(d_model, d_model)`. The output of these projections will then be reshaped and split among heads.
    *   Initializes `W_o` of shape `(d_model, d_model)` for the final linear transformation.
    *   Weights are initialized with random values from a standard normal distribution (`np.random.randn`). In a real scenario, more sophisticated initialization (like Xavier/Glorot) would be used.

2.  **`split_heads(self, x, batch_size)`:**
    *   Takes an input `x` of shape `(batch_size, seq_len, d_model)`. This `x` is the result of `Q_in @ W_q` (or K, V equivalents).
    *   Reshapes `x` to `(batch_size, seq_len, self.num_heads, self.d_k or self.d_v)`. The `-1` in `reshape` infers the last dimension (which will be `d_k` or `d_v`).
    *   Transposes the dimensions to `(batch_size, self.num_heads, seq_len, self.d_k or self.d_v)`. This groups the data by head, making it suitable for parallel attention calculations across heads.

3.  **`forward(self, Q_in, K_in, V_in, mask=None)`:**
    *   `batch_size = Q_in.shape[0]` and `seq_len_q = Q_in.shape[1]` are extracted.
    *   **Linear Projections:** `Q_in`, `K_in`, `V_in` (each of shape `(batch_size, seq_len, d_model)`) are multiplied by their respective weight matrices (`W_q`, `W_k`, `W_v`), resulting in `Q_proj`, `K_proj`, `V_proj` (each of shape `(batch_size, seq_len, d_model)`).
    *   **Split Heads:** `Q_proj`, `K_proj`, `V_proj` are passed to `split_heads` to get `Q_split`, `K_split`, `V_split`. Their shapes become `(batch_size, num_heads, seq_len, d_k or d_v)`.
    *   **Mask Expansion:** If a `mask` (shape `(batch_size, seq_len_q, seq_len_k)`) is provided, it's expanded to `(batch_size, 1, seq_len_q, seq_len_k)` by adding a new dimension for `num_heads`. This allows the same mask to be broadcast across all heads during the `scaled_dot_product_attention` call.
    *   **Scaled Dot-Product Attention:** `scaled_dot_product_attention` is called with `Q_split`, `K_split`, `V_split`, and the `mask_expanded`. This function will perform attention in parallel for all heads because the head dimension is part of the batch dimensions (`...`) that `scaled_dot_product_attention` handles.
        *   `attention_output` will have shape `(batch_size, num_heads, seq_len_q, d_v)`.
        *   `attention_weights` will have shape `(batch_size, num_heads, seq_len_q, seq_len_k)`.
    *   **Concatenate and Project:**
        *   `attention_output` is transposed from `(batch_size, num_heads, seq_len_q, d_v)` to `(batch_size, seq_len_q, num_heads, d_v)`. This brings the head dimension next to the `d_v` dimension.
        *   It's then reshaped into `concat_attention` of shape `(batch_size, seq_len_q, self.d_model)`, effectively concatenating the outputs of the heads along the last dimension (since `d_model = num_heads * d_v`).
        *   Finally, `concat_attention` is multiplied by `self.W_o` to produce the final `output` of shape `(batch_size, seq_len_q, d_model)`.
    *   The method returns the final `output` and the `attention_weights` (which can be useful for analysis, showing what each head attended to).

## 3.4 Simple Input/Output Examples

Let's test our `MultiHeadAttention` class.

```python
# Example Usage of MultiHeadAttention

# Parameters
batch_size_ex = 2
seq_len_q_ex = 4  # Sequence length for queries
seq_len_k_ex = 5  # Sequence length for keys/values (can be different from seq_len_q)
d_model_ex = 12   # Model dimension (e.g., embedding size)
num_heads_ex = 3  # Number of attention heads

# Ensure d_model is divisible by num_heads
assert d_model_ex % num_heads_ex == 0

# Initialize MultiHeadAttention layer
mha = MultiHeadAttention(d_model=d_model_ex, num_heads=num_heads_ex, seed=42)

# Dummy input data
np.random.seed(101)
Q_input = np.random.rand(batch_size_ex, seq_len_q_ex, d_model_ex)
K_input = np.random.rand(batch_size_ex, seq_len_k_ex, d_model_ex)
V_input = np.random.rand(batch_size_ex, seq_len_k_ex, d_model_ex)

# Dummy mask (optional)
# Mask: 0 = attend, 1 = do not attend
# For example, mask out the last key for all queries in the first batch item
# and the first key for all queries in the second batch item.
mask_input = np.zeros((batch_size_ex, seq_len_q_ex, seq_len_k_ex), dtype=int)
mask_input[0, :, -1] = 1 # First batch item, all queries, last key masked
mask_input[1, :, 0] = 1  # Second batch item, all queries, first key masked

print("--- Multi-Head Attention Example ---")
print(f"Input Q shape: {Q_input.shape}")
print(f"Input K shape: {K_input.shape}")
print(f"Input V shape: {V_input.shape}")
print(f"Input Mask shape: {mask_input.shape}")

# Forward pass
output_mha, attention_weights_mha = mha.forward(Q_input, K_input, V_input, mask=mask_input)

print(f"\nOutput MHA shape: {output_mha.shape}")
print(f"Attention Weights MHA shape: {attention_weights_mha.shape}")

# Verify output shape: (batch_size, seq_len_q, d_model)
assert output_mha.shape == (batch_size_ex, seq_len_q_ex, d_model_ex)

# Verify attention_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
assert attention_weights_mha.shape == (batch_size_ex, num_heads_ex, seq_len_q_ex, seq_len_k_ex)

print("\nSample of Output MHA (first batch, first query, first 2 features):\n", output_mha[0, 0, :2])
print("\nSample of Attention Weights (first batch, first head, first query, all keys):\n", attention_weights_mha[0, 0, 0, :])

# Check if masking worked (weights for masked positions should be ~0)
print("\nChecking masked attention weights:")
# First batch item, all queries, last key was masked.
# Check weights for first head, first query, last key for the first batch item.
print(f"Weight for masked K (batch 0, head 0, query 0, key {seq_len_k_ex-1}): {attention_weights_mha[0, 0, 0, -1]}")
# Second batch item, all queries, first key was masked.
# Check weights for first head, first query, first key for the second batch item.
print(f"Weight for masked K (batch 1, head 0, query 0, key 0): {attention_weights_mha[1, 0, 0, 0]}")

# Example without mask
output_mha_no_mask, _ = mha.forward(Q_input, K_input, V_input, mask=None)
print(f"\nOutput MHA (no mask) shape: {output_mha_no_mask.shape}")
```

**Running the Example:**

Save the code (including `softmax`, `scaled_dot_product_attention`, the `MultiHeadAttention` class, and the example usage) into a Python file and run it. You should see the shapes of the inputs and outputs, and samples of the results. The attention weights for the masked positions should be very close to zero.

## 3.5 Key Takeaways

*   Multi-Head Attention applies several Scaled Dot-Product Attention operations in parallel.
*   Each head uses different linear projections of Q, K, and V, allowing it to learn different aspects of the relationships in the data.
*   The outputs of the heads are concatenated and linearly projected to produce the final result.
*   This mechanism allows the model to capture a richer set of features and dependencies compared to a single attention head.
*   The overall dimensionality `d_model` is typically preserved throughout the Multi-Head Attention block.

## 3.6 What's Next?

Now that we have Multi-Head Attention, the next crucial component in a Transformer block is the **Position-wise Feed-Forward Network (FFN)**. This is a relatively simple network applied independently to each position in the sequence. We will implement this in Part 4.

Stay tuned!
