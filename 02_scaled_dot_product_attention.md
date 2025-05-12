# Part 2: Implementing Scaled Dot-Product Attention

Welcome to Part 2 of our Transformer from scratch series! In this part, we'll implement the fundamental building block of the Transformer's attention mechanism: **Scaled Dot-Product Attention**.

## 2.1 What is Attention?

In the context of neural networks, particularly sequence processing, **attention** is a mechanism that allows a model to selectively focus on certain parts of an input sequence when producing an output. Instead of treating all parts of the input equally, the model learns to assign different "attention weights" to different input elements, signifying their relevance to the current processing step.

Think of it like how humans pay attention. When you read a sentence, you might focus more on certain words to understand its meaning. Attention mechanisms in neural networks try to mimic this behavior.

## 2.2 Scaled Dot-Product Attention

Scaled Dot-Product Attention is one of the most common and effective types of attention mechanisms, and it's the one used in the original Transformer paper.

It operates on three inputs:

*   **Queries (Q):** A set of vectors representing what we are looking for.
*   **Keys (K):** A set of vectors representing the information available in the input sequence.
*   **Values (V):** A set of vectors representing the actual content or features of the input sequence.

### Intuitive Explanation with an Example

To make Q, K, and V more concrete, let's use an example sentence: "I have a cat and i love her."

Imagine the model is processing this sentence and needs to understand the relationships between words, particularly what "her" refers to.

*   **Query (Q):** Think of the Query as the current word or concept the model is focusing on and trying to gather more information about. It's like asking a question.
    *   *Example:* If the model is trying to understand what "her" refers to, then "her" (or its vector representation) acts as the Query. The implicit "question" is "Who or what is 'her' referring to in this context?"

*   **Key (K):** Think of Keys as labels or identifiers for all the words in the sentence. Each word in the input sequence has a Key associated with it. This Key is a vector that represents the word's content in a way that can be compared to the Query.
    *   *Example:* Every word in "I have a cat and i love her" would have a corresponding Key vector. The Key for "cat" would be a vector representing the concept of "cat". Similarly for "I", "have", etc.

*   **Value (V):** Think of Values as the actual content, meaning, or rich representation of each word. Once the Query, by comparing itself to all Keys, identifies which words are most relevant, it uses their corresponding Values to get the information it needs.
    *   *Example:* Each word also has a Value vector. The Value for "cat" is the rich semantic representation of "cat" that the model can use.

**How it works with the example "I have a cat and i love her":**

Let's say the model is processing the word "her" (this is our **Query**).
1.  The model takes the Query vector for "her" and compares it against all the **Key** vectors in the sentence (i.e., the Key for "I", "have", "a", "cat", "and", "i", "love", "her"). This comparison is typically done using a dot product.
2.  This comparison calculates a score for each Key. The score between the Query "her" and the Key "cat" will likely be high, because "her" (a pronoun) often refers to something mentioned earlier, and "cat" is a plausible antecedent in this context. The score between "her" and, say, "love" might be lower if the primary goal is to find the referent.
3.  These scores are then passed through a softmax function, which converts them into "attention weights". A high weight for the Key "cat" means the Query "her" should pay a lot of attention to "cat". These weights sum to 1, representing a distribution of attention.
4.  The model then takes these attention weights and uses them to compute a weighted sum of all the **Value** vectors. If the Key "cat" received a high attention weight, its corresponding Value vector contributes significantly to the resulting sum.
5.  The output of this process is a new vector, a refined representation for "her". This new vector now incorporates contextual information, strongly influenced by the Value of "cat", helping the model understand that "her" refers to "cat".

In essence, for a given Query (what the model is currently focusing on):
*   It uses **Keys** to determine "how relevant" every other part of the input is.
*   It then uses the **Values** of the relevant parts (weighted by their relevance) to construct a richer, context-aware representation of the Query.

This mechanism allows the model to dynamically decide which parts of the input are most important for understanding or processing each specific part of the sequence.

The core idea is to compute a score for each key with respect to a given query. This score determines how much attention the query should pay to that particular key-value pair. The scores are then used to compute a weighted sum of the values, resulting in the output of the attention layer.

**Mathematical Formula:**

The formula for Scaled Dot-Product Attention is:

```
Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
```

Let's break down each part:

1.  **`Q @ K.T` (Dot Product of Queries and Transposed Keys):**
    *   `Q` has dimensions `(num_queries, d_k)` or `(batch_size, seq_len_q, d_k)` if batched.
    *   `K` has dimensions `(num_keys, d_k)` or `(batch_size, seq_len_k, d_k)` if batched.
    *   `K.T` (transpose of K) will have dimensions `(d_k, num_keys)` or `(batch_size, d_k, seq_len_k)`.
    *   The dot product `Q @ K.T` results in a matrix of **attention scores** (also called compatibility scores or alignment scores) with dimensions `(num_queries, num_keys)` or `(batch_size, seq_len_q, seq_len_k)`. Each element `(i, j)` in this matrix represents how much query `i` aligns with key `j`.

2.  **`sqrt(d_k)` (Scaling Factor):**
    *   `d_k` is the dimension of the key vectors (and query vectors, as they must have the same dimension for the dot product).
    *   The scores are scaled down by dividing by the square root of `d_k`. This scaling is crucial because for large values of `d_k`, the dot products can grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. Scaling helps to counteract this effect, leading to more stable training.

3.  **`softmax(...)` (Softmax Function):**
    *   The softmax function is applied row-wise to the scaled scores. This converts the scores into probabilities (non-negative values that sum to 1 across each row).
    *   The output of the softmax, often called **attention weights**, will have the same dimensions as the scaled scores: `(num_queries, num_keys)` or `(batch_size, seq_len_q, seq_len_k)`.
    *   Each row `i` of the attention weights represents the distribution of attention that query `i` pays to all the keys.

4.  **`... @ V` (Dot Product with Values):**
    *   The attention weights are then multiplied by the `V` (Values) matrix.
    *   `V` has dimensions `(num_values, d_v)` or `(batch_size, seq_len_v, d_v)`. Note that `num_keys` must equal `num_values` (or `seq_len_k` must equal `seq_len_v`), as each key corresponds to a value.
    *   The dimension `d_v` is the dimension of the value vectors. It can be different from `d_k`.
    *   The final output of the attention mechanism has dimensions `(num_queries, d_v)` or `(batch_size, seq_len_q, d_v)`. Each output vector is a weighted sum of the value vectors, where the weights are determined by the attention scores.

**Optional Masking:**

In some scenarios (like in the decoder of a Transformer, or for handling padded sequences), we might want to prevent attention to certain positions. This is done by adding a **mask** (typically a large negative number like -infinity for elements we want to ignore) to the scaled scores *before* applying the softmax. The softmax will then assign near-zero probabilities to these masked positions.

## 2.3 NumPy Implementation

Let's implement the Scaled Dot-Product Attention using NumPy.

```python
import numpy as np

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True)) # Subtract max for numerical stability
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Calculate the attention weights and the output of the attention mechanism.

    Args:
        Q (np.ndarray): Query matrix, shape (..., seq_len_q, d_k)
        K (np.ndarray): Key matrix, shape (..., seq_len_k, d_k)
        V (np.ndarray): Value matrix, shape (..., seq_len_v, d_v) where seq_len_k == seq_len_v
        mask (np.ndarray, optional): Mask to apply to the attention scores, shape (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        tuple: (output, attention_weights)
            - output (np.ndarray): The output of the attention mechanism, shape (..., seq_len_q, d_v)
            - attention_weights (np.ndarray): The attention weights, shape (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]  # Dimension of the key vectors

    # 1. Calculate dot product of Q and K.T: (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    # We need to transpose the last two dimensions of K for matrix multiplication
    # K.transpose(0, 1, 3, 2) if K is (batch_size, num_heads, seq_len_k, d_k)
    # For simpler case (seq_len_k, d_k), K.T is fine.
    # For (batch_size, seq_len_k, d_k), K.transpose(0, 2, 1) is needed.
    # np.matmul handles broadcasting for batch dimensions correctly.
    scores = np.matmul(Q, K.swapaxes(-2, -1))  # Q @ K.T

    # 2. Scale the scores
    scaled_scores = scores / np.sqrt(d_k)

    # 3. Apply mask (if provided)
    if mask is not None:
        # The mask should be shaped to broadcast correctly with scaled_scores.
        # Typically, mask elements are 0 for positions to attend to and 1 for masked positions.
        # We want to add a large negative number to masked positions.
        scaled_scores = np.where(mask == 0, scaled_scores, -1e9) # or -np.inf for true infinity

    # 4. Apply softmax to get attention weights
    # Softmax is applied on the last axis (seq_len_k) to get probabilities over keys for each query.
    attention_weights = softmax(scaled_scores, axis=-1)

    # 5. Multiply attention weights by V: (..., seq_len_q, seq_len_k) @ (..., seq_len_v, d_v) -> (..., seq_len_q, d_v)
    # Note: seq_len_k must be equal to seq_len_v (number of keys must be number of values)
    output = np.matmul(attention_weights, V)

    return output, attention_weights

```

**Explanation of the Code:**

1.  **`softmax(x, axis=-1)` function:**
    *   This is a standard softmax implementation.
    *   `np.max(x, axis=axis, keepdims=True)` is subtracted from `x` before exponentiation. This is a common trick for numerical stability, preventing overflow when `x` contains large values. It doesn't change the output of softmax because `softmax(x) = softmax(x - c)` for any constant `c`.
    *   `keepdims=True` ensures that the dimensions are preserved after max and sum operations, allowing for proper broadcasting.
    *   `axis=-1` means the softmax is computed along the last axis of the input array.

2.  **`scaled_dot_product_attention(Q, K, V, mask=None)` function:**
    *   `d_k = Q.shape[-1]`: We get the dimension of the key vectors from the last dimension of `Q`.
    *   `scores = np.matmul(Q, K.swapaxes(-2, -1))`: This computes `Q @ K.T`. `K.swapaxes(-2, -1)` transposes the last two dimensions of `K`, which is necessary for the matrix multiplication. `np.matmul` correctly handles batch dimensions if `Q` and `K` are 3D or higher (e.g., `(batch_size, seq_len, d_model)`).
    *   `scaled_scores = scores / np.sqrt(d_k)`: Scales the scores.
    *   **Masking:**
        *   If a `mask` is provided, it's used to modify `scaled_scores`.
        *   The `mask` is expected to have `0` where attention is allowed and `1` (or `True`) where it should be prevented.
        *   `np.where(mask == 0, scaled_scores, -1e9)`: This line is crucial. If `mask` at a position is `0` (allow attention), it keeps the original `scaled_scores`. If `mask` is `1` (prevent attention), it replaces the score with a very large negative number (`-1e9`). When softmax is applied, these large negative numbers will result in near-zero probabilities.
    *   `attention_weights = softmax(scaled_scores, axis=-1)`: Computes the attention weights using our softmax function. The softmax is applied along the last axis (`seq_len_k`), so for each query, the weights for all keys sum to 1.
    *   `output = np.matmul(attention_weights, V)`: Computes the final output by taking the weighted sum of `V` using the `attention_weights`.
    *   The function returns both the final `output` and the `attention_weights` (which can be useful for visualization or analysis).

## 2.4 Simple Input/Output Examples

Let's test our implementation with some simple examples.

**Example 1: Basic Attention (No Batching)**

Suppose we have:
*   A single query vector.
*   Three key vectors.
*   Three corresponding value vectors.

```python
# Setup: Dimensions
seq_len_q = 1  # Number of queries (e.g., current word we are focusing on)
seq_len_k = 3  # Number of keys/values in the sequence (e.g., words in the source sentence)
d_k = 2        # Dimension of keys and queries
d_v = 4        # Dimension of values

# Initialize Q, K, V with some dummy data
np.random.seed(42) # for reproducibility
Q_ex1 = np.random.rand(seq_len_q, d_k)   # (1, 2)
K_ex1 = np.random.rand(seq_len_k, d_k)   # (3, 2)
V_ex1 = np.random.rand(seq_len_k, d_v)   # (3, 4)

print("Q_ex1:\n", Q_ex1)
print("K_ex1:\n", K_ex1)
print("V_ex1:\n", V_ex1)

# Calculate attention
output_ex1, attention_weights_ex1 = scaled_dot_product_attention(Q_ex1, K_ex1, V_ex1)

print("\n--- Example 1 Output ---")
print("Attention Weights (shape: {}):\n".format(attention_weights_ex1.shape), attention_weights_ex1)
print("Output (shape: {}):\n".format(output_ex1.shape), output_ex1)

# Expected shapes:
# Attention Weights: (1, 3) - one query, attention over 3 keys
# Output: (1, 4) - one query, output dimension d_v
```

**Example 2: Batched Attention with Masking**

Now, let's consider a batch of sequences and apply a mask.

```python
# Setup: Dimensions for batch
batch_size = 2
seq_len_q_b = 2  # Number of queries per item in batch
seq_len_k_b = 3  # Number of keys/values per item in batch
d_k_b = 4        # Dimension of keys and queries
d_v_b = 5        # Dimension of values

# Initialize Q, K, V with some dummy data for batch
np.random.seed(123)
Q_ex2 = np.random.rand(batch_size, seq_len_q_b, d_k_b) # (2, 2, 4)
K_ex2 = np.random.rand(batch_size, seq_len_k_b, d_k_b) # (2, 3, 4)
V_ex2 = np.random.rand(batch_size, seq_len_k_b, d_v_b) # (2, 3, 5)

# Create a mask
# Suppose for the first item in batch, the second query cannot attend to the third key.
# And for the second item, the first query cannot attend to the first key.
# Mask: 0 = attend, 1 = do not attend
mask_ex2 = np.zeros((batch_size, seq_len_q_b, seq_len_k_b), dtype=int)
mask_ex2[0, 1, 2] = 1 # Batch 0, Query 1, Key 2 is masked
mask_ex2[1, 0, 0] = 1 # Batch 1, Query 0, Key 0 is masked

print("\nQ_ex2 (shape {}):\n".format(Q_ex2.shape), Q_ex2[0,0,:2], "...") # Print a snippet
print("K_ex2 (shape {}):\n".format(K_ex2.shape), K_ex2[0,0,:2], "...")
print("V_ex2 (shape {}):\n".format(V_ex2.shape), V_ex2[0,0,:2], "...")
print("Mask_ex2 (shape {}):\n".format(mask_ex2.shape), mask_ex2)

# Calculate attention
output_ex2, attention_weights_ex2 = scaled_dot_product_attention(Q_ex2, K_ex2, V_ex2, mask=mask_ex2)

print("\n--- Example 2 Output ---")
print("Attention Weights (shape: {}):\n".format(attention_weights_ex2.shape), attention_weights_ex2)
print("Output (shape: {}):\n".format(output_ex2.shape), output_ex2)

# Check masked positions in attention weights (should be close to 0)
print("\nChecking masked positions in attention weights:")
print("Batch 0, Query 1, Key 2 weight:", attention_weights_ex2[0, 1, 2])
print("Batch 1, Query 0, Key 0 weight:", attention_weights_ex2[1, 0, 0])

# Expected shapes:
# Attention Weights: (2, 2, 3)
# Output: (2, 2, 5)
```

**Running the Examples:**

If you save the Python code (the `softmax` and `scaled_dot_product_attention` functions, and the example calls) into a `.py` file and run it, you will see the computed attention weights and outputs. The attention weights for the masked positions in Example 2 should be very close to zero.

## 2.5 Key Takeaways

*   Scaled Dot-Product Attention computes a weighted sum of `Values`, where the weights are determined by the compatibility of `Queries` and `Keys`.
*   Scaling by `sqrt(d_k)` is important for stabilizing gradients.
*   Masking allows the model to ignore certain positions during attention calculation, which is crucial for tasks like handling padding or autoregressive decoding.
*   Our NumPy implementation can handle both single instances and batches of data.

## 2.6 What's Next?

With Scaled Dot-Product Attention implemented, we are ready to build a more powerful version: **Multi-Head Attention**. This involves running the scaled dot-product attention mechanism multiple times in parallel with different, learned linear projections of Q, K, and V. We'll cover this in Part 3.

Stay tuned!
