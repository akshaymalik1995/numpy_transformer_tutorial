# Part 1: Introduction to Transformers and NumPy Setup

## 1.1 What is a Transformer?

The Transformer is a novel neural network architecture introduced in the paper "Attention Is All You Need" by Vaswani et al. (2017). It has revolutionized the field of Natural Language Processing (NLP) and has found applications in various other domains like computer vision and reinforcement learning.

Unlike traditional sequence-to-sequence models like Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTMs) networks, Transformers do not rely on recurrence. Instead, they process entire sequences of data at once using a mechanism called **attention**, specifically **self-attention**. This allows for significant parallelization during training and enables the model to capture long-range dependencies in the data more effectively.

**Key characteristics of Transformers:**

*   **Self-Attention Mechanisms:** These allow the model to weigh the importance of different parts of the input sequence when processing a particular element.
*   **Parallel Processing:** Unlike RNNs that process tokens sequentially, Transformers can process all tokens in a sequence simultaneously.
*   **Encoder-Decoder Structure:** The original Transformer consists of an encoder to process the input sequence and a decoder to generate the output sequence.
*   **Positional Encoding:** Since Transformers don't have an inherent notion of sequence order, positional information is explicitly added to the input embeddings.

Transformers have become the foundation for many state-of-the-art models like BERT, GPT, and T5.

## 1.2 Why NumPy?

While deep learning frameworks like TensorFlow and PyTorch provide high-level abstractions and automatic differentiation, implementing a Transformer from scratch using only NumPy offers several benefits:

*   **Fundamental Understanding:** It forces a deeper understanding of the underlying mathematics and mechanics of each component.
*   **Clarity:** Without the black boxes of high-level libraries, the flow of data and computations becomes more transparent.
*   **Educational Value:** It's an excellent exercise for learning how these complex models are built from basic building blocks.
*   **Lightweight:** NumPy is a fundamental package for numerical computation in Python and has minimal overhead.

Our goal here is not to build the most optimized Transformer, but to understand its core components intimately.

## 1.3 Setting Up the Environment

To follow this tutorial, you'll need Python and NumPy installed.

**Installation:**

If you don't have NumPy installed, you can install it using pip:

```bash
pip install numpy
```

**Verification:**

You can verify the installation by opening a Python interpreter and typing:

```python
import numpy as np
print(np.__version__)
```

This should print the installed version of NumPy.

## 1.4 Basic NumPy Operations for Transformers

We'll be using several NumPy operations extensively. Here's a quick refresher on some of the most important ones:

*   **Array Creation:**
    ```python
    # Create a 1D array
    arr1d = np.array([1, 2, 3])
    print("1D Array:\n", arr1d)

    # Create a 2D array (matrix)
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    print("2D Array:\n", arr2d)

    # Create an array of zeros
    zeros_arr = np.zeros((2, 3)) # Shape (2 rows, 3 columns)
    print("Zeros Array:\n", zeros_arr)

    # Create an array of ones
    ones_arr = np.ones((3, 2))
    print("Ones Array:\n", ones_arr)

    # Create an array with random values (uniform distribution between 0 and 1)
    rand_arr = np.random.rand(2, 2)
    print("Random Array (uniform):\n", rand_arr)

    # Create an array with random values (standard normal distribution)
    randn_arr = np.random.randn(2, 2)
    print("Random Array (normal):\n", randn_arr)
    ```

*   **Array Attributes:**
    ```python
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    print("Shape:", arr.shape)    # (2, 3)
    print("Dimensions:", arr.ndim) # 2
    print("Data type:", arr.dtype) # int64 (or similar, depending on your system)
    print("Size (total elements):", arr.size) # 6
    ```

*   **Matrix Multiplication (Dot Product):**
    This is fundamental to many operations in Transformers, especially attention.
    ```python
    mat_a = np.array([[1, 2], [3, 4]])
    mat_b = np.array([[5, 6], [7, 8]])

    # Element-wise multiplication
    # print("Element-wise product:\n", mat_a * mat_b)

    # Matrix multiplication (dot product)
    dot_product = np.dot(mat_a, mat_b)
    # Alternatively, using the @ operator (Python 3.5+)
    # dot_product_alt = mat_a @ mat_b
    print("Dot Product:\n", dot_product)
    # Expected output:
    # [[1*5+2*7, 1*6+2*8],
    #  [3*5+4*7, 3*6+4*8]]
    # = [[19, 22],
    #    [43, 50]]
    ```

*   **Transpose:**
    ```python
    mat = np.array([[1, 2, 3], [4, 5, 6]])
    print("Original Matrix:\n", mat)
    print("Transposed Matrix:\n", mat.T)
    # Or np.transpose(mat)
    ```

*   **Summation, Mean, Max, Min:**
    ```python
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    print("Sum of all elements:", np.sum(arr))
    print("Sum along axis 0 (columns):", np.sum(arr, axis=0))
    print("Sum along axis 1 (rows):", np.sum(arr, axis=1))

    print("Mean of all elements:", np.mean(arr))
    print("Max element:", np.max(arr))
    print("Min element:", np.min(arr))
    ```

*   **Broadcasting:**
    NumPy's broadcasting allows operations on arrays of different shapes under certain constraints. This is very powerful.
    ```python
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    scalar = 10
    print("Array + scalar:\n", arr + scalar) # Scalar added to each element

    row_vector = np.array([10, 20, 30])
    print("Array + row_vector:\n", arr + row_vector) # Row vector added to each row

    col_vector = np.array([[10], [20]])
    print("Array + col_vector:\n", arr + col_vector) # Column vector added to each column
    ```

*   **Indexing and Slicing:**
    ```python
    arr = np.array([0, 10, 20, 30, 40, 50])
    print("Element at index 2:", arr[2]) # 20
    print("Elements from index 1 to 3 (exclusive):", arr[1:3]) # [10, 20]
    print("All elements from index 2:", arr[2:]) # [20, 30, 40, 50]

    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Element at row 0, col 1:", mat[0, 1]) # 2
    print("First row:", mat[0, :]) # [1, 2, 3]
    print("First column:", mat[:, 0]) # [1, 4, 7]
    print("Submatrix (rows 0-1, cols 1-2):\n", mat[0:2, 1:3])
    # [[2, 3],
    #  [5, 6]]
    ```

*   **Mathematical Functions:**
    NumPy provides a wide range of mathematical functions that operate element-wise on arrays.
    ```python
    arr = np.array([1, 2, 3])
    print("Square root:", np.sqrt(arr))
    print("Exponential:", np.exp(arr))
    print("Logarithm (natural):", np.log(arr))

    # Softmax (we will implement this later, but good to know np.exp and np.sum)
    def simple_softmax(x):
        e_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
        return e_x / e_x.sum(axis=-1, keepdims=True)

    scores = np.array([1.0, 2.0, 0.5])
    print("Simple Softmax:", simple_softmax(scores))
    ```

## 1.5 What's Next?

In the next part, we will dive into the core mechanism of the Transformer: **Scaled Dot-Product Attention**. We'll understand its mathematical formulation and implement it using NumPy.

Stay tuned!
