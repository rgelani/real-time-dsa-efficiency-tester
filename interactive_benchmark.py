import streamlit as st
import time
import pandas as pd
import random
from typing import List, Optional, Any, Iterator, Callable
import sys
import os

# Set a higher recursion limit for building large trees
sys.setrecursionlimit(2000)

# --- CORE DSA STRUCTURES ---

class Node:
    """Basic Node structure for the Binary Tree."""
    def __init__(self, key: int):
        self.key = key
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None

class BinaryTree:
    """Handles tree construction for testing."""
    def __init__(self):
        self.idx = -1
    
    # NOTE: The buildTree below is designed for Preorder list reconstruction 
    # with sentinel values. We will replace its usage with sortedListToBST.
    def buildTree(self, nodes: List[int]) -> Optional[Node]:
        """Builds a tree from a preorder list of nodes (old logic)."""
        self.idx += 1
          
        if self.idx >= len(nodes) or nodes[self.idx] == -1:
            return None
                 
        root = Node(nodes[self.idx])
        root.left = self.buildTree(nodes)  
        root.right = self.buildTree(nodes)
        return root

# --- NEW FUNCTION: BALANCED BST CONSTRUCTION ---
def sortedListToBST(nodes: List[int], start: int, end: int) -> Optional[Node]:
    """
    Constructs a height-balanced BST from a sorted list using divide and conquer.
    This prevents the RecursionError by guaranteeing O(log N) depth.
    """
    if start > end:
        return None
    
    mid = (start + end) // 2
    root = Node(nodes[mid])
    
    root.left = sortedListToBST(nodes, start, mid - 1)
    root.right = sortedListToBST(nodes, mid + 1, end)
    
    return root

# --- ALGORITHMS (Manually copied for self-contained app) ---

# Caching is removed here to ensure accurate timing for every run
def inorder_original(root: Optional[Node], result: Optional[List[int]] = None) -> List[int]:
    """O(N) Space Complexity: Traditional recursive list-aggregation traversal."""
    if result is None:
        result = []
    if root is None:
        return result

    # The recursive calls must use the function name directly since they are copied outside the class
    inorder_original(root.left, result)
    result.append(root.key)
    inorder_original(root.right, result)
    return result

# Caching is removed here to ensure accurate timing for every run
def inorder_traversal_generator(root: Optional[Node]) -> Iterator[int]:
    """O(h) Space Complexity: Generator-based traversal using yield from."""
    if root is not None:
        # 1. Traverse Left Subtree Recursively using yield from
        yield from inorder_traversal_generator(root.left) 
        
        # 2. Yield the Root Key
        yield root.key
        
        # 3. Traverse Right Subtree Recursively using yield from
        yield from inorder_traversal_generator(root.right)


# --- BENCHMARKING LOGIC ---

def run_single_test(func: Callable[[Optional[Node]], Any], root: Optional[Node], run_count: int = 5) -> float:
    """Times the execution of a function over multiple runs."""
    total_time = 0.0
    
    for _ in range(run_count):
        start = time.perf_counter()
        
        result = func(root) 
        
        # If the result is a generator, force consumption (important for timing)
        if isinstance(result, Iterator):
            list(result)
            
        end = time.perf_counter()
        total_time += (end - start)
        
    return total_time / run_count

def run_full_benchmark(N: int) -> pd.DataFrame:
    """Builds a tree and runs the two primary benchmarks."""
    
    # 1. Build the Tree
    # Create a list of N unique sorted keys for guaranteed balance
    nodes_list = list(range(1, N + 1)) 
    
    # Use the new function to build a BALANCED tree
    with st.spinner(f"Building Balanced BST with N={N:,} nodes..."):
        test_tree = sortedListToBST(nodes_list, 0, N - 1) # <--- FIXED: Now builds a balanced tree
    
    # 2. Run Benchmarks
    # Run count is pulled from Streamlit session state
    run_count = st.session_state.RUNS 

    with st.spinner(f"Running List Aggregation (O(N) Space) on N={N:,} nodes..."):
        time_original = run_single_test(inorder_original, test_tree, run_count=run_count)
    
    with st.spinner(f"Running Generator (O(h) Space) on N={N:,} nodes..."):
        # Note: Must re-import the function for the second run since st.cache_data is used
        time_generator = run_single_test(inorder_traversal_generator, test_tree, run_count=run_count)

    # 3. Format Results
    data = {
        'Implementation': ['Original (List O(N) Space)', 'Optimized (Generator O(h) Space)'],
        'Average Time (seconds)': [time_original, time_generator]
    }
    df = pd.DataFrame(data)
    
    # Calculate difference for display
    df['Time Difference'] = df['Average Time (seconds)'].apply(lambda x: f'{x:,.8f} s')
    df['Speedup Factor'] = time_original / time_generator
    
    return df

# --- STREAMLIT UI ---

st.title("⚡ Real-Time Algorithmic Efficiency Tester")
st.markdown("### BST Traversal: List Aggregation vs. Generator")

# Explainer
st.info(
    "Enter a tree size (N) and press 'Run Benchmark' to see the performance difference in real-time. "
    "The Generator implementation uses significantly less memory (O(h) space), often leading to a speed advantage."
)

# User Input
N_input = st.number_input(
    "Tree Size (N nodes) to Test:",
    min_value=100,
    max_value=200000,
    value=10000,
    step=1000
)

# Initialize RUNS in session state if it doesn't exist
if 'RUNS' not in st.session_state:
    st.session_state.RUNS = 5

st.session_state.RUNS = st.slider("Runs per Algorithm (for Averaging):", min_value=1, max_value=10, value=st.session_state.RUNS)

if st.button("🚀 Run Real-Time Benchmark", type="primary"):
    
    st.session_state.N = N_input
    st.session_state.ran_test = True
    
    # Run the benchmark and store the result in session state
    st.session_state.results_df = run_full_benchmark(st.session_state.N)

if 'ran_test' in st.session_state and st.session_state.ran_test:
    st.header(f"Results for N = {st.session_state.N:,} Nodes")
    
    df = st.session_state.results_df
    
    # Display the primary results table
    st.dataframe(
        df[['Implementation', 'Average Time (seconds)']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'Average Time (seconds)': st.column_config.NumberColumn(format="%.8f s")
        }
    )
    
    # Display the key takeaway metrics
    original_time = df[df['Implementation'] == 'Original (List O(N) Space)']['Average Time (seconds)'].iloc[0]
    generator_time = df[df['Implementation'] == 'Optimized (Generator O(h) Space)']['Average Time (seconds)'].iloc[0]
    speedup = original_time / generator_time
    time_diff = original_time - generator_time
    
    col1, col2 = st.columns(2)
    
    col1.metric(
        label="Generator Speedup Factor",
        value=f"{speedup:.2f}x",
        delta=f"Faster by {time_diff * 1000:,.2f} ms"
    )

    col2.metric(
        label="Space Complexity Focus",
        value="O(h)",
        delta="O(N) for Original"
    )
    
    st.markdown("---")
    st.markdown("### Algorithmic Interpretation")
    st.markdown(f"""
    - **List Aggregation (Original):** Requires **$O(N)$** extra space to store all node keys in the final list.
    - **Generator (Optimized):** Requires only **$O(h)$** extra space for the recursion stack (where $h$ is the tree's height), making it more memory efficient and often faster in real-world environments.
    """)