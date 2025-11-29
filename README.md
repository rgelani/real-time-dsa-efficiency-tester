⚡ Real-Time Algorithmic Efficiency Tester: BST Traversal Optimization

This project is a powerful, self-contained application built using Streamlit that provides real-time benchmarking of two critical Binary Search Tree (BST) traversal implementations: a traditional list-aggregation approach and a memory-optimized generator-based approach.

The primary goal is to visually and quantifiably demonstrate the real-world performance impact of optimizing Space Complexity from $O(N)$ (linear space) down to $O(h)$ (space proportional to the height of the tree).

🚀 Key Features

Real-Time Benchmarking: Users can input the desired tree size (N up to 200,000 nodes) and instantly run a performance comparison of the two traversal methods.

Guaranteed Balance: The application programmatically constructs a Height-Balanced BST from a sorted list using the Divide and Conquer approach ($O(\log N)$ depth) to ensure scientific accuracy and prevent performance degradation from degenerate (linked-list-like) trees.

Space Complexity Focus: Clearly isolates and measures the difference in execution time due to memory overhead introduced by the $O(N)$ list aggregation method.

Performance Metrics: Displays the average execution time (in seconds), the time difference in milliseconds, and the resulting Generator Speedup Factor.

Self-Contained Code: All logic, including the Node structure, tree building, and traversal functions, are contained within a single Python file for ease of deployment and review.

🧠 Algorithmic Focus: $O(N)$ vs $O(h)$

The benchmark centers on the In-Order Traversal algorithm:

Implementation

Technique

Time Complexity (TC)

Space Complexity (SC)

Original

Recursive (List Aggregation)

$O(N)$

$O(N)$ (Memory for the final list)

Optimized

Recursive (Python Generator yield from)

$O(N)$

$O(h)$ (Memory for the recursion stack)

The reduction in space complexity (especially for large $N$) leads to faster execution times due to reduced memory allocation overhead.

🛠️ How to Run the Application

This project requires Python 3.9+ and the following libraries:

Installation:

pip install streamlit pandas


Execution:
Navigate to the directory containing interactive_benchmark.py and run the Streamlit command:

streamlit run interactive_benchmark.py


Testing:
The application will open in your browser. Select a "Tree Size (N nodes)" (e.g., 50,000) and click "🚀 Run Real-Time Benchmark" to see the comparison metrics.

💻 Technologies Used

Python 3.x

Streamlit (Interactive UI/Dashboard)

Pandas (Data Structuring and Display)

Data Structures: Binary Search Trees (BST), Recursion, Python Generators

This project effectively demonstrates expertise in high-performance Python, Data Structures and Algorithms (DSA), and modern data visualization techniques.