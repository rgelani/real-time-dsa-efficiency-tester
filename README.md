# ⚡ Real-Time Algorithmic Efficiency Tester: BST Traversal Optimization

This project is a self-contained Streamlit web app that benchmarks two Binary Search Tree (BST) In-Order Traversal methods in real time. It compares a traditional list-aggregation approach with a memory-efficient generator-based traversal.

The goal is to demonstrate the practical performance difference when reducing Space Complexity from **O(N)** to **O(h)**.

---

## 🚀 Key Features

- **Real-Time Benchmarking**  
  Input the tree size (up to 200,000 nodes) and instantly compare traversal performance.

- **Guaranteed Balance**  
  Automatically builds a height-balanced BST using a Divide and Conquer method (depth ≈ O(log N)) to avoid skewed trees.

- **Space Complexity Focus**  
  Measures the execution time difference caused by list aggregation (**O(N)** memory use).

- **Performance Metrics**  
  Shows average execution time, time difference (ms), and generator speedup factor.

- **Self-Contained Code**  
  All logic lives inside `interactive_benchmark.py` for easy review and deployment.

---

## 🧠 Algorithmic Focus: O(N) vs O(h)

This benchmark focuses on In-Order Traversal and compares two approaches:

| Implementation | Technique | Time Complexity | Space Complexity |
|---------------|-----------|-----------------|------------------|
| **Original** | Recursive (List Aggregation) | O(N) | O(N) |
| **Optimized** | Recursive (Generator `yield from`) | O(N) | O(h) |

Lower space usage is crucial for large datasets and often results in faster execution.

---

## 🛠️ Getting Started

### **Prerequisites**
Python 3.9+  
Dependencies:
- `streamlit`
- `pandas`

---

### **1. Installation**

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/YourUsername/real-time-dsa-efficiency-tester.git
cd real-time-dsa-efficiency-tester

# Install requirements
pip install -r requirements.txt
```

### **2. Execution**

Run the Streamlit application:

```bash
streamlit run interactive_benchmark.py
```

Your default browser will open the app automatically.

---

## 💻 Technologies Used

- **Python**  3.x
- **Streamlit** (Interactive UI)
- **Pandas** (Data formatting and display)
- **Data Structures** BST, Recursion, Python Generators
