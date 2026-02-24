# 🧩 AI Semantic Word Ladder Explorer

This project implements a **semantic word ladder search system** using pretrained GloVe word embeddings. The goal is to find a path between two words by navigating through semantically similar intermediate words using classical search algorithms.

## ✨ Features
- Uses **GloVe 100-dimensional embeddings** (20,000 vocabulary)
- Implements multiple search strategies:
  - Breadth-First Search (BFS)
  - Depth-First Search (DFS)
  - Uniform Cost Search (UCS)
  - Greedy Best-First Search
  - A* Search
- Similarity-based path cost
- Cosine similarity heuristic
- Interactive GUI built with **Streamlit**
- Adjustable number of neighbors (*k*)
- Performance metrics:
  - Steps
  - Nodes expanded
  - Path cost
  - Execution time

## 🛠️ Technologies Used
- Python 3
- NumPy
- Streamlit
- Virtual environment (venv)

## ⚙️ Setup Instructions

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
2. **Activate Virtual Environment**
- Windows
venv\Scripts\activate
- Mac/Linux
source venv/bin/activate

3. **Install Dependencies**
pip install numpy streamlit

**🚀 Running the Application:**
Make sure glove.100d.20000.txt is in the project directory.
Then run:
streamlit run app.py
The application will open in your browser.

**🔍 How It Works**
- Each word is treated as a state in a search space.
- From a given word, the algorithm moves to its top-k nearest neighbors in embedding space using cosine similarity.
- Different search strategies explore this semantic space differently:
- BFS → explores level by level
- DFS → explores deeply
- UCS → considers cumulative semantic cost
- Greedy → uses heuristic only
- A* → balances cost and heuristic
- 
**🎓 Academic Context**:This project was developed as part of an Introduction to Artificial Intelligence course assignment.
