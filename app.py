# ==========================================
# STREAMLIT GUI FOR SEMANTIC WORD LADDER
# ==========================================

import streamlit as st
import time
import numpy as np

# Import problem definition and embedding loader from main.py
from main import load_glove_embeddings, WordLadderProblem

# Import search algorithms
from codesample import (
    breadth_first_graph_search, 
    depth_limited_search, 
    uniform_cost_search,
    best_first_graph_search, 
    astar_search
)

# ==========================================
# LOAD EMBEDDINGS (CACHED)
# ==========================================

@st.cache_data
def get_embeddings():
    """
    Loads GloVe embeddings once and caches them.
    This prevents reloading the 20,000-word file every time
    the user interacts with the app.
    """
    try:
        return load_glove_embeddings("glove.100d.20000.txt")
    except FileNotFoundError:
        st.error("GloVe file not found! Ensure 'glove.100d.20000.txt' is in the folder.")
        return None


# ==========================================
# MAIN UI TITLE
# ==========================================

st.title("🔤 AI Semantic Word Ladder Explorer")

# Load embeddings
embeddings = get_embeddings()


# ==========================================
# SIDEBAR CONTROLS
# ==========================================

if embeddings:
    
    # Sidebar configuration panel
    st.sidebar.header("Search Settings")
    
    # User inputs start word
    start_word = st.sidebar.text_input("Start Word", "panah").lower().strip()
    
    # User inputs goal word
    goal_word = st.sidebar.text_input("Goal Word", "haloze").lower().strip()
    
    # Algorithm selection dropdown
    algo_choice = st.sidebar.selectbox(
        "Algorithm", 
        ["BFS", "DFS", "UCS", "Greedy", "A*"]
    )
    
    # Slider to control number of neighbors (branching factor)
    k_val = st.sidebar.slider("Neighbors (k)", 5, 50, 15)
    
    # Limit to prevent excessive node expansions (GUI safety)
    max_nodes = st.sidebar.number_input(
        "Max Expansion Limit", 
        1000, 50000, 10000
    )

    # ==========================================
    # RUN SEARCH BUTTON
    # ==========================================

    if st.sidebar.button("Run Search"):
        
        # Validate that words exist in vocabulary
        if start_word not in embeddings or goal_word not in embeddings:
            st.error("One of the words is not in the vocabulary!")
        
        else:
            # Create Word Ladder problem instance
            prob = WordLadderProblem(
                start_word, 
                goal_word, 
                embeddings, 
                k=k_val, 
                max_limit=max_nodes
            )
            
            # Start timing execution
            start_t = time.time()
            res = None
            
            # Show loading spinner while algorithm runs
            with st.spinner(f"Running {algo_choice}..."):
                
                # Run selected algorithm
                if algo_choice == "BFS":
                    res = breadth_first_graph_search(prob)
                
                elif algo_choice == "DFS":
                    # Depth-limited to prevent infinite exploration
                    res = depth_limited_search(prob, limit=40)
                    if res == 'cutoff':
                        res = None
                
                elif algo_choice == "UCS":
                    res = uniform_cost_search(prob)
                
                elif algo_choice == "Greedy":
                    # Uses heuristic only
                    res = best_first_graph_search(prob, prob.h)
                
                elif algo_choice == "A*":
                    res = astar_search(prob)
            
            # Stop timing
            duration = time.time() - start_t
        
        # ==========================================
        # DISPLAY RESULTS
        # ==========================================

        if res:
            # Extract word sequence from Node path
            path = [n.state for n in res.path()]
            
            st.success("Path Found!")
            
            # Display evaluation metrics (required for assignment)
            m1, m2, m3, m4 = st.columns(4)
            
            m1.metric("Steps", len(path) - 1)
            m2.metric("Nodes Expanded", prob.nodes_expanded)
            m3.metric("Path Cost", f"{res.path_cost:.4f}")
            m4.metric("Time Taken", f"{duration:.4f}s")
            
            # Display semantic ladder
            st.markdown("### Semantic Ladder:")
            st.info(" ➔ ".join([f"**{w}**" for w in path]))
        
        else:
            # If no path found
            st.error("No path found.")
            
            # Still show performance statistics
            m1, m2 = st.columns(2)
            m1.metric("Nodes Expanded", prob.nodes_expanded)
            m2.metric("Time Taken", f"{duration:.4f}s")