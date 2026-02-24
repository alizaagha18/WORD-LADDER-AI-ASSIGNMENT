import numpy as np

from codesample import (
    Problem, 
    breadth_first_graph_search, 
    depth_limited_search, 
    uniform_cost_search, 
    best_first_graph_search, 
    astar_search
)

# 1. DATA LOADING & MATH UTILITIES
import numpy as np
from codesample import Problem

import numpy as np
from codesample import Problem

def load_glove_embeddings(filepath):
    #Reads GloVe embedding file and returns a dictionary:
    #{ word : 100-dimensional numpy vector }"""
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()              # split line into tokens
            word = values[0]                  # first token = word
            vector = np.asarray(values[1:], dtype='float32')  # remaining = embedding
            embeddings[word] = vector         # store in dictionary
    return embeddings

# 2. WORD LADDER SEARCH PROBLEM

class WordLadderProblem(Problem):
    """
        initial: start word
        goal: target word
        embeddings_dict: {word : vector}
        k: number of neighbors
        max_limit: max node expansions (safety for GUI)
    """
    def __init__(self, initial, goal, embeddings_dict, k=15, max_limit=10000):
        super().__init__(initial, goal)
        self.k = k
        self.nodes_expanded = 0
        self.max_limit = max_limit 
        
        # Performance optimization: Convert dict to Matrix
        self.words = list(embeddings_dict.keys())
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        matrix = np.array([embeddings_dict[w] for w in self.words]) #embedding matrix
        
        # Normalize vectors for fast Cosine Similarity (dot product)
        # then prestore normalized goal vector for hueristic
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix_norm = matrix / norms
        self.goal_vec = self.matrix_norm[self.word_to_idx[self.goal]]

    def actions(self, state):
        #Returns top-k most similar words (neighbors)
        # Stop expansion if limit reached to protect the GUI
        if self.nodes_expanded >= self.max_limit:
            return []
            
        self.nodes_expanded += 1
        
        idx = self.word_to_idx[state]
        v = self.matrix_norm[idx]
        
        # Calculate similarity to all words at once
        similarities = np.dot(self.matrix_norm, v)
        
        # Find top k neighbors (excluding the  itself)
        top_indices = np.argsort(similarities)[-(self.k + 1):-1]
        return [self.words[i] for i in reversed(top_indices)]

    def result(self, state, action):
        return action #next state = selected neighbour

    def path_cost(self, c, state1, action, state2):
        # Cost = 1 - Cosine Similarity 
        v1 = self.matrix_norm[self.word_to_idx[state1]]
        v2 = self.matrix_norm[self.word_to_idx[state2]]
        return c + (1.0 - np.dot(v1, v2))

    def h(self, node):
        # Heuristic for A* and Greedy 
        #h(n) = 1 - cosine_similarity(n, goal)
        v_node = self.matrix_norm[self.word_to_idx[node.state]]
        return 1.0 - np.dot(v_node, self.goal_vec)
    
# 3. ALGORITHM RUNNER

def run_searches(start_word, goal_word, embeddings):
    """Runs all 5 algorithms and prints their outputs."""
    
    # Gracefully handle out-of-vocabulary words
    if start_word not in embeddings or goal_word not in embeddings:
        print("Error: Start or goal word is not in the vocabulary.")
        return

    # Instantiate the problem
    problem = WordLadderProblem(start_word, goal_word, embeddings, k=15)
    
    print(f"\nSearching for path: '{start_word}' -> '{goal_word}'\n" + "="*40)

    # 1. Breadth-First Search
    print("Running BFS...")
    bfs_node = breadth_first_graph_search(problem)
    print_result("BFS", bfs_node)

    # 2. Depth-First Search (Using depth-limited to prevent infinite loops)
    print("Running DFS (Depth Limited to 20)...")
    dfs_node = depth_limited_search(problem, limit=100)
    # depth_limited_search returns 'cutoff' if it hits the limit without finding a goal
    if dfs_node == 'cutoff':
        dfs_node = None 
    print_result("DFS", dfs_node)

    # 3. Uniform Cost Search
    print("Running UCS...")
    ucs_node = uniform_cost_search(problem)
    print_result("UCS", ucs_node)

    # 4. Greedy Best-First Search 
    # (Using best_first_graph_search with our custom heuristic problem.h)
    print("Running Greedy Best-First Search...")
    greedy_node = best_first_graph_search(problem, problem.h)
    print_result("Greedy", greedy_node)

    # 5. A* Search
    print("Running A* Search...")
    astar_node = astar_search(problem)
    print_result("A*", astar_node)


def print_result(algo_name, node):
    """Helper to extract the path and path length from the goal Node."""
    if node is None:
        print(f"[{algo_name}] No path found.\n")
    else:
        # node.path() returns Node objects, we extract the state (word) from each
        path = [n.state for n in node.path()]
        print(f"[{algo_name}] Path found!")
        print(f"   Ladder: {' -> '.join(path)}")
        print(f"   Steps:  {len(path) - 1}")
        print(f"   Cost:   {node.path_cost:.4f}\n")


# 4. MAIN EXECUTION

if __name__ == "__main__":
    # Update the path to wherever your glove file is stored
    glove_file = "glove.100d.20000.txt" 
    
    try:
        glove_dict = load_glove_embeddings(glove_file)
        # Test with a pair
        run_searches("barcellona", "bill", glove_dict)
    except FileNotFoundError:
        print(f"Could not find {glove_file}. Please ensure it is in the same directory.")