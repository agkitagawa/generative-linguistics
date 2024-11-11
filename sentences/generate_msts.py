import os
import numpy as np
import networkx as nx

base_folder = "/Users/annakitagawa/Downloads/Research/PTB/WSJ_sentence_samples/"
embeddings_folder = f"{base_folder}embeddings/"
mst_folder = f"{base_folder}msts/"

embedding_map = {}
count = 0

for root, dirs, files in os.walk(embeddings_folder):
    for filename in files:
        file_path = os.path.join(root, filename)
        with open(file_path, 'r') as file:
            arr = np.loadtxt(file, dtype=np.float32)

        for i, embedding in enumerate(arr):
            embedding_map[i] = embedding

        num_embeddings = len(embedding_map)
        distance_matrix = np.zeros((num_embeddings, num_embeddings))

        for i in range(num_embeddings):
            embedding1 = embedding_map[i]

            for j in range(num_embeddings):
                embedding2 = embedding_map[j]

                dist = np.linalg.norm(embedding1 - embedding2)
                distance_matrix[i, j] = dist

        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [1] * size

            def find(self, u):
                if self.parent[u] != u:
                    self.parent[u] = self.find(self.parent[u])
                return self.parent[u]

            def union(self, u, v):
                rootU = self.find(u)
                rootV = self.find(v)

                if rootU != rootV:
                    if self.rank[rootU] > self.rank[rootV]:
                        self.parent[rootV] = rootU
                    elif self.rank[rootU] < self.rank[rootV]:
                        self.parent[rootU] = rootV
                    else:
                        self.parent[rootV] = rootU
                        self.rank[rootU] += 1

        # doesn't deal with case of multiple intersecting at same time
        def kruskal(num_points, distance_matrix):
            edges = []
            
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    edges.append((i, j, distance_matrix[i][j]))
            
            edges.sort(key=lambda x: x[2])
            
            uf = UnionFind(num_points)
            mst_edges = []
            
            for u, v, weight in edges:
                if uf.find(u) != uf.find(v):
                    uf.union(u, v)
                    mst_edges.append((u, v, weight)) 
            
            return mst_edges

        mst = kruskal(num_embeddings, distance_matrix)

        stripped_edges = []
        neighbors = {}

        for v1, v2, dist in mst:
            stripped_edges.append((v1, v2))
            if v1 not in neighbors:
                neighbors[v1] = []
            if v2 not in neighbors:
                neighbors[v2] = []
            
            neighbors[v1].append(v2)
            neighbors[v2].append(v1)

        # check graph for cycle
        def dfs(v, visited, parent, graph):
            visited[v] = True
            for neighbor in graph[v]:
                if not visited[neighbor]:
                    if dfs(neighbor, visited, v, graph):
                        return True
                elif neighbor != parent:
                    return True
            return False

        def has_cycle(graph):
            visited = [False] * len(graph)
            for v in range(len(graph)):
                if not visited[v]:
                    if dfs(v, visited, -1, graph):
                        return True
            return False

        if has_cycle(neighbors):
            print("cycle detected")

        G = nx.Graph()
        G.add_nodes_from(list(range(num_embeddings)))
        G.add_edges_from(stripped_edges)


        file = f"{mst_folder}medium_sample_mst_{count}.edgelist"
        nx.write_edgelist(G, file)

        count += 1