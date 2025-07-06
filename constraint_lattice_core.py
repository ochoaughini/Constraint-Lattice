class ConstraintLattice:
    def __init__(self):
        self.nodes = {}
        self.constraints = []

    def add_node(self, node_id, value=None):
        self.nodes[node_id] = value

    def add_constraint(self, constraint_func, inputs, outputs):
        self.constraints.append({
            'func': constraint_func,
            'inputs': inputs,
            'outputs': outputs
        })

    def propagate(self):
        from collections import defaultdict
        
        # Build dependency graph: node -> set of constraint indices that depend on it
        dep_graph = defaultdict(set)
        for idx, constraint in enumerate(self.constraints):
            for input_id in constraint['inputs']:
                dep_graph[input_id].add(idx)
        
        # Build constraint graph: constraint index -> set of constraint indices that depend on its outputs
        constraint_graph = defaultdict(set)
        for idx, constraint in enumerate(self.constraints):
            for output_id in constraint['outputs']:
                for dependent_idx in dep_graph.get(output_id, set()):
                    constraint_graph[idx].add(dependent_idx)
        
        # Find SCCs in the constraint graph
        sccs = self._tarjan_scc(constraint_graph)
        # We want to process in topological order (so we reverse the list of SCCs returned by Tarjan)
        sccs_topological = list(reversed(sccs))
        
        print(f"Found {len(sccs_topological)} SCCs")
        for i, scc in enumerate(sccs_topological):
            print(f"Processing SCC {i}: {scc}")
            changed = True
            iteration = 0
            while changed:
                iteration += 1
                print(f"  Iteration {iteration}")
                changed = False
                for constraint_idx in scc:
                    constraint = self.constraints[constraint_idx]
                    input_vals = [self.nodes[i] for i in constraint['inputs']]
                    print(f"    Applying constraint {constraint_idx}: {constraint['func'].__name__} on inputs {input_vals}")
                    output_vals = constraint['func'](*input_vals)
                    if not isinstance(output_vals, tuple):
                        output_vals = (output_vals,)
                    for i, output_id in enumerate(constraint['outputs']):
                        old_val = self.nodes[output_id]
                        new_val = output_vals[i]
                        if old_val != new_val:
                            print(f"      Updating {output_id} from {old_val} to {new_val}")
                            self.nodes[output_id] = new_val
                            changed = True

    def _tarjan_scc(self, graph):
        """Tarjan's algorithm for strongly connected components."""
        index = 0
        stack = []
        indices = {}
        lowlinks = {}
        on_stack = {}
        sccs = []

        def strongconnect(node):
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack[node] = True

            for neighbor in graph.get(node, set()):
                if neighbor not in indices:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif on_stack[neighbor]:
                    lowlinks[node] = min(lowlinks[node], indices[neighbor])

            if lowlinks[node] == indices[node]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == node:
                        break
                sccs.append(scc)

        for node in graph:
            if node not in indices:
                strongconnect(node)

        return sccs
