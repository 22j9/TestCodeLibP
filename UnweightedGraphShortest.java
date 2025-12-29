import java.util.*;
import java.math.*;

class UnweightedGraphShortest {
    public int shortestDistance(int n, int[][] edges, int source, int target) {
        // TODO: Implement shortestDistance logic
        if (source == target) {
            return 0;
        }
        Map<Integer, List<Integer>> graph = new HashMap<>();
        makeGraph(n, edges, graph);
        if (!graph.containsKey(source) || !graph.containsKey(target)) {
            return -1;
        }
        int distance = 0;
        Queue<Integer> bfsGraphBuffer = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        bfsGraphBuffer.offer(source);
        visited.add(source);
        while (!bfsGraphBuffer.isEmpty()) {
            int layerSize = bfsGraphBuffer.size();
            for (int i = 0; i < layerSize; i++) {
                int current = bfsGraphBuffer.poll();
                for (int next : graph.get(current)) {
                    if (next == target) {
                        return distance + 1;
                    }
                    if (!visited.contains(next)) {
                        bfsGraphBuffer.offer(next);
                        visited.add(next);
                    }
                }
            }
            distance++;
        }
        return -1;
    }

    private void makeGraph(int n, int[][] edges, Map<Integer, List<Integer>> graph) {
        // process vertices
        for (int i = 0; i < n; i++) {
            graph.put(i, new ArrayList<>());
        }
        // process edges
        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }
    }
}
