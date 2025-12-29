package org.example.done;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;

public class Pinterest {
    int[] dirs = {0, 1, 0, -1, 0};

    public static String roundDecimalString(String decimalStr, int places) {
        if (!decimalStr.contains(".")) {
            return decimalStr;  // No decimal point means it's an integer, nothing to round
        }

        String[] parts = decimalStr.split("\\.");
        String integerPart = parts[0];
        String decimalPart = parts[1];

        if (decimalPart.length() <= places) {
            return decimalStr;  // Nothing to round if the decimal part is shorter or equal to the desired places
        }

        // The digit right after the rounding place
        int roundingDigit = Character.getNumericValue(decimalPart.charAt(places));

        StringBuilder rounded = new StringBuilder(integerPart + "." + decimalPart.substring(0, places));

        if (roundingDigit >= 5) {
            // We need to round up
            for (int i = rounded.length() - 1; i >= 0; i--) {
                if (rounded.charAt(i) == '.') {
                    continue;  // Skip the decimal point
                }
                int currentDigit = Character.getNumericValue(rounded.charAt(i));
                if (currentDigit < 9) {
                    rounded.setCharAt(i, Character.forDigit(currentDigit + 1, 10));
                    break;
                } else {
                    rounded.setCharAt(i, '0');
                }
                if (i == 0) {
                    rounded.insert(0, '1');
                }
            }
        }

        return rounded.toString();
    }

    //-----------------------------------------------------------------------------------

    public static void main(String[] args) {
        Pinterest pinterest = new Pinterest();
        System.out.println(pinterest.restaurantReservation(800, 2100, 5, List.of(
                List.of("900", "930", "3"),
                List.of("915", "945", "2")
        )));
    }

    // https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/
    // Method to find the shortest path in a grid with obstacles that can be eliminated
    public int shortestPath(int[][] grid, int k) {
        int res = 0; // Initialize the result, which will store the shortest path length.
        // Create a visited array to keep track of the maximum remaining obstacles that can be eliminated at each cell.
        int[][] visited = new int[grid.length][grid[0].length];
        for (int[] r : visited) {
            Arrays.fill(r, -1); // Initialize visited array with -1, indicating that no cell has been visited yet.
        }
        Queue<int[]> q = new ArrayDeque<>(); // Use a queue to perform BFS (Breadth-First Search).
        q.add(new int[]{0, 0, k}); // Add the starting point to the queue with the initial number of obstacles that can be eliminated.
        visited[0][0] = k; // Mark the starting point as visited with the initial number of obstacles that can be eliminated.
        // Direction array to help move in 4 directions (right, down, left, up).
        int[] dirs = new int[]{1, 0, -1, 0, 1};
        while (!q.isEmpty()) { // Continue until the queue is empty.
            int l = q.size(); // Process nodes level by level.
            while (l-- > 0) { // Iterate through all nodes at the current level.
                int[] curr = q.poll(); // Retrieve and remove the head of the queue.
                int i = curr[0], j = curr[1], r = curr[2]; // Current row, column, and remaining obstacles that can be eliminated.
                if (i == grid.length - 1 && j == grid[0].length - 1) { // Check if the destination is reached.
                    return res; // Return the number of steps taken to reach the destination.
                }
                for (int d = 0; d < 4; d++) { // Explore all 4 directions.
                    int ni = i + dirs[d], nj = j + dirs[d + 1]; // Calculate the new position.
                    if (ni < 0 || ni >= grid.length || nj < 0 || nj >= grid[0].length)
                        continue; // Check if the new position is out of bounds.
                    int nr = r - grid[ni][nj]; // Calculate the remaining obstacles that can be eliminated after moving to the new position.
                    if (nr < 0 || nr <= visited[ni][nj])
                        continue; // Skip if no more obstacles can be eliminated or if a better path has been visited.
                    q.add(new int[]{ni, nj, nr}); // Add the new position to the queue.
                    visited[ni][nj] = nr; // Update the visited array with the new number of remaining obstacles that can be eliminated.
                }
            }
            ++res; // Increment the result after processing all nodes at the current level.
        }
        return -1; // Return -1 if the destination cannot be reached within the given constraints.
    }

    //-----------------------------------------------------------------------------------

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();

        dfs(candidates, target, 0, temp, res);
        return res;
    }

    //-----------------------------------------------------------------------------------

    private void dfs(int[] candidates, int target, int cur, List<Integer> temp, List<List<Integer>> res) {
        if (cur >= candidates.length || target <= 0) {
            if (target == 0) {
                res.add(new ArrayList<>(temp));
            }
            return;
        }
        for (int i = cur; i < candidates.length; i++) {
            temp.add(candidates[i]);
            dfs(candidates, target - candidates[i], i, temp, res);
            temp.remove(temp.size() - 1);
        }
    }

    //-----------------------------------------------------------------------------------

    public int shortestWay(String source, String target) {
        boolean[] cnt = new boolean[26]; // Array to count occurrences of each character in 'source'
        for (char c : source.toCharArray()) {
            cnt[c - 'a'] = true; // Increment the count for each character found in 'source'
        }
        for (char c : target.toCharArray()) {
            if (!cnt[c - 'a']) {
                return -1; // If any character in 'target' is not found in 'source', return -1
            }
        }
        int j = 0, res = 1; // 'j' tracks the current position in 'target', 'res' counts the number of subsequences
        while (j < target.length()) { // Iterate through 'target'
            int i = 0; // 'i' tracks the current position in 'source'
            // match as many characters as possible in 'source' with 'target'
            while (i < source.length() && j < target.length()) { // Iterate through 'source' and 'target'
                if (source.charAt(i) == target.charAt(j)) {
                    j++; // If characters match, move to the next character in 'target'
                }
                i++; // Move to the next character in 'source'
            }
            if (j < target.length()) { // If not at the end of 'target', increment 'res' for a new subsequence
                res++;
            }
        }
        return res; // Return the number of subsequences needed to form 'target' from 'source'
    }

    //-----------------------------------------------------------------------------------

    public String reorganizeString(String s) {
        // Create map of each char to its count
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            int count = map.getOrDefault(c, 0) + 1;
            // Impossible to form a solution
            if (count > (s.length() + 1) / 2) {
                return "";
            }
            map.put(c, count);
        }
        // Greedy: fetch char of max count as next char in the result.
        // Use PriorityQueue to store pairs of (char, count) and sort by count DESC.
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> b[1] - a[1]);
        for (char c : map.keySet()) {
            pq.add(new int[]{c, map.get(c)});
        }
        // Build the result.
        StringBuilder sb = new StringBuilder();
        while (!pq.isEmpty()) {
            int[] first = pq.poll();
            if (sb.isEmpty() || first[0] != sb.charAt(sb.length() - 1)) {
                sb.append((char) first[0]);
                if (--first[1] > 0) {
                    pq.add(first);
                }
            } else {
                if (pq.isEmpty()) {
                    return sb.toString();
                }
                int[] second = pq.poll();
                sb.append((char) second[0]);
                if (--second[1] > 0) {
                    pq.add(second);
                }
                pq.add(first);
            }
        }
        return sb.toString();
    }

    public int numIslands(char[][] grid) {
        int res = 0;
        if (grid == null || grid.length == 0 || grid[0].length == 0) return res;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    res++;
                    dfs(grid, i, j);
                }
            }
        }
        return res;
    }

    void dfs(char[][] grid, int i, int j) {
        if (i >= 0 && i < grid.length && j >= 0 && j < grid[0].length && grid[i][j] == '1') {
            grid[i][j] = '0';
            for (int d = 0; d < 4; d++) {
                dfs(grid, i + dirs[d], j + dirs[d + 1]);
            }
        }
    }

    //-----------------------------------------------------------------------------------

    List<List<String>> spamCalls(List<List<String>> calls, List<List<String>> spams) {
        Map<String, Integer> spamMap = new HashMap<>();
        for (List<String> spam : spams) {
            spamMap.put(spam.get(0), Integer.parseInt(spam.get(1)));
        }

        Map<String, Map<String, Integer>> counts = new HashMap<>();
        for (var call : calls) {
            String caller = call.get(0);
            String callee = call.get(1);
            Integer time = Integer.parseInt(call.get(2));
            if (spamMap.containsKey(caller) && Math.abs(time - spamMap.get(caller)) <= 2) {
                counts.putIfAbsent(caller, new HashMap<>());
                counts.get(caller).put(callee, counts.get(caller).getOrDefault(callee, 0) + 1);
            }
        }

        List<List<String>> res = new ArrayList<>();
        for (var entry : counts.entrySet()) {
            String caller = entry.getKey();
            for (var calleeEntry : entry.getValue().entrySet()) {
                String callee = calleeEntry.getKey();
                int count = calleeEntry.getValue();
                res.add(List.of(caller, callee, String.valueOf(count)));
            }
        }
        return res;
    }

    //-----------------------------------------------------------------------------------

    public int totalWaitTime(int n, int m, int[] times) {
        // there are more agents than customers
        if (n > m) {
            return 0;
        }

        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        for (int i = 0; i < times.length; i++) { // O(nlogn)
            pq.add(new int[]{0, i});
        }

        for (int i = 1; i <= m; i++) { // m * log(n)
            int[] curr = pq.poll();
            int endTime = curr[0];
            int index = curr[1];
            // if two agents become free at the same time then one of them will be picked randomly - since we are only interested in the wait time
            // for the customer but not how early they finish. Finishing early can be made the second priority when end times are equal by pushing
            // {endTime, times[i], i} -> priority order => endTime > times[i] > i (if one is equal the least of others will be picked)
            pq.add(new int[]{endTime + times[index], index});
        }

        // we can also return pq.top().second for the agent to which this customer is assigned
        int endTime = pq.peek()[0];
        int res = times[pq.peek()[1]];
        while (!pq.isEmpty() && pq.peek()[0] == endTime) {
            int[] curr = pq.poll();
            int index = curr[1];
            if (times[index] < res) {
                res = times[index];
            }
        }

        return res;
    }

    //-----------------------------------------------------------------------------------

    public String countAndSay(int n) {
        String num = "1";
        while (n-- > 1) {
            num = countAndSay(num);
        }
        return num;
    }

    private String countAndSay(String num) {
        StringBuilder sb = new StringBuilder();
        int count, n = num.length();
        for (int i = 0, j = 0; i < n; i = j) {
            while (j < n && num.charAt(j) == num.charAt(i)) {
                j++;
            }
            sb.append(j - i).append(num.charAt(i));
        }
        return sb.toString();
    }

    //-----------------------------------------------------------------------------------

    public List<String> reverseCountAndSay(String input) {
        if (null == input || input.length() <= 1) return new ArrayList<>();
        List<String> result = new ArrayList<>();
        if (input.length() == 2) {
            int count = Integer.parseInt(input.substring(0, 1));
            char digit = input.charAt(1);
            result.add(String.valueOf(digit).repeat(Math.max(0, count)));
            return result;
        } else if (input.length() == 3) {
            int count = Integer.parseInt(input.substring(0, 2));
            char digit = input.charAt(2);
            result.add(String.valueOf(digit).repeat(Math.max(0, count)));
            return result;
        } else {
            int count = Integer.parseInt(input.substring(0, input.length() - 1));
            char digit = input.charAt(input.length() - 1);
            result.add(String.valueOf(digit).repeat(Math.max(0, count)));
            for (int i = 2; i < input.length() - 1; i++) {
                List<String> left = reverseCountAndSay(input.substring(0, i));
                List<String> right = reverseCountAndSay(input.substring(i));
                for (String l : left) {
                    for (String r : right) {
                        result.add(l + r);
                    }
                }
            }
        }

        return result;
    }

    //-----------------------------------------------------------------------------------

    public List<String> findItinerary(List<List<String>> tickets) {
        Map<String, PriorityQueue<String>> targets = new HashMap<>();
        for (List<String> ticket : tickets) {
            targets.computeIfAbsent(ticket.get(0), k -> new PriorityQueue<>()).add(ticket.get(1));
        }
        List<String> route = new LinkedList<>();
        Stack<String> stack = new Stack<>();
        stack.push("JFK");
        while (!stack.empty()) {
            if (targets.containsKey(stack.peek()) && !targets.get(stack.peek()).isEmpty()) {
                stack.push(targets.get(stack.peek()).poll());
            } else {
                route.add(0, stack.pop());
            }
        }
        return route;
    }

    public List<List<String>> restaurantReservation(int open, int close, int capacity, List<List<String>> reservations) {
        List<List<String>> res = new ArrayList<>();
        Map<Integer, Integer> map = new TreeMap<>();
        map.put(open, capacity);
        for (List<String> r : reservations) {
            int start = Integer.parseInt(r.get(0));
            int end = Integer.parseInt(r.get(1));
            int c = Integer.parseInt(r.get(2));
            if (start >= open && end <= close) {
                map.put(start, map.getOrDefault(start, 0) - c);
                map.put(end, map.getOrDefault(end, 0) + c);
            }
        }

        Integer prev = null, prevCap = null;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            int time = entry.getKey();
            int diff = entry.getValue();
            if (prev == null) {
                prev = time;
                prevCap = diff;
                continue;
            }
            res.add(List.of(String.valueOf(prev), String.valueOf(time), String.valueOf(prevCap)));
            prev = time;
            prevCap = prevCap + diff;
        }
        res.add(List.of(String.valueOf(prev), String.valueOf(close), String.valueOf(prevCap)));
        return res;
    }

    //-----------------------------------------------------------------------------------

    public List<int[]> employeeFreeTime(List<List<int[]>> schedule) {
        List<int[]> allSchedules = new ArrayList<int[]>();
        for (List<int[]> list : schedule)
            allSchedules.addAll(list);
        allSchedules.sort((interval1, interval2) -> {
            if (interval1[0] != interval2[0])
                return interval1[0] - interval2[0];
            else
                return interval1[1] - interval2[1];
        });

        List<int[]> sorted = new ArrayList<int[]>();
        int[] interval0 = allSchedules.get(0);
        int curStart = interval0[0], curEnd = interval0[1];
        int size = allSchedules.size();
        for (int i = 1; i < size; i++) {
            int[] interval = allSchedules.get(i);
            if (interval[0] <= curEnd) {
                curEnd = Math.max(curEnd, interval[1]);
            } else {
                sorted.add(new int[]{curStart, curEnd});
                curStart = interval[0];
                curEnd = interval[1];
            }
        }
        sorted.add(new int[]{curStart, curEnd});

        List<int[]> freeTimeList = new ArrayList<>();
        for (int i = 1; i < sorted.size(); i++) {
            freeTimeList.add(new int[]{sorted.get(i - 1)[1], sorted.get(i)[0]});
        }
        return freeTimeList;
    }

    //-----------------------------------------------------------------------------------

    public int minTotalDistance(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        List<Integer> rows = new ArrayList<>();
        List<Integer> cols = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    rows.add(i);
                    cols.add(j);
                }
            }
        }
        Collections.sort(cols);
        int i = rows.get(rows.size() / 2);
        int j = cols.get(cols.size() / 2);
        return f(rows, i) + f(cols, j);
    }

    private int f(List<Integer> arr, int x) {
        int s = 0;
        for (int v : arr) {
            s += Math.abs(v - x);
        }
        return s;
    }

    //-----------------------------------------------------------------------------------

    public int minTransfers(int[][] transactions) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int[] t : transactions) {
            map.put(t[0], map.getOrDefault(t[0], 0) + t[2]);
            map.put(t[1], map.getOrDefault(t[1], 0) - t[2]);
        }

        List<Integer> list = new ArrayList<>();
        for (int v : map.values()) {
            if (v != 0) {
                list.add(v);
            }
        }

        int[] res = new int[]{Integer.MAX_VALUE};
        dfs(list, 0, 0, res);
        return res[0];
    }

    private void dfs(List<Integer> list, int start, int count, int[] res) {
        while (start < list.size() && list.get(start) == 0) {
            start++;
        }
        if (start == list.size()) {
            res[0] = Math.min(res[0], count);
            return;
        }
        for (int i = start + 1; i < list.size(); i++) {
            if (list.get(start) * list.get(i) < 0) {
                list.set(i, list.get(i) + list.get(start));
                dfs(list, start + 1, count + 1, res);
                list.set(i, list.get(i) - list.get(start));
            }
        }
    }

    //-----------------------------------------------------------------------------------

    public int uniquePathsIII(final int[][] grid) {
        final int m = grid.length, n = grid[0].length;

        int startX = 0, startY = 0, count = 0;

        for (int x = 0; x < m; ++x) {
            for (int y = 0; y < n; ++y) {
                if (grid[x][y] == 0) {
                    count++;
                } else if (grid[x][y] == 1) {
                    startX = x;
                    startY = y;
                }
            }
        }

        return this.dfs(grid, startX, startY, count);
    }

    private int dfs(final int[][] grid, final int x, final int y, int remaining) {
        if (x < 0 || x >= grid.length || y < 0 || y >= grid[0].length || grid[x][y] == -1)
            return 0;

        if (grid[x][y] == 2)
            return remaining == -1 ? 1 : 0;

        remaining--;

        final int tmp = grid[x][y];
        grid[x][y] = -1;

        final int total = dfs(grid, x + 1, y, remaining) + dfs(grid, x, y + 1, remaining) +
                dfs(grid, x - 1, y, remaining) + dfs(grid, x, y - 1, remaining);

        grid[x][y] = tmp;

        return total;
    }

    //-----------------------------------------------------------------------------------

    public boolean canReachTarget(int[] numbers, int target) {
        return dfs(numbers, 0, numbers[0], target);
    }

    private boolean dfs(int[] numbers, int index, int current, int target) {
        if (index == numbers.length - 1) {
            return current == target;
        }

        // Try addition
        if (dfs(numbers, index + 1, current + numbers[index + 1], target)) {
            return true;
        }

        // Try multiplication
        return dfs(numbers, index + 1, current * numbers[index + 1], target);
    }

    //-----------------------------------------------------------------------------------

    public static void deleteNodeAndChildren(int[] parentIndex, int nodeToDelete) {
        // Create a map to store the children of each node
        Map<Integer, List<Integer>> childrenMap = new HashMap<>();

        for (int i = 0; i < parentIndex.length; i++) {
            if (parentIndex[i] != i) {
                childrenMap.computeIfAbsent(parentIndex[i], k -> new ArrayList<>()).add(i);
            }
        }

        // Set to keep track of nodes to delete
        Set<Integer> nodesToDelete = new HashSet<>();
        nodesToDelete.add(nodeToDelete);

        // Use a queue for breadth-first search (BFS)
        Queue<Integer> queue = new LinkedList<>();
        queue.add(nodeToDelete);

        while (!queue.isEmpty()) {
            int currentNode = queue.poll();
            List<Integer> children = childrenMap.get(currentNode);
            if (children != null) {
                for (int child : children) {
                    nodesToDelete.add(child);
                    queue.add(child);
                }
            }
        }

        // Mark the nodes to delete
        for (int node : nodesToDelete) {
            parentIndex[node] = -1; // Assuming -1 indicates a deleted node
        }
    }

    //-----------------------------------------------------------------------------------

    static class AhoCorasick {
        Node root = new Node();

        void insert(String word, int index) {
            Node node = root;
            for (char c : word.toCharArray()) {
                int i = c - 'a';
                if (node.children[i] == null) {
                    node.children[i] = new Node();
                }
                node = node.children[i];
            }
            node.indexes.add(index);
        }

        void build() {
            Queue<Node> q = new LinkedList<>();
            q.add(root);
            while (!q.isEmpty()) {
                Node node = q.poll();
                for (int i = 0; i < 26; i++) {
                    if (node.children[i] != null) {
                        Node fail = node.fail;
                        Node next = node.children[i];
                        while (fail.children[i] == null) {
                            fail = fail.fail;
                        }
                        next.fail = fail.children[i];
                        next.indexes.addAll(fail.children[i].indexes);
                        q.add(next);
                    }
                }
            }
        }

        List<Integer> search(String text) {
            List<Integer> res = new ArrayList<>();
            Node node = root;
            for (int i = 0; i < text.length(); i++) {
                char c = text.charAt(i);
                int j = c - 'a';
                while (node.children[j] == null) {
                    node = node.fail;
                }
                node = node.children[j];
                res.addAll(node.indexes);
            }
            return res;
        }

        static class Node {
            Node[] children = new Node[26];
            Node fail;
            List<Integer> indexes = new ArrayList<>();
        }
    }
}