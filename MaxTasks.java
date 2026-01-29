class MaxTasks {
    // Time O(nlogn) Space O(logn); Greedy
    public int maxTasks(int[][] tasks) {
        if (tasks == null || tasks.length == 0) {
            return 0;
        }
        Arrays.sort(tasks, (a, b) -> (a[1] - b[1]));
        // tail -> current meeting can attend
        int[] tail = tasks[0];
        int maxCanAttend = 1;
        for (int i = 1; i < tasks.length; i++) {
            int[] current = tasks[i];
            // always attend a meeting ending earlier
            // disjoint -> attend both, move tail
            if (!checkOverlap(tail, current)) {
                maxCanAttend++;
                tail = current;
            }
        }
        return maxCanAttend;
    }

    private boolean checkOverlap(int[] taskA, int[] taskB) {
        return Math.max(taskA[0], taskB[0]) < Math.min(taskA[1], taskB[1]);
    }
}
