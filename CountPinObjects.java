public class Solution {
    private final Relation relation;

    public Solution(List<List<Pixel>> pixels) {
        this.relation = new Relation(pixels);
    }

    int countObjects(char[][] grid) {
        int objectCount = 0;
        int row = grid.length;
        int col = grid[0].length;
        boolean[][] visited = new boolean[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == '*' && !visited[i][j]) {
                    dfs(grid, new Pixel(i, j), i, j, visited);
                    objectCount++;
                }
            }
        }
        return objectCount;
    }

    private void dfs(char[][] grid, Pixel rep, int x, int y, boolean[][] visited) {
        if (x < 0 || x >= grid.length || y < 0 || y >= grid[0].length || grid[x][y] != '*' || visited[x][y]) {
            return;
        }
        if (!relation.isSameObject(rep, new Pixel(x, y))) {
            return;
        }
        visited[x][y] = true;
        dfs(grid, rep, x - 1, y, visited);
        dfs(grid, rep, x + 1, y, visited);
        dfs(grid, rep, x, y - 1, visited);
        dfs(grid, rep, x, y + 1, visited);
    }
}
