class PinBoard {
    public int minBoardsToConnect(List<List<Integer>> boards, int pin1, int pin2) {
        if (pin1 == pin2) {
            return 0;
        }
        Map<Integer, List<Integer>> pinToBoardNum = new HashMap<>();
        makeGraph(boards, pinToBoardNum);
        if (!pinToBoardNum.containsKey(pin1) || !pinToBoardNum.containsKey(pin2)) {
            return -1;
        }
        int minBoards = 0;
        Queue<Integer> bfsPinBuffer = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        bfsPinBuffer.offer(pin1);
        while (!bfsPinBuffer.isEmpty()) {
            int layerSize = bfsPinBuffer.size();
            for (int i = 0; i < layerSize; i++) {
                int currentPin = bfsPinBuffer.poll();
                // list of all boards connected to this pin
                List<Integer> connectedBoards = pinToBoardNum.get(currentPin);
                for (int boardNum : connectedBoards) {
                    List<Integer> pinsOnBoard = boards.get(boardNum);
                    for (int pin : pinsOnBoard) {
                        if (pin == pin2) {
                            return minBoards;
                        }
                        if (!visited.contains(pin)) {
                            bfsPinBuffer.offer(pin);
                            visited.add(pin);
                        }
                    }
                }
            }
            minBoards++;
        }
        return -1;
    }

    private void makeGraph(List<List<Integer>> boards, Map<Integer, List<Integer>> pinToBoardNum) {
        for (int i = 0; i < boards.size(); i++) {
            // i is the board number
            List<Integer> board = boards.get(i);
            for (int pin : board) {
                if (!pinToBoardNum.containsKey(pin)) {
                    pinToBoardNum.put(pin, new ArrayList<>());
                }
                pinToBoardNum.get(pin).add(i);
            }
        }
    }
}
