class PinSystem {
    // for each type -> top pins
    Map<String, PriorityQueue<Pin>> typeToTopPins;
    Set<Integer> pinIds;

    public PinSystem() {
        typeToTopPins = new HashMap<>();
        pinIds = new HashSet<>();
    }

    public boolean addPin(int id, double score, String type) {
        // check if pin id already exists
        if (pinIds.contains(id)) {
            return false;
        }
        pinIds.add(id);
        if (!typeToTopPins.containsKey(type)) {
            typeToTopPins.put(type, new PriorityQueue<>((a, b) -> Double.compare(b.score, a.score)));
        }
        typeToTopPins.get(type).offer(new Pin(id, score));
        return true;
    }

    public List<Integer> getTopK(String type, int k) {
        if (!typeToTopPins.containsKey(type)) {
            return new ArrayList<>();
        }
        List<Pin> topKpins = new ArrayList<>();
        PriorityQueue<Pin> topPins = typeToTopPins.get(type);
        int topSize = Math.min(k, topPins.size());
        for (int i = 0; i < topSize; i++) {
            topKpins.add(topPins.poll());
        }
        List<Integer> output = new ArrayList<>();
        // restore original Max Heap
        for (Pin pin : topKpins) {
            topPins.offer(pin);
            output.add(pin.id);
        }
        return output;
    }

    class Pin {
        int id;
        double score;

        public Pin(int id, double score) {
            this.id = id;
            this.score = score;
        }
    }
}
