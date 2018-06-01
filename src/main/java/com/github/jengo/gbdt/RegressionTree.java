package com.github.jengo.gbdt;

import com.google.common.collect.Lists;

import java.util.List;

/**
 * {@link RegressionTree}
 */
public class RegressionTree {

    private static final double UNKNOWN_THRESHOLD = -10000.0;

    /** 根节点 */
    private Node root;
    /** 树节点按列表存储 */
    private List<Node> nodes;
    /** 叶节点个数 */
    private int leafSize = 0;

    public int getLeafSize() {
        return this.leafSize;
    }

    private static class NodeChildIndex {
        public int lt;
        public int ge;
        public int un;
    }

    public void load(String treeInLine) throws Exception {
        String[] nodeInLines = treeInLine.split("\n");

        List<NodeChildIndex> indexes = Lists.newArrayList();
        this.nodes = Lists.newArrayList();
        this.leafSize = 0;
        for (String nodeInLine : nodeInLines) { //建立节点链表
            String[] fields = nodeInLine.split(" ");

            Node node = new Node();
            node.feaIdx = Integer.parseInt(fields[0]);
            node.feaThr = Double.parseDouble(fields[1]);
            try {
                node.isLeaf = (Integer.parseInt(fields[2]) > 0);
            } catch (Exception e) {
                node.isLeaf = Boolean.parseBoolean(fields[2]);
            }
            node.score = Double.parseDouble(fields[3]);
            if (node.isLeaf) {
                node.leafIdx = this.leafSize;
                this.leafSize++;
            }
            nodes.add(node);

            NodeChildIndex idx = new NodeChildIndex();
            idx.lt = Integer.parseInt(fields[4]);
            idx.ge = Integer.parseInt(fields[5]);
            idx.un = Integer.parseInt(fields[6]);
            indexes.add(idx);
        }

        for (int i = 0; i < this.nodes.size(); i++) { //建立节点之间的关系
            NodeChildIndex idx = indexes.get(i);
            Node node = this.nodes.get(i);
            if (idx.lt > 0) {
                node.children[Node.LT] = this.nodes.get(idx.lt);
            }
            if (idx.ge > 0) {
                node.children[Node.GE] = this.nodes.get(idx.ge);
            }
            if (idx.un > 0) {
                node.children[Node.UNKNOWN] = this.nodes.get(idx.un);
            }
        }

        this.root = this.nodes.get(0);
    }

    public double predict(Tuple t) {
        return predict(root, t);
    }

    public int locateLeaf(Tuple t) {
        return locateLeaf(root, t);
    }

    public double predict(Tuple t, List<Double> w) {
        return predict(root, t, w);
    }

    private static int locateLeaf(Node node, Tuple t) {
        if (node.isLeaf)
            return node.leafIdx;

        Node childNode = locateChildNode(node, t);
        if (childNode != null) {
            return locateLeaf(childNode, t);
        } else {
            return node.leafIdx;
        }
    }

    private static double predict(Node node, Tuple t) {
        if (node.isLeaf)
            return node.score;

        Node childNode = locateChildNode(node, t);
        if (childNode != null) {
            return predict(childNode, t);
        } else {
            return node.score;
        }
    }

    private static double predict(Node node, Tuple t, List<Double> w) {
        if (node.isLeaf) {
            return node.score;
        }
        Node childNode = locateChildNode(node, t);
        if (childNode != null) {
            double x = (childNode.score - node.score);
            x += w.get(node.feaIdx);
            w.set(node.feaIdx, x);
            return predict(childNode, t, w);
        } else {
            return node.score;
        }
    }

    private static Node locateChildNode(Node node, Tuple t) {
        double nodeVal = t.feaVals.get(node.feaIdx);
        if (nodeVal <= UNKNOWN_THRESHOLD) {
            return node.children[Node.UNKNOWN];
        } else if (nodeVal < node.feaThr) {
            return node.children[Node.LT];
        } else {
            return node.children[Node.GE];
        }
    }

}
