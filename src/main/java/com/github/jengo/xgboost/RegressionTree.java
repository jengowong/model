package com.github.jengo.xgboost;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import java.util.List;
import java.util.Map;

/**
 * {@link RegressionTree}
 */
public class RegressionTree {

    private static final int NO_CHILD_NODE = -1000;

    /** 根节点 */
    private Node root;
    /** 树节点按列表存储 */
    private List<Node> nodes = Lists.newArrayList();

    public void load(String treeInLine) throws Exception {
        String[] nodeInLines = treeInLine.split("\n");

        Map<Integer, Node> nodeIdx2Node = Maps.newHashMap();
        Map<Integer, NodeChildIndex> nodeIdx2Children = Maps.newHashMap();

        for (int i = 1; i < nodeInLines.length; i++) { //建立节点链表
            String[] fields = nodeInLines[i].split("[:,=>< \\[\\]]");
            Node node = new Node();
            NodeChildIndex nodeChildIndex = new NodeChildIndex();
            if (fields.length == 11) {
                node.nodeIdx = Integer.parseInt(fields[0].trim());
                node.feaIdx = Integer.parseInt(fields[2].substring(1));
                node.feaThr = Double.parseDouble(fields[3]);
                node.score = 0.0D;
                node.isLeaf = false;
                nodeChildIndex.leftChildIndex = Integer.parseInt(fields[6]);
                nodeChildIndex.rightChildIndex = Integer.parseInt(fields[8]);
                if (fields[10].equals(fields[6])) {
                    node.missingThenTakeLeftChild = true;
                } else if (fields[10].equals(fields[8])) {
                    node.missingThenTakeLeftChild = false;
                } else {
                    System.out.println("Error! missing: nodeInLine=" + nodeInLines[i]);
                    System.out.println("Error! missing: treeInLine=" + treeInLine);
                }
            } else if (fields.length == 3) {
                node.nodeIdx = Integer.parseInt(fields[0].trim());
                node.feaIdx = -1;
                node.feaThr = -100000.0D;
                node.score = Double.parseDouble(fields[2]);
                node.isLeaf = true;
                nodeChildIndex.leftChildIndex = NO_CHILD_NODE;
                nodeChildIndex.rightChildIndex = NO_CHILD_NODE;
            } else {
                System.out.println("Error! nodeFields.length=" + fields.length);
            }

            this.nodes.add(node);
            nodeIdx2Node.put(node.nodeIdx, node);
            nodeIdx2Children.put(node.nodeIdx, nodeChildIndex);
        }
        for (Node node : this.nodes) { //建立节点之间的关系
            NodeChildIndex nodeChildIndex = nodeIdx2Children.get(node.nodeIdx);
            if (NO_CHILD_NODE != nodeChildIndex.leftChildIndex) {
                node.leftChild = nodeIdx2Node.get(nodeChildIndex.leftChildIndex);
            }
            if (NO_CHILD_NODE != nodeChildIndex.rightChildIndex) {
                node.rightChild = nodeIdx2Node.get(nodeChildIndex.rightChildIndex);
            }
        }
        this.root = this.nodes.get(0);
    }

    public double predict(Tuple t) {
        return predict(this.root, t);
    }

    private static double predict(Node node, Tuple tuple) {
        if (node.isLeaf) {
            return node.score;
        } else if (tuple.feaIdx2Val.containsKey(node.feaIdx)) {
            double feaVal = tuple.feaIdx2Val.get(node.feaIdx);
            return feaVal < node.feaThr
                    ? (
                    node.leftChild != null
                            ? predict(node.leftChild, tuple)
                            : node.score)
                    : (
                    node.rightChild != null
                            ? predict(node.rightChild, tuple)
                            : node.score);
        } else {
            return node.missingThenTakeLeftChild
                    ? (
                    node.leftChild != null
                            ? predict(node.leftChild, tuple)
                            : node.score)
                    : (
                    node.rightChild != null
                            ? predict(node.rightChild, tuple)
                            : node.score);
        }
    }

    private static class NodeChildIndex {
        public int leftChildIndex;
        public int rightChildIndex;

        @Override
        public String toString() {
            return ToStringBuilder.reflectionToString(this, ToStringStyle.JSON_STYLE);
        }
    }

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this, ToStringStyle.JSON_STYLE);
    }

}
