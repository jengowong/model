package com.github.jengo.gbdt;

/**
 * {@link Node}
 */
public class Node {
    public static final int LT = 0;
    public static final int GE = 1;
    public static final int UNKNOWN = 2;
    public static final int CHILDSIZE = 3;

    /** feature index */
    public int feaIdx = -1;
    /** feature threshold */
    public double feaThr = 0;
    /** predict score */
    public double score = 0;
    /** is leaf */
    public boolean isLeaf = false;
    /** leaf index */
    public int leafIdx = -1;
    /** children */
    public Node[] children = {null, null, null};
}
