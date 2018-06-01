package com.github.jengo.xgboost;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

/**
 * {@link Node}
 */
public class Node {
    /** node index */
    public int nodeIdx = -1;
    /** feature index */
    public int feaIdx = -1;
    /** feature threshold */
    public double feaThr = 0.0D;
    /** feature score */
    public double score = 0.0D;
    /** is leaf node */
    public boolean isLeaf = false;
    /** left child */
    public Node leftChild = null;
    /** right child */
    public Node rightChild = null;
    /** missing then take the left child */
    public boolean missingThenTakeLeftChild = true;

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this, ToStringStyle.JSON_STYLE);
    }

}
