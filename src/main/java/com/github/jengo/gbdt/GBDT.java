package com.github.jengo.gbdt;

import com.google.common.collect.Lists;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * {@link GBDT}
 */
public class GBDT {

    /** 偏差 */
    private double bias;
    /** 收缩率 */
    private double shrinkage;
    /** 回归树林 */
    private List<RegressionTree> trees;

    public void loadFromFile(String filePath) throws Exception {
        String s = FileUtils.readFileToString(new File(filePath));
        load(s);
    }

    public void loadFromStream(InputStream inputStream) throws Exception {
        String s = IOUtils.toString(inputStream);
        load(s);
    }

    public void load(String fileInLine) throws Exception {
        String[] treeInLines = fileInLine.split("\n;\n");

        this.shrinkage = Double.parseDouble(treeInLines[0]);
        this.bias = Double.parseDouble(treeInLines[1]);
        this.trees = Lists.newArrayList();
        for (int i = 2; i < treeInLines.length; i++) {
            String treeInLine = treeInLines[i];
            RegressionTree tree = new RegressionTree();
            tree.load(treeInLine);
            this.trees.add(tree);
        }
    }


    public double predict(Tuple t) {
        double s = bias;
        for (RegressionTree tree : trees) {
            s += (shrinkage * tree.predict(t));
        }
        return s;
    }

    public double predict(Tuple t, List<Double> w) {
        double s = bias;
        for (RegressionTree tree : trees) {
            s += (shrinkage * tree.predict(t, w));
        }
        return s;
    }

    public List<Integer> calcOneHotFeature(Tuple t) {
        ArrayList<Integer> oneHotList = new ArrayList<Integer>();
        int acc = 0;
        for (RegressionTree tree : trees) {
            int leafIndex = tree.locateLeaf(t);
            if (leafIndex >= 0) {
                oneHotList.add(acc + leafIndex);
            }
            acc += tree.getLeafSize();
        }
        return oneHotList;
    }

    @Deprecated
    public static void main(String[] args) throws Exception {
        GBDT gbdt = new GBDT();
        gbdt.loadFromFile("/Users/jengowang/Applications/model/src/main/resources/gbdt.model");
    }

}
