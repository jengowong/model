package com.github.jengo.gbdt;

import com.google.common.collect.Lists;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.List;

/**
 * {@link GBDT}
 */
public class GBDT {

    /** 偏差 */
    private double bias = 0.0D;
    /** 收缩率 */
    private double shrinkage = 1.0D;
    /** 回归树林 */
    private List<RegressionTree> trees = Lists.newArrayList();

    public void loadFromFile(String filePath) throws Exception {
        String s = FileUtils.readFileToString(new File(filePath));
        load(s);
    }

    private void load(String fileInLine) throws Exception {
        String[] treeInLines = fileInLine.split("\n;\n");

        this.shrinkage = Double.parseDouble(treeInLines[0]);
        this.bias = Double.parseDouble(treeInLines[1]);
        for (int i = 2; i < treeInLines.length; i++) {
            String treeInLine = treeInLines[i];
            RegressionTree tree = new RegressionTree();
            tree.load(treeInLine);
            this.trees.add(tree);
        }
    }

    public double predict(Tuple tuple) {
        double s = bias;
        for (RegressionTree tree : trees) {
            s += (shrinkage * tree.predict(tuple));
        }
        return s;
    }

    public double predict(Tuple tuple, List<Double> w) {
        double s = bias;
        for (RegressionTree tree : trees) {
            s += (shrinkage * tree.predict(tuple, w));
        }
        return s;
    }

    public List<Integer> calcOneHotFeature(Tuple t) {
        List<Integer> oneHotList = Lists.newArrayList();
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

        String feaVals = "1:30 2:1755.790100724157 3:796.1674598120957 4:40 5:146 6:260.0 7:321.0 8:78.0 9:101.0 10:226 11:0.0 12:0.0 13:2.0 14:4.0 15:0 16:1933 17:288916.0 18:0 19:0.034286453616250476 20:0.0 21:0.0 22:0.0 23:0.03714447877908809 24:0.019911822351273995 25:7.396142627798688E-4 26:0.019911822351273995 27:609 28:530 29:42 30:0.0 31:0.0 32:0.0 33:-1 34:-1 35:0.7407407407407407 36:80.0 37:0.1445146136297039 38:0.03367289190718732 39:1.0 40:0.004866224963761959 41:0.14524677007057665 42:0.0 43:0.0 44:0.0 45:106 46:66.0 47:0.6111111111111112 48:0.07446808510638298 49:323.0 50:332.0 51:268.0 52:206.0 53:150.0 54:83.0 55:65.0 56:0.24253731343283583 57:0.5597014925373134 58:0.8297213622291022 59:10.0 60:10.0 61:7.0 62:0.0 63:0.0 64:2.0 65:1.0 66:0.14285714285714285 67:0.0 68:0.7 69:0 70:0 71:0 72:0 73:0 74:0 75:0";
        Tuple t = new Tuple();
        for (String feaVal : feaVals.split(" ")) {
            t.feaVals.add(Double.parseDouble(feaVal.split(":")[1]));
        }
        System.out.println(gbdt.predict(t));
    }

}
