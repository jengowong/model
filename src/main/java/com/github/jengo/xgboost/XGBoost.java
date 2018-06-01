package com.github.jengo.xgboost;

import com.google.common.collect.Lists;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import java.io.File;
import java.util.List;

/**
 * {@link XGBoost}
 */
public class XGBoost {

    /** 偏差 */
    private double bias = 0.0D;
    /** 收缩率 */
    private double shrinkage = 1.0D;
    /** 回归树林 */
    private List<RegressionTree> trees = Lists.newArrayList();

    public void setBias(double bias) {
        this.bias = bias;
    }

    public void setShrinkage(double shrinkage) {
        this.shrinkage = shrinkage;
    }

    public static XGBoost buildXGBoost(String filePath) throws Exception {
        XGBoost xgboost = new XGBoost();
        String fileInLine = FileUtils.readFileToString(new File(filePath));
        String[] treeInLines = fileInLine.trim().split("booster");
        for (int i = 1; i < treeInLines.length; i++) {
            RegressionTree tree = new RegressionTree();
            tree.load(treeInLines[i]);
            xgboost.trees.add(tree);
        }
        return xgboost;
    }

    public static List<Tuple> buildTuples(String filePath) throws Exception {
        List<Tuple> tuples = Lists.newArrayList();
        String fileInLine = FileUtils.readFileToString(new File(filePath));
        String[] feaValInLines = fileInLine.trim().split("\n");
        for (String feaValInLine : feaValInLines) {
            String[] feaVals = feaValInLine.split(" ");
            tuples.add(buildTuple(feaVals));
        }
        return tuples;
    }

    private static Tuple buildTuple(String[] feaVals) {
        Tuple tuple = new Tuple();
        for (int i = 1; i < feaVals.length; i++) {
            String[] kv = feaVals[i].split(":");
            tuple.feaIdx2Val.put(Integer.parseInt(kv[0]), Double.parseDouble(kv[1]));
        }
        return tuple;
    }

    public double predict(Tuple tuple, boolean isTransform) {
        double score = this.bias;
        for (RegressionTree tree : trees) {
            score += (this.shrinkage * tree.predict(tuple));
        }
        if (isTransform) {
            score = this.sigmoid(score);
        }
        return score;
    }

    private double sigmoid(double x) {
        return 1.0D / (1.0D + Math.exp(-x));
    }

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this, ToStringStyle.JSON_STYLE);
    }

    @Deprecated
    public static void main(String[] args) throws Exception {
        XGBoost xgboost = buildXGBoost("/Users/jengowang/Applications/model/src/main/resources/xgboost.model");
        xgboost.setBias(0.5D);
        List<Tuple> tuples = buildTuples("/Users/jengowang/Applications/model/src/main/resources/mushroom.test");
        for (Tuple tuple : tuples) {
            System.out.printf("%.5f\n", xgboost.predict(tuple, false));
        }
    }

}
