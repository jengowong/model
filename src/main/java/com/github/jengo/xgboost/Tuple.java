package com.github.jengo.xgboost;

import com.google.common.collect.Maps;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;

import java.util.Map;

/**
 * {@link Tuple}
 */
public class Tuple {

    /** { featureIdx, featureValue } */
    public Map<Integer, Double> feaIdx2Val = Maps.newHashMap();

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this, ToStringStyle.JSON_STYLE);
    }

}
