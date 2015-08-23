/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cc.mallet.topics;

import java.util.Arrays;

/**
 *
 * @author hmetaxa
 */
public class FTree {

    protected double[] w;
    protected int T;

    public FTree() {
        T = 0;
        w = null;
    }

    public FTree(int num) {
        init(num);
    }

    public void init(int num) {
        T = num;
        w = new double[2 * T];
    }

    public FTree(double[] weights) {
        T = weights.length;
        init(T);
        constructTree(weights);
    }

//    public void recompute(double[] weights) {
//        constructTree(weights);
//    }
    public FTree clone() {
        FTree ret = new FTree(T);
        System.arraycopy(w, 0, ret.w, 0, T);
        return ret;
    }

    public void constructTree(double[] weights) {
        // Reversely initialize elements
        for (int i = 2 * T - 1; i > 0; --i) {
            if (i >= T) {
                w[i] = weights[i - T];
            } else {
                w[i] = w[2 * i] + w[2 * i + 1];
            }
        }
    }

    public int sample(double u) {
        int i = 1;
       // u = u * w[i];
        while (i < T) {
            if (u < w[2 * i]) {
                i = 2 * i;
            } else {
                u = u - w[2 * i];
                i = 2 * i + 1;
            }
        }

        return i - T;
    }

    public void update(int t, double new_w) {
        // t = 0..T-1, 
        int i = t + T;
        double delta = new_w - w[i];
        while (i > 0) {
            w[i] += delta;
            i = i / 2;
        }
    }

    public double getComponent(int t) {
        // t = 0..T-1
        return w[t + T];
    }

    public static void main(String[] args) {

        try {

            double[] temp = {1, 2, 3, 4};
            FTree tree = new FTree(temp);

            int tmp = tree.sample(3);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
