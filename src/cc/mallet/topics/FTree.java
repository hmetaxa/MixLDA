/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cc.mallet.topics;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.io.Serializable;
import java.util.BitSet;

/**
 *
 * @author hmetaxa
 */
public class FTree {

// implements Serializable, Cloneable, Iterable<Double>
    /**
     * serializable value *
     */
   

    protected double[] tree;
    protected int size;
    //protected BitSet activeTopics ; 
    //protected int activeTopicsNum ; 

    
//     private static final long serialVersionUID = -680739021358875431L;
//     
//    /**
//     * stored hash *
//     */
//    private transient int hash;
//
//    /**
//     * stored string *
//     */
//    private transient String treeString;
//
//    /* default reading serialization */
//    private void readObject(ObjectInputStream inputStream) throws ClassNotFoundException, IOException {
//        inputStream.defaultReadObject();
//    }
//
//    /* default writing serialization */
//    private void writeObject(ObjectOutputStream outputStream) throws IOException {
//        outputStream.defaultWriteObject();
//    }

    public FTree() {
        size = 0;
        tree = null;
    }

    public FTree(int size) {
        init(size);
    }

    public void init(int size) {

        if (size <= 0 || size == Integer.MAX_VALUE) {
            throw new IllegalArgumentException();
        }

        this.size = size;
        tree = new double[2 * size];
        //activeTopics = new BitSet(size);

       // this.hash = 0;
        //this.treeString = null;
    }

    public FTree(double[] weights) {
        size = weights.length;
        init(size);
        constructTree(weights);
    }

//    public void recompute(double[] weights) {
//        constructTree(weights);
//    }
    public synchronized FTree clone() {
        try {
            FTree ret = (FTree) super.clone(); // new FTree(size);
            ret.tree = Arrays.copyOf(this.tree, this.tree.length);
            return ret;
        } catch (CloneNotSupportedException e) {
            // this shouldn't happen, since we are Cloneable
            throw new InternalError();
        }
    }

    public synchronized void constructTree(double[] weights) {
//        activeTopicsNum = 0;
//     for (int i = 0; i <size; i++) {
//         boolean activeTopic = weights[i]!=0;
//         activeTopics.set(i, activeTopic);
//          activeTopicsNum ++;    
//     }
//
//     activeTopicsNum += 2; //leave two spare topics
//     tree = new double[2 * activeTopicsNum];
// Reversely initialize elements
        Arrays.fill(tree, 0);
        for (int i = 2 * size - 1; i > 0; --i) {
            if (i >= size) {
                tree[i] = weights[i - size];
            } else {
                tree[i] = tree[2 * i] + tree[2 * i + 1];
            }
        }
    }

    public synchronized int sample(double u) {
        if (u > 1) {
            throw new IllegalArgumentException();
        }

        // due to multi threading / queue based updates, we should only pass the sample [0,1] from uniform
        u = u * tree[1];
        int i = 1;

        while (i < size) {
            if (u < tree[2 * i]) {
                i = 2 * i;
            } else {
                u = u - tree[2 * i];
                i = 2 * i + 1;
            }
        }

        return i - size;
    }

    public synchronized void update(int topic, double new_value) {
        // t = 0..T-1, 
        int i = topic + size;
        double delta = new_value - tree[i];
        while (i > 0) {
            tree[i] += delta;
            i = i / 2;
        }
    }

    public synchronized double getComponent(int t) {
        // t = 0..T-1
        return tree[t + size];
    }

    public static void main(String[] args) {

        try {

            double[] temp = {1, 2, 3, 4};
            FTree tree = new FTree(temp);

            int tmp = tree.sample(0.4);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
