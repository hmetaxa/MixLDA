/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cc.mallet.topics;

import cc.mallet.util.Randoms;
import java.util.ArrayList;

/**
 *
 * @author Omiros
 */
public class FastQUpdaterRunnable implements Runnable {

     public FastQUpdaterRunnable(int numTopics,
            double[] alpha, double alphaSum,
            double beta, Randoms random,
            ArrayList<TopicAssignment> data,
            int[][] typeTopicCounts,
            int[] tokensPerTopic,
            int startDoc, int numDocs, FTree[] trees) {
         
     }
     
    public void run() {

    }
}
