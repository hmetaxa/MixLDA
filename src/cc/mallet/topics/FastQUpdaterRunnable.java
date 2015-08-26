/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cc.mallet.topics;

import cc.mallet.util.Randoms;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CyclicBarrier;

/**
 *
 * @author Omiros
 */
public class FastQUpdaterRunnable implements Runnable {

    protected int[][] typeTopicCounts; // indexed by <feature index, topic index>
    protected int[] tokensPerTopic; // indexed by <topic index>
    protected FTree[] trees; //store 
    protected List<ConcurrentLinkedQueue<FastQDelta>> queues;
    protected double[] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
    protected double alphaSum;
    protected double beta;   // Prior on per-topic multinomial distribution over words
    protected double betaSum;
    private final CyclicBarrier cyclicBarrier;
    boolean useCycleProposals = false;
    public static final double DEFAULT_BETA = 0.01;

    public FastQUpdaterRunnable(
            int[][] typeTopicCounts,
            int[] tokensPerTopic,
            FTree[] trees,
            List<ConcurrentLinkedQueue<FastQDelta>> queues,
            double[] alpha, double alphaSum,
            double beta, boolean useCycleProposals,
            CyclicBarrier cyclicBarrier) {

        this.alphaSum = alphaSum;
        this.cyclicBarrier = cyclicBarrier;
        this.alpha = alpha;
        this.beta = beta;
        this.betaSum = beta * typeTopicCounts.length;
        this.queues = queues;
        this.typeTopicCounts = typeTopicCounts;
        this.tokensPerTopic = tokensPerTopic;
        this.trees = trees;
        this.useCycleProposals = useCycleProposals;
        //finishedSamplingTreads = new boolean

    }

    public boolean isFinished = true;

    public void run() {

        Set<Integer> finishedSamplingTreads = new HashSet<Integer>();

        if (!isFinished) {
            System.out.println("already running!");
            return;
        }
        isFinished = false;
        try {
            while (!isFinished) {

                FastQDelta delta;
                int[] currentTypeTopicCounts;
                for (int x = 0; x < queues.size(); x++) {
                    while ((delta = queues.get(x).poll()) != null) {

                        if (delta.Modality == -1 && delta.NewTopic == -1 && delta.OldTopic == -1 && delta.Type == -1) { // thread x has finished
                            finishedSamplingTreads.add(x);
                            isFinished = finishedSamplingTreads.size() == queues.size();
                            continue;
                        }
                        currentTypeTopicCounts = typeTopicCounts[delta.Type];

                        // Decrement the global topic count totals
                        currentTypeTopicCounts[delta.OldTopic]--;
                        currentTypeTopicCounts[delta.NewTopic]++;
                        
                        tokensPerTopic[delta.OldTopic]--;
                        assert (tokensPerTopic[delta.OldTopic] >= 0) : "old Topic " + delta.OldTopic + " below 0";
                        tokensPerTopic[delta.NewTopic]++;
                        
                        //Update tree
                        if (useCycleProposals) {
                            trees[delta.Type].update(delta.OldTopic, ((currentTypeTopicCounts[delta.OldTopic] + beta) / (tokensPerTopic[delta.OldTopic] + betaSum)));
                            trees[delta.Type].update(delta.NewTopic, ((currentTypeTopicCounts[delta.NewTopic] + beta) / (tokensPerTopic[delta.NewTopic] + betaSum)));
                        } else {
                            trees[delta.Type].update(delta.OldTopic, (alpha[delta.OldTopic] * (currentTypeTopicCounts[delta.OldTopic] + beta) / (tokensPerTopic[delta.OldTopic] + betaSum)));
                            trees[delta.Type].update(delta.NewTopic, (alpha[delta.NewTopic] * (currentTypeTopicCounts[delta.NewTopic] + beta) / (tokensPerTopic[delta.NewTopic] + betaSum)));
                        }
                        

                    }

                }

                try {
                    Thread.currentThread().sleep(20);
                } catch (Exception ex) {
                    ex.printStackTrace();
                }

            }

            try {
                cyclicBarrier.await();
            } catch (InterruptedException e) {
                System.out.println("Main Thread interrupted!");
                e.printStackTrace();
            } catch (BrokenBarrierException e) {
                System.out.println("Main Thread interrupted!");
                e.printStackTrace();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
