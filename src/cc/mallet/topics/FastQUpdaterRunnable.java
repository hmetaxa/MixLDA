/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cc.mallet.topics;

import static cc.mallet.topics.FastQParallelTopicModel.logger;
import cc.mallet.types.Dirichlet;
import cc.mallet.util.MalletLogger;
import cc.mallet.util.Randoms;
import static java.lang.Math.log;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CyclicBarrier;
import java.util.logging.Logger;
import org.knowceans.util.RandomSamplers;
import org.knowceans.util.Vectors;

/**
 *
 * @author Omiros
 */
public class FastQUpdaterRunnable implements Runnable {

    public static Logger logger = MalletLogger.getLogger(FastQUpdaterRunnable.class.getName());
    protected int[][] typeTopicCounts; // indexed by <feature index, topic index>
    protected int[] tokensPerTopic; // indexed by <topic index>
    protected FTree[] trees; //store 
    protected List<ConcurrentLinkedQueue<FastQDelta>> queues;
    protected double[] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
    protected double alphaSum;
    protected double beta;   // Prior on per-topic multinomial distribution over words
    protected double betaSum;

    protected double tablesCnt;
    protected double gamma = 1;
    protected double gammaRoot = 10;
    protected int numTopics;
    protected int numTypes;
    // The max over typeTotals, used for beta optimization
    protected int maxTypeCount;
    protected Randoms random;
    protected int[] docLengthCounts; // histogram of document sizes
    public int[][] topicDocCounts; // histogram of document/topic counts, indexed by <topic index, sequence position index>

    //protected FTree betaSmoothingTree;
    private final CyclicBarrier cyclicBarrier;
    boolean useCycleProposals = false;
    public static final double DEFAULT_BETA = 0.01;
    boolean optimizeParams = false;

    // Optimize gamma hyper params
    RandomSamplers samp;
    HashSet<Integer> inActiveTopicIndex = new HashSet<Integer>(); //inactive topic index for all modalities
    private NumberFormat formatter;

    public FastQUpdaterRunnable(
            int[][] typeTopicCounts,
            int[] tokensPerTopic,
            FTree[] trees,
            List<ConcurrentLinkedQueue<FastQDelta>> queues,
            double[] alpha, double alphaSum,
            double beta, boolean useCycleProposals,
            CyclicBarrier cyclicBarrier,
            int numTopics,
            int[] docLengthCounts,
            int[][] topicDocCounts,
            int numTypes,
            int maxTypeCount
    //        , FTree betaSmoothingTree
    ) {

        this.alphaSum = alphaSum;
        this.cyclicBarrier = cyclicBarrier;
        this.alpha = alpha;
        this.beta = beta;
        this.betaSum = betaSum;
        this.queues = queues;
        this.typeTopicCounts = typeTopicCounts;
        this.tokensPerTopic = tokensPerTopic;
        this.trees = trees;
        this.useCycleProposals = useCycleProposals;
        this.numTopics = numTopics;
        this.docLengthCounts = docLengthCounts;
        this.topicDocCounts = topicDocCounts;
        this.numTypes = numTypes;
        //this.betaSmoothingTree = betaSmoothingTree;
        //finishedSamplingTreads = new boolean

    }

    public boolean isFinished = true;

    public void setOptimizeParams(boolean optimizeParams) {
        this.optimizeParams = optimizeParams;
    }

    public void run() {

        Set<Integer> finishedSamplingTreads = new HashSet<Integer>();

        if (!isFinished) {
            System.out.println("already running!");
            return;
        }
        isFinished = false;
        if (optimizeParams) {
            optimizeGamma();
            updateAlphaAndSmoothing();
            optimizeBeta();
            recalcTrees();
        }

        
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

                        topicDocCounts[delta.OldTopic][tokensPerTopic[delta.OldTopic]]--;
                        tokensPerTopic[delta.OldTopic]--;
                        assert (tokensPerTopic[delta.OldTopic] >= 0) : "old Topic " + delta.OldTopic + " below 0";
                        if (tokensPerTopic[delta.OldTopic] > 0) {
                            topicDocCounts[delta.OldTopic][tokensPerTopic[delta.OldTopic]]++;
                        }

                        if (tokensPerTopic[delta.NewTopic] > 0) {
                            topicDocCounts[delta.NewTopic][tokensPerTopic[delta.NewTopic]]--;
                        }
                        tokensPerTopic[delta.NewTopic]++;
                        topicDocCounts[delta.NewTopic][tokensPerTopic[delta.NewTopic]]++;

                        //Update tree
                        if (useCycleProposals) {
                            trees[delta.Type].update(delta.OldTopic, ((currentTypeTopicCounts[delta.OldTopic] + beta) / (tokensPerTopic[delta.OldTopic] + betaSum)));
                            trees[delta.Type].update(delta.NewTopic, ((currentTypeTopicCounts[delta.NewTopic] + beta) / (tokensPerTopic[delta.NewTopic] + betaSum)));

                            //betaSmoothingTree.update(delta.OldTopic, (beta / (tokensPerTopic[delta.OldTopic] + betaSum)));
                            //betaSmoothingTree.update(delta.NewTopic, ( beta / (tokensPerTopic[delta.NewTopic] + betaSum)));
                        } else {
                            trees[delta.Type].update(delta.OldTopic, (gamma *alpha[delta.OldTopic] * (currentTypeTopicCounts[delta.OldTopic] + beta) / (tokensPerTopic[delta.OldTopic] + betaSum)));
                            trees[delta.Type].update(delta.NewTopic, (gamma *alpha[delta.NewTopic] * (currentTypeTopicCounts[delta.NewTopic] + beta) / (tokensPerTopic[delta.NewTopic] + betaSum)));
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

      public void optimizeBeta() {
        // The histogram starts at count 0, so if all of the
        //  tokens of the most frequent type were assigned to one topic,
        //  we would need to store a maxTypeCount + 1 count.
        int[] countHistogram = new int[maxTypeCount + 1];

        //  Now count the number of type/topic pairs that have
        //  each number of tokens.
        for (int type = 0; type < numTypes; type++) {

            int[] counts = typeTopicCounts[type];

            for (int topic = 0; topic < numTopics; topic++) {
                int count = counts[topic];
                if (count > 0) {
                    countHistogram[count]++;
                }
            }
        }

        // Figure out how large we need to make the "observation lengths"
        //  histogram.
        int maxTopicSize = 0;
        for (int topic = 0; topic < numTopics; topic++) {
            if (tokensPerTopic[topic] > maxTopicSize) {
                maxTopicSize = tokensPerTopic[topic];
            }
        }

        // Now allocate it and populate it.
        int[] topicSizeHistogram = new int[maxTopicSize + 1];
        for (int topic = 0; topic < numTopics; topic++) {
            topicSizeHistogram[tokensPerTopic[topic]]++;
        }

        betaSum = Dirichlet.learnSymmetricConcentration(countHistogram,
                topicSizeHistogram,
                numTypes,
                betaSum);
        beta = betaSum / numTypes;

        //TODO: copy/update trees in threads
        logger.info("[beta: " + formatter.format(beta) + "] ");
        // Now publish the new value
       // for (int thread = 0; thread < numThreads; thread++) {
       //     runnables[thread].resetBeta(beta, betaSum);
       // }

    }
      
    private void optimizeGamma() {

        // hyperparameters for DP and Dirichlet samplers
        // Teh+06: Docs: (1, 1), M1-3: (0.1, 0.1); HMM: (1, 1)
        double aalpha = 5;
        double balpha = 0.1;
        //double abeta = 0.1;
        //double bbeta = 0.1;
        // Teh+06: Docs: (1, 0.1), M1-3: (5, 0.1), HMM: (1, 1)
        double agamma = 5;
        double bgamma = 0.1;
        // number of samples for parameter samplers
        int R = 10;

//        //int[][] docLengthCounts = new int[numModalities][histogramSize]; // histogram of document sizes taking into consideration (summing up) all modalities
//        //int[][][] topicDocCounts = new int[numModalities][numTopics][histogramSize]; // histogram of document/topic counts, indexed by <topic index, sequence position index> considering all modalities
//        double[] tablesPerModality = new double[numModalities];
//        Arrays.fill(tablesPerModality, 0);
//        double totalTables = 0;
//
//        for (Byte mod = 0; mod < numModalities; mod++) {
//            for (int thread = 0; thread < numThreads; thread++) {
//                tablesPerModality[mod] += Math.ceil(runnables[thread].getTablesPerModality()[mod] / (double) numThreads);
//            }
//            totalTables += tablesPerModality[mod];
//        }
        for (int r = 0; r < R; r++) {
            // gamma: root level (Escobar+West95) with n = T
            // (14)
            double eta = samp.randBeta(gammaRoot + 1, tablesCnt);
            double bloge = bgamma - log(eta);
            // (13')
            double pie = 1. / (1. + (tablesCnt * bloge / (agamma + numTopics - 1)));
            // (13)
            int u = samp.randBernoulli(pie);
            gammaRoot = samp.randGamma(agamma + numTopics - 1 + u, 1. / bloge);

            // for (byte m = 0; m < numModalities; m++) {
            // alpha: document level (Teh+06)
            double qs = 0;
            double qw = 0;
            for (int j = 0; j < docLengthCounts.length; j++) {
                for (int i = 0; i < docLengthCounts[j]; i++) {
                    // (49) (corrected)
                    qs += samp.randBernoulli(j / (j + gamma));
                    // (48)
                    qw += log(samp.randBeta(gamma + 1, j));
                }
            }
            // (47)
            gamma = samp.randGamma(aalpha + tablesCnt - qs, 1. / (balpha - qw));

            //  }
        }
        logger.info("GammaRoot: " + gammaRoot);
        //for (byte m = 0; m < numModalities; m++) {
        logger.info("Gamma: " + gamma);
        //}

    }

    private void updateAlphaAndSmoothing() {
        double[] mk = new double[numTopics + 1];

        for (int t = 0; t < numTopics; t++) {
            inActiveTopicIndex.add(t); //inActive by default and activate if found 
        }

        // for (byte m = 0; m < numModalities; m++) {
        for (int t = 0; t < numTopics; t++) {

            //int k = kactive.get(kk);
            for (int i = 0; i < topicDocCounts[t].length; i++) {
                //for (int j = 0; j < numDocuments; j++) {

                if (topicDocCounts[t][i] > 0 && i > 1) {
                    inActiveTopicIndex.remove(t);
                    //sample number of tables
                    // number of tables a CRP(alpha tau) produces for nmk items
                    //TODO: See if  using the "minimal path" assumption  to reduce bookkeeping gives the same results. 
                    //Huge Memory consumption due to  topicDocCounts (* NumThreads), and striling number of first kind allss double[][] 
                    //Also 2x slower than the parametric version due to UpdateAlphaAndSmoothing

                    int curTbls = 0;
                    try {
                        curTbls = random.nextAntoniak(gamma * alpha[t], i);

                    } catch (Exception e) {
                        curTbls = 1;
                    }

                    mk[t] += (topicDocCounts[t][i] * curTbls);
                    //mk[m][t] += 1;//direct minimal path assignment Samplers.randAntoniak(gamma[m] * alpha[m].get(t),  tokensPerTopic[m].get(t));
                    // nmk[m].get(k));
                } else if (topicDocCounts[t][i] > 0 && i == 1) //nmk[m].get(k) = 0 or 1
                {
                    inActiveTopicIndex.remove(t);
                    mk[t] += topicDocCounts[t][i];
                }
            }
        }
        // }// end outter for loop

        //for (byte m = 0; m < numModalities; m++) {
        //alpha[m].fill(0, numTopics, 0);
        alphaSum = 0;
        mk[numTopics] = gammaRoot;
        tablesCnt = Vectors.sum(mk);

        double[] tt = sampleDirichlet(mk);

        for (int kk = 0; kk <= numTopics; kk++) {
            //int k = kactive.get(kk);
            alpha[kk] = tt[kk];
            alphaSum += gamma * tt[kk];
            //tau.set(k, tt[kk]);
        }

//            if (alpha[m].size() < numTopics + 1) {
//                alpha[m].add(tt[numTopics]);
//            } else {
//                alpha[m].set(numTopics, tt[numTopics]);
//            }
        //tau.set(K, tt[K]);
        //}
      //Recalc trees
    }

    private double[] sampleDirichlet(double[] p) {
        double magnitude = 1;
        double[] partition;

        magnitude = 0;
        partition = new double[p.length];

        // Add up the total
        for (int i = 0; i < p.length; i++) {
            magnitude += p[i];
        }

        for (int i = 0; i < p.length; i++) {
            partition[i] = p[i] / magnitude;
        }

        double distribution[] = new double[partition.length];

//		For each dimension, draw a sample from Gamma(mp_i, 1)
        double sum = 0;
        for (int i = 0; i < distribution.length; i++) {

            distribution[i] = random.nextGamma(partition[i] * magnitude, 1);
            if (distribution[i] <= 0) {
                distribution[i] = 0.0001;
            }
            sum += distribution[i];
        }

//		Normalize
        for (int i = 0; i < distribution.length; i++) {
            distribution[i] /= sum;
        }

        return distribution;
    }

    private void recalcTrees() {
        //recalc trees
        double[] temp = new double[numTopics];
        for (int w = 0; w < numTypes; ++w) {

            int[] currentTypeTopicCounts = typeTopicCounts[w];
            for (int currentTopic = 0; currentTopic < numTopics; currentTopic++) {

                // temp[currentTopic] = (currentTypeTopicCounts[currentTopic] + beta)  / (tokensPerTopic[currentTopic] + betaSum);
                if (useCycleProposals) {
                    temp[currentTopic] = (currentTypeTopicCounts[currentTopic] + beta) / (tokensPerTopic[currentTopic] + betaSum); //with cycle proposal
                } else {
                    temp[currentTopic] = gamma *alpha[currentTopic] * (currentTypeTopicCounts[currentTopic] + beta) / (tokensPerTopic[currentTopic] + betaSum);
                }

            }

            //trees[w] = new FTree(temp);
            trees[w].constructTree(temp);

            //reset temp
            Arrays.fill(temp, 0);

        }

//         if (useCycleProposals) {
//            //Arrays.fill(temp, 0);
//
//            for (int currentTopic = 0; currentTopic < numTopics; currentTopic++) {
//                temp[currentTopic] = (beta) / (tokensPerTopic[currentTopic] + betaSum); //with cycle proposal
//            }
//            
////            betaSmoothingTree.constructTree(temp);
//        }
    }
}
