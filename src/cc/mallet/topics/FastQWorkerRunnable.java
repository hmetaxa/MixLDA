/* Copyright (C) 2005 Univ. of Massachusetts Amherst, Computer Science Dept.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.	For further
 information, see the file `LICENSE' included with this distribution. */
package cc.mallet.topics;

import java.util.Arrays;
import java.util.ArrayList;

import java.util.zip.*;

import java.io.*;
import java.text.NumberFormat;

import cc.mallet.types.*;
import cc.mallet.util.Randoms;
import java.util.Queue;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ThreadLocalRandom;

/**
 * A parallel topic model runnable task using FTrees / cycling proposals.
 *
 * @author Omiros Metaxas
 */
public class FastQWorkerRunnable implements Runnable {

    //boolean isFinished = true;
    ArrayList<TopicAssignment> data;
    int startDoc, numDocs;
    protected int numTopics; // Number of topics to be fit
    // These values are used to encode type/topic counts as
    //  count/topic pairs in a single int.
    protected int topicMask;
    protected int topicBits;
    protected int numTypes;
    protected double[] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
    protected double[] alphaSum;
    protected double[] beta;   // Prior on per-topic multinomial distribution over words
    protected double[] betaSum;
    protected double[] gamma;
    public static final double DEFAULT_BETA = 0.01;
    protected double docSmoothingOnlyMass = 0.0;
    protected double[] docSmoothingOnlyCumValues;

    protected int[][] typeTopicCounts; // indexed by <feature index, topic index>
    protected int[] tokensPerTopic; // indexed by <topic index>
    // for dirichlet estimation
    //protected int[] docLengthCounts; // histogram of document sizes
    //protected int[][] topicDocCounts; // histogram of document/topic counts, indexed by <topic index, sequence position index>
    boolean shouldSaveState = false;
    protected FTree[] trees; //store 
    protected Randoms random;
    protected int threadId = -1;
    protected Queue<FastQDelta> queue;
    private final CyclicBarrier cyclicBarrier;
    protected int MHsteps = 1;
    boolean useCycleProposals = false;

    public FastQWorkerRunnable(int numTopics,
            double[] alpha, double[] alphaSum,
            double[] beta,
            double[] betaSum,
            double[] gamma,
            Randoms random,
            ArrayList<TopicAssignment> data,
            int[][] typeTopicCounts,
            int[] tokensPerTopic,
            int startDoc, int numDocs, FTree[] trees, boolean useCycleProposals,
            int threadId,
//            ConcurrentLinkedQueue<FastQDelta> queue, 
            CyclicBarrier cyclicBarrier
    //, FTree betaSmoothingTree
    ) {

        this.data = data;
        this.threadId = threadId;
        //this.queue = queue;
        this.cyclicBarrier = cyclicBarrier;

        this.numTopics = numTopics;
        this.numTypes = typeTopicCounts.length;

        //trees = new FTree[this.numTypes];
        if (Integer.bitCount(numTopics) == 1) {
            // exact power of 2
            topicMask = numTopics - 1;
            topicBits = Integer.bitCount(topicMask);
        } else {
            // otherwise add an extra bit
            topicMask = Integer.highestOneBit(numTopics) * 2 - 1;
            topicBits = Integer.bitCount(topicMask);
        }

        this.typeTopicCounts = typeTopicCounts;
        this.tokensPerTopic = tokensPerTopic;
        this.trees = trees;

        this.alphaSum = alphaSum;
        this.alpha = alpha;
        this.beta = beta;
        this.betaSum = betaSum;
        this.random = random;
        this.gamma = gamma;

        this.startDoc = startDoc;
        this.numDocs = numDocs;
        this.useCycleProposals = useCycleProposals;

        docSmoothingOnlyCumValues = new double[numTopics];

        //System.err.println("WorkerRunnable Thread: " + numTopics + " topics, " + topicBits + " topic bits, " + 
        //				   Integer.toBinaryString(topicMask) + " topic mask");
    }

     public void setQueue( Queue<FastQDelta> queue) {
        this.queue = queue;
    }
     
    public int[] getTokensPerTopic() {
        return tokensPerTopic;
    }

    public int[][] getTypeTopicCounts() {
        return typeTopicCounts;
    }

//    public int[] getDocLengthCounts() {
//        return docLengthCounts;
//    }
//    public int[][] getTopicDocCounts() {
//        return topicDocCounts;
//    }
//    public void initializeAlphaStatistics(int size) {
////        docLengthCounts = new int[size];
//     //   topicDocCounts = new int[numTopics][size];
//    }
    public void collectAlphaStatistics() {
        shouldSaveState = true;
    }

    public void resetBeta(double[] beta, double[] betaSum) {
        this.beta = beta;
        this.betaSum = betaSum;
    }

    // p(w|t=z, all) = (alpha[topic] + topicPerDocCounts[d])       *   ( (typeTopicCounts[w][t]/(tokensPerTopic[topic] + betaSum)) + beta/(tokensPerTopic[topic] + betaSum)  )
    // masses:         alphasum     + select a random topics from doc       FTree for active only topics (leave 2-3 spare)                     common FTree f
    //              (binary search)                                               get index from typeTopicsCount
    public void run() {

        try {

            // Initialize the doc smoothing-only sampling bucket (Sum(a[i])
            docSmoothingOnlyMass = 0;
            if (useCycleProposals) {
                // cachedCoefficients cumulative array that will be used for binary search
                for (int topic = 0; topic < numTopics; topic++) {
                    docSmoothingOnlyMass += gamma[0] * alpha[topic];
                    docSmoothingOnlyCumValues[topic] = docSmoothingOnlyMass;
                }
            }
            for (int doc = startDoc;
                    doc < data.size() && doc < startDoc + numDocs;
                    doc++) {

//				  if (doc % 10 == 0) {
//				  System.out.println("processing doc " + doc);
//				  }
//				
                FeatureSequence tokenSequence
                        = (FeatureSequence) data.get(doc).instance.getData();
                LabelSequence topicSequence
                        = (LabelSequence) data.get(doc).topicSequence;

                if (useCycleProposals) {
                    sampleTopicsForOneDocCyclingProposals(tokenSequence, topicSequence,
                            true);
                } else {
                    sampleTopicsForOneDoc(tokenSequence, topicSequence,
                            true);
                }

            }

            shouldSaveState = false;
            //isFinished = true;

            queue.add(new FastQDelta(-1, -1, -1, -1, -1, -1));

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

    public static int lower_bound(double[] arr, double key, int len) {
        //int len = arr.length;
        int lo = 0;
        int hi = len - 1;
        int mid = (lo + hi) / 2;
        while (true) {
            //int cmp = arr[mid].compareTo(key);
            if (arr[mid] >= key) {
                hi = mid - 1;
                if (hi < lo) {
                    return mid;
                }
            } else {
                lo = mid + 1;
                if (hi < lo) {
                    return mid < len - 1 ? mid + 1 : -1;
                }
            }
            mid = (lo + hi) / 2; //(hi-lo)/2+lo in order not to overflow?  or  (lo + hi) >>> 1
        }
    }

    public static int lower_bound(Comparable[] arr, Comparable key) {
        int len = arr.length;
        int lo = 0;
        int hi = len - 1;
        int mid = (lo + hi) / 2;
        while (true) {
            int cmp = arr[mid].compareTo(key);
            if (cmp == 0 || cmp > 0) {
                hi = mid - 1;
                if (hi < lo) {
                    return mid;
                }
            } else {
                lo = mid + 1;
                if (hi < lo) {
                    return mid < len - 1 ? mid + 1 : -1;
                }
            }
            mid = (lo + hi) / 2; //(hi-lo)/2+lo in order not to overflow?  or  (lo + hi) >>> 1
        }
    }

    protected void sampleTopicsForOneDoc(FeatureSequence tokenSequence,
            FeatureSequence topicSequence,
            boolean readjustTopicsAndStats /* currently ignored */) {

        try {
            int[] oneDocTopics = topicSequence.getFeatures();

            int[] currentTypeTopicCounts;
            int[] localTopicIndex = new int[numTopics];
            double[] topicDocWordMasses = new double[numTopics];
            int type, oldTopic, newTopic;
            FTree currentTree;

            int docLength = tokenSequence.getLength();

            int[] localTopicCounts = new int[numTopics];

            //		populate topic counts
            for (int position = 0; position < docLength; position++) {
                if (oneDocTopics[position] == FastQParallelTopicModel.UNASSIGNED_TOPIC) {
                    continue;
                }
                localTopicCounts[oneDocTopics[position]]++;
            }

            // Build an array that densely lists the topics that
            //  have non-zero counts.
            int denseIndex = 0;
            for (int topic = 0; topic < numTopics; topic++) {
                if (localTopicCounts[topic] != 0) {
                    localTopicIndex[denseIndex] = topic;
                    denseIndex++;
                }
            }

            // Record the total number of non-zero topics
            int nonZeroTopics = denseIndex;

            //	Iterate over the positions (words) in the document 
            for (int position = 0; position < docLength; position++) {
                type = tokenSequence.getIndexAtPosition(position);
                oldTopic = oneDocTopics[position];

                currentTypeTopicCounts = typeTopicCounts[type];
                currentTree = trees[type];

                if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {

                    // Decrement the local doc/topic counts
                    localTopicCounts[oldTopic]--;

                    // Maintain the dense index, if we are deleting
                    //  the old topic
                    if (localTopicCounts[oldTopic] == 0) {
                        // First get to the dense location associated with  the old topic.
                        denseIndex = 0;
                        // We know it's in there somewhere, so we don't  need bounds checking.
                        while (localTopicIndex[denseIndex] != oldTopic) {
                            denseIndex++;
                        }
                        // shift all remaining dense indices to the left.
                        while (denseIndex < nonZeroTopics) {
                            if (denseIndex < localTopicIndex.length - 1) {
                                localTopicIndex[denseIndex]
                                        = localTopicIndex[denseIndex + 1];
                            }
                            denseIndex++;
                        }
                        nonZeroTopics--;
                    }

                    // Decrement the global type topic counts  at the end (through delta / queue)
                }

                //		compute word / doc mass for binary search
                double topicDocWordMass = 0.0;

                for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
                    int topic = localTopicIndex[denseIndex];
                    int n = localTopicCounts[topic];
                    topicDocWordMass += n * (currentTypeTopicCounts[topic] + beta[0]) / (tokensPerTopic[topic] + betaSum[0]);
                    //topicDocWordMass +=  n * trees[type].getComponent(topic);
                    topicDocWordMasses[denseIndex] = topicDocWordMass;

                }

                double nextUniform = ThreadLocalRandom.current().nextDouble();
                double sample = nextUniform * (topicDocWordMass + currentTree.tree[1]);
                newTopic = -1;

                //double sample = ThreadLocalRandom.current().nextDouble() * (topicDocWordMass + trees[type].tree[1]);
                newTopic = sample < topicDocWordMass ? localTopicIndex[lower_bound(topicDocWordMasses, sample, nonZeroTopics)] : currentTree.sample(nextUniform);
//            if (sample < topicDocWordMass) {
//
//                //int tmp = lower_bound(topicDocWordMasses, sample, nonZeroTopics);
//                newTopic = localTopicIndex[lower_bound(topicDocWordMasses, sample, nonZeroTopics)]; //actual topic
//
//            } else {
//
//                newTopic = currentTree.sample(nextUniform);
//            }

                if (newTopic == -1) {
                    System.err.println("WorkerRunnable sampling error on word topic mass: " + sample + " " + trees[type].tree[1]);
                    newTopic = numTopics - 1; // TODO is this appropriate
                    //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
                }

                //assert(newTopic != -1);
                //			Put that new topic into the counts
                oneDocTopics[position] = newTopic;

                //increment local counts
                localTopicCounts[newTopic]++;

                // If this is a new topic for this document, add the topic to the dense index.
                if (localTopicCounts[newTopic] == 1) {
                    // First find the point where we  should insert the new topic by going to
                    //  the end  and working backwards
                    denseIndex = nonZeroTopics;
                    while (denseIndex > 0
                            && localTopicIndex[denseIndex - 1] > newTopic) {
                        localTopicIndex[denseIndex]
                                = localTopicIndex[denseIndex - 1];
                        denseIndex--;
                    }
                    localTopicIndex[denseIndex] = newTopic;
                    nonZeroTopics++;
                }

                //add delta to the queue
                if (newTopic != oldTopic) {
                    //queue.add(new FastQDelta(oldTopic, newTopic, type, 0, 1, 1));
                    queue.add(new FastQDelta(oldTopic, newTopic, type, 0, localTopicCounts[oldTopic], localTopicCounts[newTopic]));
                }

            }

        } catch (Exception e) {
            e.printStackTrace();
        }

//        if (shouldSaveState) {
//            // Update the document-topic count histogram,
//            //  for dirichlet estimation
//            //[docLength]++;
//
//            for (int topic = 0; topic < numTopics; topic++) {
//                topicDocCounts[topic][localTopicCounts[topic]]++;
//            }
//        }
    }

    protected void sampleTopicsForOneDocCyclingProposals(FeatureSequence tokenSequence,
            FeatureSequence topicSequence,
            boolean readjustTopicsAndStats /* currently ignored */) {

        int[] oneDocTopics = topicSequence.getFeatures();

        int[] currentTypeTopicCounts;
        FTree currentTree;
        int type, oldTopic, newTopic, currentTopic;

        int docLength = tokenSequence.getLength();

        int[] localTopicCounts = new int[numTopics];

        //		populate topic counts
        for (int position = 0; position < docLength; position++) {
            if (oneDocTopics[position] == ParallelTopicModel.UNASSIGNED_TOPIC) {
                continue;
            }
            localTopicCounts[oneDocTopics[position]]++;
        }

        //	Iterate over the positions (words) in the document 
        for (int position = 0; position < docLength; position++) {
            type = tokenSequence.getIndexAtPosition(position);
            oldTopic = oneDocTopics[position];

            currentTypeTopicCounts = typeTopicCounts[type];
            currentTree = trees[type];

            if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {

                // Decrement the local doc/topic counts
                localTopicCounts[oldTopic]--;

                // Multi core (queue based) approximation: All global counts will be updated at the end 
            }

            currentTopic = oldTopic;

            for (int MHstep = 0; MHstep < MHsteps; MHstep++) {
                //in every Metropolis hasting step we are taking two samples cycling related doc & word proposals and we eventually keep the best one
                // we can sample from doc topic mass immediately after word topic mass, as it doesn't get affected by current topic selection!! 
                //--> thus we get & check both samples from cycle proposal in every step

                //  if (!useDocProposal) {
                //Sample Word topic mass
                double sample = ThreadLocalRandom.current().nextDouble();//* (trees[type].tree[1]);

                //	Make sure it actually gets set
                newTopic = -1;

                newTopic = currentTree.sample(sample);

                if (newTopic == -1) {
                    System.err.println("WorkerRunnable sampling error on word topic mass: " + sample + " " + trees[type].tree[1]);
                    newTopic = numTopics - 1; // TODO is this appropriate
                    //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
                }

                if (currentTopic != newTopic) {
                    // due to queue based multi core approximation we should decrement global counts whenever appropriate
                    //both decr/incr of global arrays and trees is happening at the end... So at this point they contain the current (not decreased values)
                    // model_old & model_new should be based on decreased values, whereas probabilities (prop_old & prop_new) on the current ones
                    // BUT global cnts are changing due to QUeue based updates so there is no meaning in them
                    double model_old = (localTopicCounts[currentTopic] + gamma[0] * alpha[currentTopic]) * (currentTypeTopicCounts[currentTopic] + beta[0]) / (tokensPerTopic[currentTopic] + betaSum[0]);
                    double model_new = (localTopicCounts[newTopic] + gamma[0] * alpha[newTopic]) * (currentTypeTopicCounts[newTopic] + beta[0]) / (tokensPerTopic[newTopic] + betaSum[0]);
                    double prop_old = (currentTypeTopicCounts[currentTopic] + beta[0]) / (tokensPerTopic[currentTopic] + betaSum[0]);
                    double prop_new = (currentTypeTopicCounts[newTopic] + beta[0]) / (tokensPerTopic[newTopic] + betaSum[0]);
                    double acceptance = (model_new * prop_old) / (model_old * prop_new);

//                    double prop_old2 = (oldTopic == currentTopic)
//                            ? (localTopicCounts[currentTopic] + 1 + alpha[currentTopic])
//                            : (localTopicCounts[currentTopic] + alpha[currentTopic]);
//                    double prop_new2 = (newTopic == oldTopic)
//                            ? (localTopicCounts[newTopic] + 1 + alpha[newTopic])
//                            : (localTopicCounts[newTopic] + alpha[newTopic]);
//                    acceptance = (temp_new * prop_old * prop_new2) / (temp_old * prop_new * prop_old2);
//                    
                    //3. Compare against uniform[0,1]
                    currentTopic = acceptance >= 1 ? newTopic : ThreadLocalRandom.current().nextDouble() < acceptance ? newTopic : currentTopic;
//                    if (acceptance >= 1 || random.nextUniform() < acceptance) {
//                        currentTopic = newTopic;
//                    }
                }

                // Sample Doc topic mass 
                sample = ThreadLocalRandom.current().nextDouble() * (docSmoothingOnlyMass + docLength - 1);
                double origSample = sample;

                //	Make sure it actually gets set
                newTopic = -1;

                if (sample < docSmoothingOnlyMass) {

                    newTopic = lower_bound(docSmoothingOnlyCumValues, sample, numTopics);

                } else { //just select one random topic from DocTopics excluding the current one
                    sample -= docSmoothingOnlyMass;
                    int tmpPos = (int) sample < position ? (int) sample : (int) sample + 1;
                    newTopic = oneDocTopics[tmpPos];

                }

                if (newTopic == -1) {
                    System.err.println("WorkerRunnable sampling error on doc topic mass: " + origSample + " " + sample + " " + docSmoothingOnlyMass);
                    newTopic = numTopics - 1; // TODO is this appropriate
                    //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
                }

                if (currentTopic != newTopic) {
                    //both decr/incr of global arrays and trees is happening at the end... So at this point they contain the current (not decreased values)
                    // model_old & model_new should be based on decreased values, whereas probabilities (prop_old & prop_new) on the current ones

                    double model_old = (localTopicCounts[currentTopic] + gamma[0] * alpha[currentTopic]) * (currentTypeTopicCounts[currentTopic] + beta[0]) / (tokensPerTopic[currentTopic] + betaSum[0]);
                    double model_new = (localTopicCounts[newTopic] + gamma[0] * alpha[newTopic]) * (currentTypeTopicCounts[newTopic] + beta[0]) / (tokensPerTopic[newTopic] + betaSum[0]);
                    double prop_old = (oldTopic == currentTopic)
                            ? (localTopicCounts[currentTopic] + 1 + gamma[0] * alpha[currentTopic])
                            : (localTopicCounts[currentTopic] + gamma[0] * alpha[currentTopic]);
                    double prop_new = (newTopic == oldTopic)
                            ? (localTopicCounts[newTopic] + 1 + gamma[0] * alpha[newTopic])
                            : (localTopicCounts[newTopic] + gamma[0] * alpha[newTopic]);
                    double acceptance = (model_new * prop_old) / (model_old * prop_new);
                    //acceptance = (temp_new * prop_old * trees[type].getComponent(newTopic)) / (temp_old * prop_new * trees[type].getComponent(oldTopic));

//                    double prop_old2 = (currentTypeTopicCounts[currentTopic] + beta) / (tokensPerTopic[currentTopic] + betaSum);
//                    double prop_new2 = (currentTypeTopicCounts[newTopic] + beta) / (tokensPerTopic[newTopic] + betaSum);
//                    acceptance = (model_new * prop_old * prop_new2) / (model_old * prop_new * prop_old2);
//                    
                    //3. Compare against uniform[0,1]
                    //currentTopic = acceptance >= 1 ? newTopic : random.nextUniform() < acceptance ? newTopic : currentTopic;
                    currentTopic = acceptance >= 1 ? newTopic : ThreadLocalRandom.current().nextDouble() < acceptance ? newTopic : currentTopic;

//                    if (acceptance >= 1 || random.nextUniform() < acceptance) {
//                        currentTopic = newTopic;
//                    }
                }

                // }
            }

            //assert(newTopic != -1);
            //			Put that new topic into the counts
            oneDocTopics[position] = currentTopic;

            localTopicCounts[currentTopic]++;

            if (currentTopic != oldTopic) {

                queue.add(new FastQDelta(oldTopic, currentTopic, type, 0, localTopicCounts[oldTopic], localTopicCounts[currentTopic]));

            }

        }

//        if (shouldSaveState) {
//            // Update the document-topic count histogram,
//            //  for dirichlet estimation
//            //[docLength]++;
//
//            for (int topic = 0; topic < numTopics; topic++) {
//                topicDocCounts[topic][localTopicCounts[topic]]++;
//            }
//        }
    }

}
