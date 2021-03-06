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
import java.util.concurrent.ThreadLocalRandom;

/**
 * A parallel topic model runnable task.
 *
 * @author David Mimno, Andrew McCallum
 */
public class FastWorkerRunnable implements Runnable {

    boolean isFinished = true;
    ArrayList<TopicAssignment> data;
    int startDoc, numDocs;
    protected int numTopics; // Number of topics to be fit
    // These values are used to encode type/topic counts as
    //  count/topic pairs in a single int.
    protected int topicMask;
    protected int topicBits;
    protected int numTypes;
    protected double[] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
    protected double alphaSum;
    protected double beta;   // Prior on per-topic multinomial distribution over words
    protected double betaSum;
    public static final double DEFAULT_BETA = 0.01;
    protected double smoothingOnlyMass = 0.0;
    protected Double[] smoothingOnlyCumValues;
    protected int[][] typeTopicCounts; // indexed by <feature index, topic index>
    protected int[] tokensPerTopic; // indexed by <topic index>
    // for dirichlet estimation
    protected int[] docLengthCounts; // histogram of document sizes
    protected int[][] topicDocCounts; // histogram of document/topic counts, indexed by <topic index, sequence position index>
    boolean shouldSaveState = false;
    boolean shouldBuildLocalCounts = true;
    protected FTree[] trees; //store 
    protected Randoms random;
    int MHsteps = 2;
    boolean useCycleProposals = false;

    public FastWorkerRunnable(int numTopics,
            double[] alpha, double alphaSum,
            double beta, Randoms random,
            ArrayList<TopicAssignment> data,
            int[][] typeTopicCounts,
            int[] tokensPerTopic,
            int startDoc, int numDocs, FTree[] trees, boolean useCycleProposals) {

        this.data = data;

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
        this.betaSum = beta * numTypes;
        this.random = random;

        this.startDoc = startDoc;
        this.numDocs = numDocs;
        this.useCycleProposals = useCycleProposals;

        smoothingOnlyCumValues = new Double[numTopics];

        //System.err.println("WorkerRunnable Thread: " + numTopics + " topics, " + topicBits + " topic bits, " + 
        //				   Integer.toBinaryString(topicMask) + " topic mask");
    }

    /**
     * If there is only one thread, we don't need to go through communication
     * overhead. This method asks this worker not to prepare local type-topic
     * counts. The method should be called when we are using this code in a
     * non-threaded environment.
     */
    public void makeOnlyThread() {
        shouldBuildLocalCounts = false;
    }

    public int[] getTokensPerTopic() {
        return tokensPerTopic;
    }

    public int[][] getTypeTopicCounts() {
        return typeTopicCounts;
    }

    public int[] getDocLengthCounts() {
        return docLengthCounts;
    }

    public int[][] getTopicDocCounts() {
        return topicDocCounts;
    }

    public void initializeAlphaStatistics(int size) {
        docLengthCounts = new int[size];
        topicDocCounts = new int[numTopics][size];
    }

    public void collectAlphaStatistics() {
        shouldSaveState = true;
    }

    public void resetBeta(double beta, double betaSum) {
        this.beta = beta;
        this.betaSum = betaSum;
    }

    /**
     * Once we have sampled the local counts, trash the "global" type topic
     * counts and reuse the space to build a summary of the type topic counts
     * specific to this worker's section of the corpus.
     */
    public void buildLocalTypeTopicCounts() {

        // Clear the topic totals
        Arrays.fill(tokensPerTopic, 0);

        // Clear the type/topic counts,
        for (int[] typeTopicCount : typeTopicCounts) {
            Arrays.fill(typeTopicCount, 0);
        }

        for (int doc = startDoc;
                doc < data.size() && doc < startDoc + numDocs;
                doc++) {

            TopicAssignment document = data.get(doc);

            FeatureSequence tokens = (FeatureSequence) document.instance.getData();
            FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;

            int[] topics = topicSequence.getFeatures();
            for (int position = 0; position < tokens.size(); position++) {

                int topic = topics[position];

                if (topic == ParallelTopicModel.UNASSIGNED_TOPIC) {
                    continue;
                }

                tokensPerTopic[topic]++;

                int type = tokens.getIndexAtPosition(position);

                typeTopicCounts[type][topic]++;
            }
        }
    }

    // p(w|t=z, all) = (alpha[topic] + topicPerDocCounts[d])       *   ( (typeTopicCounts[w][t]/(tokensPerTopic[topic] + betaSum)) + beta/(tokensPerTopic[topic] + betaSum)  )
    // masses:         alphasum     + select a random topics from doc       FTree for active only topics (leave 2-3 spare)                     common FTree f
    //              (binary search)                                               get index from typeTopicsCount
    public void run() {

        try {

            if (!isFinished) {
                System.out.println("already running!");
                return;
            }

            isFinished = false;

            // Initialize the smoothing-only sampling bucket (Sum(a[i])
            smoothingOnlyMass = 0;
            // cachedCoefficients cumulative array that will be used for binary search
            for (int topic = 0; topic < numTopics; topic++) {
                smoothingOnlyMass += alpha[topic];
                smoothingOnlyCumValues[topic] = smoothingOnlyMass;
            }

            //init trees
//            double[] temp = new double[numTopics];
//            for (int w = 0; w < numTypes; ++w) {
//
//                int[] currentTypeTopicCounts = typeTopicCounts[w];
//                for (int currentTopic = 0; currentTopic < numTopics; currentTopic++) {
//
//                    temp[currentTopic] = (currentTypeTopicCounts[currentTopic] + beta) / (tokensPerTopic[currentTopic] + betaSum);
//                }
//               
//                //trees[w].init(numTopics);
//                trees[w] = new FTree(temp);
//                //reset temp
//                Arrays.fill(temp, 0);
//
//            }
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

            if (shouldBuildLocalCounts) {
                buildLocalTypeTopicCounts();
            }

            shouldSaveState = false;
            isFinished = true;

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

        int[] oneDocTopics = topicSequence.getFeatures();
        double[] topicDocWordMasses = new double[numTopics];
        int[] currentTypeTopicCounts;
        int[] localTopicIndex = new int[numTopics];
        int type, oldTopic, newTopic;

        int docLength = tokenSequence.getLength();

        int[] localTopicCounts = new int[numTopics];

        //Double[] topicDocWordMasses = new Double[numTopics];
        //		populate topic counts
        for (int position = 0; position < docLength; position++) {
            if (oneDocTopics[position] == ParallelTopicModel.UNASSIGNED_TOPIC) {
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

            if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {

                // Decrement the local doc/topic counts
                localTopicCounts[oldTopic]--;

                // Maintain the dense index, if we are deleting
                //  the old topic
                if (localTopicCounts[oldTopic] == 0) {

                    // First get to the dense location associated with
                    //  the old topic.
                    denseIndex = 0;

                    // We know it's in there somewhere, so we don't 
                    //  need bounds checking.
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

                // Decrement the global type topic count totals
                currentTypeTopicCounts[oldTopic]--;
                // Decrement the global topic count totals
                tokensPerTopic[oldTopic]--;
                assert (tokensPerTopic[oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";

                trees[type].update(oldTopic, (alpha[oldTopic] * (currentTypeTopicCounts[oldTopic] + beta) / (tokensPerTopic[oldTopic] + betaSum)));
            }

            //		compute word / doc mass for binary search
            double topicDocWordMass = 0.0;
            //double[] topicDocWordMasses = new double[nonZeroTopics];
            //Arrays.fill(topicDocWordMasses, (Double)0.0);

            for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
                int topic = localTopicIndex[denseIndex];
                int n = localTopicCounts[topic];

                //	initialize the normalization constant for the (B * n_{t|d}) term
                topicDocWordMass += n * (currentTypeTopicCounts[topic] + beta) / (tokensPerTopic[topic] + betaSum);
                topicDocWordMasses[denseIndex] = topicDocWordMass;

            }

            double nextUniform = ThreadLocalRandom.current().nextDouble();
            double sample = nextUniform * (topicDocWordMass + trees[type].tree[1]);
            newTopic = -1;

            //double sample = ThreadLocalRandom.current().nextDouble() * (topicDocWordMass + trees[type].tree[1]);
            if (sample < topicDocWordMass) {

                int tmp = lower_bound(topicDocWordMasses, sample, nonZeroTopics);
                int tmpnewTopic = localTopicIndex[tmp]; //actual topic
                
                for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
                    int topic = localTopicIndex[denseIndex];
                    int n = localTopicCounts[topic];

                    sample -= n * (currentTypeTopicCounts[topic] + beta) / (tokensPerTopic[topic] + betaSum);

                    if (sample <= 0.0) {
                        newTopic = topic;
                        break;
                    }
                }
                
                if (tmpnewTopic!=newTopic) 
                    System.err.println("Binary search error: " + tmpnewTopic + " " + newTopic);

            } else {
                //sample -= topicDocWordMass;
                newTopic = trees[type].sample(nextUniform);
            }

            if (newTopic == -1) {
                System.err.println("WorkerRunnable sampling error on word topic mass: " + sample + " " + trees[type].tree[1]);
                newTopic = numTopics - 1; // TODO is this appropriate
                //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
            }

            //assert(newTopic != -1);
            //			Put that new topic into the counts
            oneDocTopics[position] = newTopic;

            localTopicCounts[newTopic]++;

            // If this is a new topic for this document,
            //  add the topic to the dense index.
            if (localTopicCounts[newTopic] == 1) {

                // First find the point where we 
                //  should insert the new topic by going to
                //  the end (which is the only reason we're keeping
                //  track of the number of non-zero
                //  topics) and working backwards
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

            currentTypeTopicCounts[newTopic]++;

            tokensPerTopic[newTopic]++;

            trees[type].update(newTopic, (alpha[newTopic] * (currentTypeTopicCounts[newTopic] + beta) / (tokensPerTopic[newTopic] + betaSum)));

//            if (newTopic != oldTopic) {
//
//                currentTypeTopicCounts[oldTopic]--;
//                currentTypeTopicCounts[newTopic]++;
//
//                tokensPerTopic[oldTopic]--;
//                assert (tokensPerTopic[oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
//                tokensPerTopic[newTopic]++;
//
//                trees[type].update(oldTopic, (alpha[oldTopic] * (currentTypeTopicCounts[oldTopic] + beta) / (tokensPerTopic[oldTopic] + betaSum)));
//                trees[type].update(newTopic, (alpha[newTopic] * (currentTypeTopicCounts[newTopic] + beta) / (tokensPerTopic[newTopic] + betaSum)));
//            }
        }

        if (shouldSaveState) {
            // Update the document-topic count histogram,
            //  for dirichlet estimation
            docLengthCounts[docLength]++;

            for (int topic = 0; topic < numTopics; topic++) {
                topicDocCounts[topic][localTopicCounts[topic]]++;
            }
        }

    }

    protected void sampleTopicsForOneDocCyclingProposals(FeatureSequence tokenSequence,
            FeatureSequence topicSequence,
            boolean readjustTopicsAndStats /* currently ignored */) {

        int[] oneDocTopics = topicSequence.getFeatures();

        int[] currentTypeTopicCounts;
        int type, oldTopic, newTopic, currentTopic;

        int docLength = tokenSequence.getLength();
        int i;

        int[] localTopicCounts = new int[numTopics];

        //		populate topic counts
        for (int position = 0; position < docLength; position++) {
            if (oneDocTopics[position] == ParallelTopicModel.UNASSIGNED_TOPIC) {
                continue;
            }
            localTopicCounts[oneDocTopics[position]]++;
        }

        boolean useDocProposal = true;
        //	Iterate over the positions (words) in the document 
        for (int position = 0; position < docLength; position++) {
            type = tokenSequence.getIndexAtPosition(position);
            oldTopic = oneDocTopics[position];

            currentTypeTopicCounts = typeTopicCounts[type];

            if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {

                // Decrement the local doc/topic counts
                localTopicCounts[oldTopic]--;

                // Multi core (queue based) approximation: All global counts will be updated at the end 
                // Decrement the global type topic count totals
//                currentTypeTopicCounts[oldTopic]--;
//                // Decrement the global topic count totals
//                tokensPerTopic[oldTopic]--;
//                assert (tokensPerTopic[oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
//                trees[type].update(oldTopic, ((currentTypeTopicCounts[oldTopic] + beta) / (tokensPerTopic[oldTopic] + betaSum)));
            }

            currentTopic = oldTopic;

            for (int MHstep = 0; MHstep < MHsteps; MHstep++) {

                //  if (!useDocProposal) {
                //Sample Word topic mass
                double sample = ThreadLocalRandom.current().nextDouble();// * (trees[type].tree[1]);

                //	Make sure it actually gets set
                newTopic = -1;

                newTopic = trees[type].sample(sample);

                if (newTopic == -1) {
                    System.err.println("WorkerRunnable sampling error on word topic mass: " + sample + " " + trees[type].tree[1]);
                    newTopic = numTopics - 1; // TODO is this appropriate
                    //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
                }

                if (currentTopic != newTopic) {
                    //2. Find acceptance probability
                    // due to queue based multi core approximation we should decrement global counts whenever appropriate
                    double temp_old = (oldTopic == currentTopic) ? (localTopicCounts[currentTopic] + alpha[currentTopic]) * (currentTypeTopicCounts[currentTopic] - 1 + beta) / (tokensPerTopic[currentTopic] - 1 + betaSum)
                            : (localTopicCounts[currentTopic] + alpha[currentTopic]) * trees[type].getComponent(currentTopic);
                    double temp_new = (oldTopic == newTopic) ? (localTopicCounts[newTopic] + alpha[newTopic]) * (currentTypeTopicCounts[newTopic] - 1 + beta) / (tokensPerTopic[newTopic] - 1 + betaSum)
                            : (localTopicCounts[newTopic] + alpha[newTopic]) * trees[type].getComponent(newTopic);
                    //double prop_old = (oldTopic == currentTopic) ? (localTopicCounts[currentTopic] + 1 + alpha[currentTopic]) : (localTopicCounts[currentTopic] + alpha[currentTopic]);
                    //double prop_new = (newTopic == oldTopic) ? (localTopicCounts[newTopic] + 1 + alpha[newTopic]) : (localTopicCounts[newTopic] + alpha[newTopic]);

                    double acceptance = (temp_new * trees[type].getComponent(currentTopic)) / (temp_old * trees[type].getComponent(newTopic)); // trees[type].getComponent(currentTopic) contains alra
                    //acceptance = (temp_new * prop_new * trees[type].getComponent(oldTopic)) / (temp_old * prop_old * trees[type].getComponent(newTopic));

                    // acceptance = temp_new / temp_old;
                    //3. Compare against uniform[0,1]
                    currentTopic = acceptance >= 1 ? newTopic : ThreadLocalRandom.current().nextDouble() < acceptance ? newTopic : currentTopic;
//                    if (acceptance >= 1 || random.nextUniform() < acceptance) {
//                        currentTopic = newTopic;
//                    }
                }
               // } else {

                // Sample Doc topic mass 
                // we can sample from doc topic mass immediately as it doesn't get affected by current topic selection!! 
                //--> thus we get & check both samples from cycle proposal in every step
                sample = ThreadLocalRandom.current().nextDouble() * (smoothingOnlyMass + docLength - 1);
                double origSample = sample;

                //	Make sure it actually gets set
                newTopic = -1;

                if (sample < smoothingOnlyMass) {

                    newTopic = lower_bound(smoothingOnlyCumValues, sample);

                } else { //just select one random topic from DocTopics excluding the current one
                    sample -= smoothingOnlyMass;
                    int tmpPos = (int) sample < position ? (int) sample : (int) sample + 1;
                    newTopic = oneDocTopics[tmpPos];

                }

                if (newTopic == -1) {
                    System.err.println("WorkerRunnable sampling error on doc topic mass: " + origSample + " " + sample + " " + smoothingOnlyMass);
                    newTopic = numTopics - 1; // TODO is this appropriate
                    //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
                }

                if (currentTopic != newTopic) {
                    //2. Find acceptance probability
                    //double temp_old = (localTopicCounts[currentTopic] + alpha[currentTopic]) * (currentTypeTopicCounts[currentTopic] + beta) / (tokensPerTopic[currentTopic] + betaSum);
                    //double temp_new = (localTopicCounts[newTopic] + alpha[newTopic]) * (currentTypeTopicCounts[newTopic] + beta) / (tokensPerTopic[newTopic] + betaSum);
                    double temp_old = (oldTopic == currentTopic) ? (localTopicCounts[currentTopic] + alpha[currentTopic]) * (currentTypeTopicCounts[currentTopic] - 1 + beta) / (tokensPerTopic[currentTopic] - 1 + betaSum)
                            : (localTopicCounts[currentTopic] + alpha[currentTopic]) * trees[type].getComponent(currentTopic);
                    double temp_new = (oldTopic == newTopic) ? (localTopicCounts[newTopic] + alpha[newTopic]) * (currentTypeTopicCounts[newTopic] - 1 + beta) / (tokensPerTopic[newTopic] - 1 + betaSum)
                            : (localTopicCounts[newTopic] + alpha[newTopic]) * trees[type].getComponent(newTopic);
                    double prop_old = (oldTopic == currentTopic) ? (localTopicCounts[currentTopic] + 1 + alpha[currentTopic]) : (localTopicCounts[currentTopic] + alpha[currentTopic]);
                    double prop_new = (newTopic == oldTopic) ? (localTopicCounts[newTopic] + 1 + alpha[newTopic]) : (localTopicCounts[newTopic] + alpha[newTopic]);
                    double acceptance = (temp_new * prop_old) / (temp_old * prop_new);
                    //acceptance = (temp_new * prop_old * trees[type].getComponent(newTopic)) / (temp_old * prop_new * trees[type].getComponent(oldTopic));

                    //3. Compare against uniform[0,1]
                    //currentTopic = acceptance >= 1 ? newTopic : random.nextUniform() < acceptance ? newTopic : currentTopic;
                    currentTopic = acceptance >= 1 ? newTopic : ThreadLocalRandom.current().nextDouble() < acceptance ? newTopic : currentTopic;

//                    if (acceptance >= 1 || random.nextUniform() < acceptance) {
//                        currentTopic = newTopic;
//                    }
                }

                // }
            }

            useDocProposal = !useDocProposal;
            //assert(newTopic != -1);

            //			Put that new topic into the counts
            oneDocTopics[position] = currentTopic;

            localTopicCounts[currentTopic]++;

            if (currentTopic != oldTopic) {

                // Decrement the global type topic count totals
                currentTypeTopicCounts[oldTopic]--;
                // Decrement the global topic count totals
                tokensPerTopic[oldTopic]--;
                assert (tokensPerTopic[oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
                //Update tree
                trees[type].update(oldTopic, ((currentTypeTopicCounts[oldTopic] + beta) / (tokensPerTopic[oldTopic] + betaSum)));

                //add new count
                currentTypeTopicCounts[currentTopic]++;
                tokensPerTopic[currentTopic]++;
                trees[type].update(currentTopic, ((currentTypeTopicCounts[currentTopic] + beta) / (tokensPerTopic[currentTopic] + betaSum)));
            }

        }

        if (shouldSaveState) {
            // Update the document-topic count histogram,
            //  for dirichlet estimation
            docLengthCounts[docLength]++;

            for (int topic = 0; topic < numTopics; topic++) {
                topicDocCounts[topic][localTopicCounts[topic]]++;
            }
        }

    }
}
