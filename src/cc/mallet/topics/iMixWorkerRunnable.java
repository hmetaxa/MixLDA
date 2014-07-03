/* Copyright (C) 2013 Omiros Metaxas */
package cc.mallet.topics;

import java.util.Arrays;
import java.util.ArrayList;

//import java.util.zip.*;
//import java.io.*;
//import java.text.NumberFormat;
import cc.mallet.types.*;
import cc.mallet.util.Randoms;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;
//import java.util.HashMap;
import java.util.concurrent.ThreadLocalRandom;

/**
 * A parallel semi supervised topic model runnable task.
 *
 * @author Omiros Metaxas extending MALLET Parallel topic model of author David
 * Mimno, Andrew McCallum test BOX sync
 *
 */
public class iMixWorkerRunnable implements Runnable {

    public class MassValue {

        public double topicTermMass;
        public double topicBetaMass;
        public double smoothingOnlyMass;
        //public int nonZeroTopics;
        //public double few;//frequency exclusivity weight we have an array for that
    }
    boolean isFinished = true;
    boolean ignoreLabels = false;
    //boolean ignoreSkewness = false;
    ArrayList<MixTopicModelTopicAssignment> data;
    int startDoc, numDocs;
    protected int numTopics; // Number of topics to be fit
    //protected int numCommonTopics;
    protected byte numModalities;
    //protected int numIndependentTopics;
    // These values are used to encode type/topic counts as
    //  count/topic pairs in a single int.
    protected int topicMask;
    protected int topicBits;
    protected int[] numTypes;
    protected double[] avgTypeCount; //not used for now
    protected int[][] typeTotals; //not used for now
    //homer
    //protected final double[] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics

    protected double[] gamma;

    protected TDoubleArrayList[] alpha;
    protected double[] alphaSum;
    protected double[] beta;   // Prior on per-topic multinomial distribution over words
    protected double[] betaSum;
    public static final double DEFAULT_BETA = 0.01;
    //homer 
    protected double[] smoothingOnlyMass;// = 0.0;
    protected TDoubleArrayList[] smoothOnlyCachedCoefficients;
    protected TIntArrayList[][] typeTopicCounts; //per modality // indexed by <modalityIndex, feature index, topic index>
    protected TIntArrayList[] tokensPerTopic; //per modality// indexed by <modalityIndex,topic index>
    //protected int[][][] typeTopicCounts; // indexed by <modality index, feature index, topic index>
    //protected int[][] tokensPerTopic; // indexed by <modality index, topic index>
    // for dirichlet estimation
    protected int[][] docLengthCounts; // histogram of document sizes
    protected TIntObjectHashMap<int[]>[] topicDocCounts; // histogram of document/topic counts, indexed by <topic index, sequence position index>
    boolean shouldSaveState = false;
    boolean shouldBuildLocalCounts = true;
    protected Randoms random;
    // The skew index of eachType
    protected final double[][] typeSkewIndexes;
    protected double[] skewWeight;// = 1;
    protected double[][] p_a; // a for beta prior for modalities correlation
    protected double[][] p_b; // b for beta prir for modalities correlation
    protected boolean fastSampling = false; // b for beta prir for modalities correlation
    double[][][] pDistr_Mean; // modalities correlation distribution accross documents (used in a, b beta params optimization)
    //double[][][] pDistr_Var; // modalities correlation distribution accross documents (used in a, b beta params optimization)
    //double avgSkew = 0;

    public iMixWorkerRunnable(int numTopics, int numIndependentTopics,
            TDoubleArrayList[] alpha, double[] alphaSum,
            double[] beta, Randoms random,
            final ArrayList<MixTopicModelTopicAssignment> data,
            TIntArrayList[][] typeTopicCounts,
            TIntArrayList[] tokensPerTopic,
            int startDoc, int numDocs, boolean ignoreLabels, byte numModalities,
            double[][] typeSkewIndexes, iMixParallelTopicModel.SkewType skewOn, double[] skewWeight, double[][] p_a, double[][] p_b, double[] gamma) {

        this.data = data;

        this.numTopics = numTopics;
        //this.numIndependentTopics = numIndependentTopics;
        this.numModalities = numModalities;
        //this.numCommonTopics = numTopics - numIndependentTopics * numModalities;
        this.numTypes = new int[numModalities];
        this.betaSum = new double[numModalities];
        this.skewWeight = skewWeight;
        this.p_a = p_a;  //new double[numModalities][numModalities];
        this.p_b = p_b;
        this.smoothingOnlyMass = new double[numModalities];
        this.smoothOnlyCachedCoefficients = new TDoubleArrayList[numModalities];
        this.typeSkewIndexes = typeSkewIndexes;
        this.gamma = gamma;

        for (byte i = 0; i < numModalities; i++) {
            this.numTypes[i] = typeTopicCounts[i].length;
            this.betaSum[i] = beta[i] * numTypes[i];
            this.smoothOnlyCachedCoefficients[i] = new TDoubleArrayList(numTopics);
            this.smoothOnlyCachedCoefficients[i].fill(0, numTopics, 0);

            //Arrays.fill(this.p[i], 1d);
        }

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

        this.alphaSum = alphaSum;
        this.alpha = alpha;
        this.beta = beta;

        this.random = random;

        this.startDoc = startDoc;
        this.numDocs = numDocs;

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

    public TIntArrayList[] getTokensPerTopic() {
        return tokensPerTopic;
    }

    public TIntArrayList[][] getTypeTopicCounts() {
        return typeTopicCounts;
    }

    public int[][] getDocLengthCounts() {
        return docLengthCounts;
    }

    public TIntObjectHashMap<int[]>[] getTopicDocCounts() {
        return topicDocCounts;
    }

    public double[][][] getPDistr_Mean() {
        return pDistr_Mean;
    }

//    public double[][][] getPDistr_Var() {
//        return pDistr_Var;
//    }
    public void initializeAlphaStatistics(int size) {
        docLengthCounts = new int[numModalities][size];
        topicDocCounts = new TIntObjectHashMap[numModalities];
        for (byte i = 0; i < numModalities; i++) {
            topicDocCounts[i] = new TIntObjectHashMap<int[]>(numTopics);
            for (int topic = 0; topic < numTopics; topic++) {
                topicDocCounts[i].put(topic, new int[docLengthCounts[i].length]);
            }
        }
        //  [size];
    }

    public void collectAlphaStatistics() {
        shouldSaveState = true;
    }

    public void resetBeta(double[] beta, double[] betaSum) {
        this.beta = beta;
        this.betaSum = betaSum;
    }

    public void resetP_a(double[][] p_a) {
        this.p_a = p_a;
    }

    public void resetP_b(double[][] p_b) {
        this.p_b = p_b;
    }

    public void resetSkewWeight(double[] skewWeight) {
        this.skewWeight = skewWeight;
    }

    public void resetGamma(double[] gamma) {
        this.gamma = gamma;
    }

    public void resetFastSampling(boolean fastSampling) {
        this.fastSampling = fastSampling;
    }

    /**
     * Once we have sampled the local counts, trash the "global" type topic
     * counts and reuse the space to build a summary of the type topic counts
     * specific to this worker's section of the corpus.
     */
    public void buildLocalTypeTopicCounts() {

        // Clear the type/topic counts, only 
        //  looking at the entries before the first 0 entry.
        for (byte i = 0; i < numModalities; i++) {

            // Clear the topic totals
            //tokensPerTopic[i].reset();
            tokensPerTopic[i].fill(0, numTopics, 0);
            //Arrays.fill(tokensPerTopic[i], 0);

            for (int type = 0; type < typeTopicCounts[i].length; type++) {

                typeTopicCounts[i][type].fill(0, numTopics, 0);//.reset();
//                int[] topicCounts = typeTopicCounts[i][type];
//
//                int position = 0;
//                while (position < topicCounts.length
//                        && topicCounts[position] > 0) {
//                    topicCounts[position] = 0;
//                    position++;
//                }
            }
        }

        for (int doc = startDoc;
                doc < data.size() && doc < startDoc + numDocs;
                doc++) {
            for (byte i = 0; i < numModalities; i++) {

                TopicAssignment document = data.get(doc).Assignments[i];
                if (document != null) {
                    FeatureSequence tokens = (FeatureSequence) document.instance.getData();
                    FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;

                    int[] topics = topicSequence.getFeatures();
                    for (int position = 0; position < tokens.size(); position++) {

                        int topic = topics[position];

                        if (topic == ParallelTopicModel.UNASSIGNED_TOPIC) {
                            System.err.println(" buildLocalTypeTopicCounts UNASSIGNED_TOPIC");
                            continue;
                        }

                        tokensPerTopic[i].set(topic, tokensPerTopic[i].get(topic) + 1);

                        // The format for these arrays is 
                        //  the topic in the rightmost bits
                        //  the count in the remaining (left) bits.
                        // Since the count is in the high bits, sorting (desc)
                        //  by the numeric value of the int guarantees that
                        //  higher counts will be before the lower counts.
                        int type = tokens.getIndexAtPosition(position);

                        TIntArrayList currentTypeTopicCounts = typeTopicCounts[i][type];

                        // Start by assuming that the array is either empty
                        //  or is in sorted (descending) order.
                        // Here we are only adding counts, so if we find 
                        //  an existing location with the topic, we only need
                        //  to ensure that it is not larger than its left neighbor.
                        int index = 0;
                        int currentTopic = currentTypeTopicCounts.get(index) & topicMask;
                        int currentValue;

                        while (currentTypeTopicCounts.get(index) > 0 && currentTopic != topic) {
                            index++;
                            if (index == currentTypeTopicCounts.size()) {
                                System.out.println("overflow on type " + type);
                            }
                            currentTopic = currentTypeTopicCounts.get(index) & topicMask;
                        }
                        currentValue = currentTypeTopicCounts.get(index) >> topicBits;

                        if (currentValue == 0) {
                            // new value is 1, so we don't have to worry about sorting
                            //  (except by topic suffix, which doesn't matter)

                            currentTypeTopicCounts.set(index,
                                    (1 << topicBits) + topic);
                        } else {
                            currentTypeTopicCounts.set(index,
                                    ((currentValue + 1) << topicBits) + topic);

                            // Now ensure that the array is still sorted by 
                            //  bubbling this value up.
                            while (index > 0
                                    && currentTypeTopicCounts.get(index) > currentTypeTopicCounts.get(index - 1)) {
                                int temp = currentTypeTopicCounts.get(index);
                                currentTypeTopicCounts.set(index, currentTypeTopicCounts.get(index - 1));
                                currentTypeTopicCounts.set(index - 1, temp);

                                index--;
                            }
                        }
                    }
                }
            }
        }
    }

    public void run() {

        try {

            if (!isFinished) {
                System.out.println("already running!");
                return;
            }
            //this.pDistr_Var = new double[numModalities][numModalities][data.size()];
            this.pDistr_Mean = new double[numModalities][numModalities][data.size()];

            isFinished = false;

            // Initialize the smoothing-only sampling bucket
            Arrays.fill(smoothingOnlyMass, 0d);

            for (byte i = 0; i < numModalities; i++) {

                // Initialize the cached coefficients, using only smoothing.
                //  These values will be selectively replaced in documents with
                //  non-zero counts in particular topics.
                //for (int topic = 0; topic < numCommonTopics; topic++) {
                for (int topic = 0; topic < numTopics; topic++) {
                    smoothingOnlyMass[i] += gamma[i] * alpha[i].get(topic) * beta[i] / (tokensPerTopic[i].get(topic) + betaSum[i]);
                    smoothOnlyCachedCoefficients[i].set(topic, gamma[i] * alpha[i].get(topic) / (tokensPerTopic[i].get(topic) + betaSum[i]));
                }

//                for (int topic = numCommonTopics + i * numIndependentTopics; topic < numCommonTopics + (i + 1) * numIndependentTopics; topic++) {
//
//                    smoothingOnlyMass[i] += alpha[i].get(topic) * beta[i] / (tokensPerTopic[i].get(topic) + betaSum[i]);
//                    smoothOnlyCachedCoefficients[i].set(topic, alpha[i].get(topic) / (tokensPerTopic[i].get(topic) + betaSum[i]));
//                }
            }

            for (int doc = startDoc;
                    doc < data.size() && doc < startDoc + numDocs;
                    doc++) {

                /*
                 if (doc % 10000 == 0) {
                 System.out.println("processing doc " + doc);
                 }
                 */
//                if (doc % 10 == 0) {
//                    System.out.println("processing doc " + doc);
//                }
//                if (data.get(doc).Assignments[0] != null) {
//                    FeatureSequence tokenSequence =
//                            (FeatureSequence) data.get(doc).Assignments[0].instance.getData();
//                    LabelSequence topicSequence =
//                            (LabelSequence) data.get(doc).Assignments[0].topicSequence;
//                    sampleTopicsForOneDoc(tokenSequence, topicSequence);
//
//                }
                sampleTopicsForOneDoc(doc);

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

//    protected void addNewTopicAndUpdateSmoothing() {
//
//        //newTopics++;
//        for (byte m = 0; m < numModalities; m++) {
//            // Initialize the cached coefficients, using only smoothing.
//            //  These values will be selectively replaced in documents with
//            //  non-zero counts in particular topics.
//            alpha[m].set(numTopics, alpha[m].get(numTopics - 1));
//            smoothingOnlyMass[m] += gamma[m] * alpha[m].get(numTopics) * beta[m] / (tokensPerTopic[m].get(numTopics) + betaSum[m]);
//            smoothOnlyCachedCoefficients[m].set(numTopics, gamma[m] * alpha[m].get(numTopics) / (tokensPerTopic[m].get(numTopics) + betaSum[m]));
//
//        }
//        numTopics++;
//    }
    protected int initSampling(
            MixTopicModelTopicAssignment doc,
            TDoubleArrayList[] totalMassPerModalityAndTopic,
            int[][] oneDocTopics,
            FeatureSequence[] tokenSequence,
            int[] docLength,
            TIntArrayList[] localTopicCounts,
            TIntArrayList localTopicIndex) {

        for (byte i = 0; i < numModalities; i++) {
            docLength[i] = 0;
            totalMassPerModalityAndTopic[i] = new TDoubleArrayList();
            totalMassPerModalityAndTopic[i].fill(0, numTopics, 0);

            localTopicCounts[i] = new TIntArrayList();
            localTopicCounts[i].fill(0, numTopics, 0);

            if (doc.Assignments[i] != null) {
                oneDocTopics[i] = doc.Assignments[i].topicSequence.getFeatures();

                //System.arraycopy(oneDocTopics[i], 0, doc.Assignments[i].topicSequence.getFeatures(), 0, doc.Assignments[i].topicSequence.getFeatures().length-1);
                tokenSequence[i] = ((FeatureSequence) doc.Assignments[i].instance.getData());

                docLength[i] = tokenSequence[i].getLength();

                //		populate topic counts
                for (int position = 0; position < docLength[i]; position++) {
                    if (oneDocTopics[i][position] == ParallelTopicModel.UNASSIGNED_TOPIC) {
                        System.err.println(" Init Sampling UNASSIGNED_TOPIC");
                        continue;
                    }
                    localTopicCounts[i].set(oneDocTopics[i][position], localTopicCounts[i].get(oneDocTopics[i][position]) + 1);

                }
            }

//            for (int topic = 0; topic < numCommonTopics; topic++) {
//                totalMassPerModalityAndTopic[i].set(topic, (localTopicCounts[i].get(topic) + alpha.get(topic))
//                        / (docLength[i] + alphaSum));
//            }
//
//            for (int topic = numCommonTopics + i * numIndependentTopics; topic < numCommonTopics + (i + 1) * numIndependentTopics; topic++) {
//                totalMassPerModalityAndTopic[i].set(topic, (localTopicCounts[i].get(topic) + alpha.get(topic))
//                        / (docLength[i] + alphaSum));
//            }
        }
        // Build an array that densely lists the topics that
        //  have non-zero counts.
        int denseIndex = 0;
        for (int topic = 0; topic < numTopics; topic++) {
            int i = 0;
            boolean topicFound = false;
            while (i < numModalities && !topicFound) {
                if (localTopicCounts[i].get(topic) != 0) {
                    localTopicIndex.set(denseIndex, topic);
                    denseIndex++;
                    topicFound = true;
                }
                i++;
            }
        }

        // Record the total number of non-zero topics
        int nonZeroTopics = denseIndex;

        //		Initialize the topic count/beta sampling bucket
        // Initialize cached coefficients and the topic/beta 
        //  normalizing constant.
//        int topic = -1;
//
//        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//            topic = localTopicIndex.get(denseIndex);
//            for (byte m = 0; m < numModalities; m++) {
//
//                if (topic < numCommonTopics || (topic >= numCommonTopics + m * numIndependentTopics && topic < numCommonTopics + (m + 1) * numIndependentTopics)) {
//
//                    //	initialize the total mass per modality: ( n_{t|d}+a(m) / (docLength[m]+aSum))
//                    double normSumN = (localTopicCounts[m].get(topic) + alpha.get(topic))
//                            / (docLength[m] + alphaSum);
//
//                    totalMassPerModalityAndTopic[m].set(topic, totalMassPerModalityAndTopic[m].get(topic) + normSumN);
//
//                }
//
//            }
//        }
        return nonZeroTopics;
    }

    protected void initSamplingForModality(
            TDoubleArrayList cachedCoefficients,
            TDoubleArrayList totalMassOtherModalities,
            TDoubleArrayList[] totalMassPerModalityAndTopic,
            int[] docLength,
            TIntArrayList[] localTopicCounts,
            TIntArrayList localTopicIndex,
            double[] topicBetaMass,
            double[][] p,
            int m, //active modality
            int nonZeroTopics) {

        //cachedCoefficients.reset();
        cachedCoefficients.fill(0, numTopics, 0);
        //totalMassOtherModalities.reset();
        totalMassOtherModalities.fill(0, numTopics, 0);
        int topic = -1;

//        for (topic = 0; topic < numCommonTopics; topic++) {
//
//            for (byte i = 0; i < numModalities; i++) {
//
//                if (i != m) {
//                    totalMassOtherModalities.set(topic, totalMassOtherModalities.get(topic) + p[m][i] * totalMassPerModalityAndTopic[i].get(topic));
//                }
//
//
//                totalMassOtherModalities.set(topic, totalMassOtherModalities.get(topic) * (docLength[m] + alphaSum));
//            }
//        }
        for (int denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {

            topic = localTopicIndex.get(denseIndex);
            //totalMassOtherModalities.set(topic, 0);

//if (topic < numCommonTopics || (topic >= numCommonTopics + m * numIndependentTopics && topic < numCommonTopics + (m + 1) * numIndependentTopics)) {
            //  if (topic < numTopics) {
            for (byte i = 0; i < numModalities; i++) {
                if (i != m && docLength[i] != 0) {
                    totalMassOtherModalities.set(topic, totalMassOtherModalities.get(topic) + p[m][i] * localTopicCounts[i].get(topic) / docLength[i]);
                }
            }

            totalMassOtherModalities.set(topic, totalMassOtherModalities.get(topic) * (docLength[m] + alphaSum[m]));
            //	initialize the normalization constant for the (B * n_{t|d}) term
            double normSumN = (localTopicCounts[m].get(topic) + totalMassOtherModalities.get(topic))
                    / (tokensPerTopic[m].get(topic) + betaSum[m]);

            topicBetaMass[m] += beta[m] * normSumN;
            cachedCoefficients.set(topic, normSumN);

            // }
        }

    }

    protected int removeOldTopicContribution(
            TDoubleArrayList cachedCoefficients,
            TDoubleArrayList totalMassOtherModalities,
            TIntArrayList[] localTopicCounts,
            TIntArrayList localTopicIndex,
            double[] topicBetaMass,
            int oldTopic,
            int nonZeroTopics,
            // final int[] docLength,
            byte m //modality
    //double[][] p
    ) {

        // if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {
        //	Remove this token from all counts. 
        // Remove this topic's contribution to the 
        //  normalizing constants
        smoothingOnlyMass[m] -= gamma[m] * alpha[m].get(oldTopic) * beta[m]
                / (tokensPerTopic[m].get(oldTopic) + betaSum[m]);

        double normSumN = (localTopicCounts[m].get(oldTopic) + totalMassOtherModalities.get(oldTopic))
                / (tokensPerTopic[m].get(oldTopic) + betaSum[m]);

        topicBetaMass[m] -= beta[m] * normSumN;
        //cachedCoefficients.set(oldTopic, normSumN);

        // Decrement the local doc/topic counts
        localTopicCounts[m].set(oldTopic, localTopicCounts[m].get(oldTopic) - 1);

        // Decrement the global topic count totals
        tokensPerTopic[m].set(oldTopic, tokensPerTopic[m].get(oldTopic) - 1);
        assert (tokensPerTopic[m].get(oldTopic) >= 0) : "old Topic " + oldTopic + " below 0";

        // Add the old topic's contribution back into the
        //  normalizing constants.
        smoothingOnlyMass[m] += gamma[m] * alpha[m].get(oldTopic) * beta[m]
                / (tokensPerTopic[m].get(oldTopic) + betaSum[m]);

        smoothOnlyCachedCoefficients[m].set(oldTopic, gamma[m] * alpha[m].get(oldTopic) / (tokensPerTopic[m].get(oldTopic) + betaSum[m]));

        normSumN = (localTopicCounts[m].get(oldTopic) + totalMassOtherModalities.get(oldTopic))
                / (tokensPerTopic[m].get(oldTopic) + betaSum[m]);

        topicBetaMass[m] += beta[m] * normSumN;
        cachedCoefficients.set(oldTopic, normSumN);

        // Maintain the dense index, if we are deleting
        //  the old topic
        boolean isDeletedTopic = localTopicCounts[m].get(oldTopic) == 0;
        byte jj = 0;
        while (isDeletedTopic && jj < numModalities) {
            // if (jj != m) { //do not check m twice
            isDeletedTopic = localTopicCounts[jj].get(oldTopic) == 0;
            // }
            jj++;
        }

        //isDeletedTopic = false;//todo omiros test
        if (isDeletedTopic) {

            // First get to the dense location associated with
            //  the old topic.
            int denseIndex = 0;

            // We know it's in there somewhere, so we don't 
            //  need bounds checking.
            while (localTopicIndex.get(denseIndex) != oldTopic) {
                denseIndex++;
            }

            // shift all remaining dense indices to the left.
            while (denseIndex < nonZeroTopics) {
                if (denseIndex < localTopicIndex.size() - 1) {
                    localTopicIndex.set(denseIndex,
                            localTopicIndex.get(denseIndex + 1));
                }
                denseIndex++;
            }

            nonZeroTopics--;
        }

        //omiors test ... recalc all beta 
        return nonZeroTopics;

    }

    //TODO: I recalc them every time because sometimes I had a sampling error in FindTopicIn Beta Mass.. 
    //I shouldn't need it, thus I should check it again
    protected void recalcBetaAndCachedCoefficients(
            TDoubleArrayList[] cachedCoefficients,
            TIntArrayList[] localTopicCounts,
            TIntArrayList localTopicIndex,
            double[] topicBetaMass,
            int nonZeroTopics,
            final int[] docLength,
            byte m, //modality
            double[][] p) {

        Arrays.fill(topicBetaMass, 0);
        for (byte i = 0; i < numModalities; i++) {
            cachedCoefficients[i].fill(0, numTopics, 0);
            //Arrays.fill(cachedCoefficients[i], 0);
        }

        for (int denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
            int topic = localTopicIndex.get(denseIndex);
            //if (topic < numCommonTopics || (topic >= numCommonTopics + m * numIndependentTopics && topic < numCommonTopics + (m + 1) * numIndependentTopics)) {

            for (byte j = 0; j < numModalities; j++) {
                if (docLength[j] > 0) {
                    double normSumN = p[m][j] * localTopicCounts[j].get(topic)
                            / (docLength[j] * (tokensPerTopic[m].get(topic) + betaSum[m]));

                    topicBetaMass[m] += beta[m] * normSumN;
                    cachedCoefficients[m].set(topic, cachedCoefficients[m].get(topic) + normSumN);
                }
            }
            //}

        }
    }

    protected double calcTopicScores(
            TDoubleArrayList cachedCoefficients,
            int oldTopic,
            byte m,
            TDoubleArrayList topicTermScores,
            TIntArrayList currentTypeTopicCounts,
            int[] docLength,
            double termSkew) {
        // Now go over the type/topic counts, decrementing
        //  where appropriate, and calculating the score
        //  for each topic at the same time.

        //final double[][] smoothOnlyCachedCoefficientsLcl = this.smoothOnlyCachedCoefficients;
        int index = 0;
        int currentTopic, currentValue;
        double score;

//        if (oldTopic == ParallelTopicModel.UNASSIGNED_TOPIC) {
//            System.err.println(" Remove Old Topic Contribution UNASSIGNED_TOPIC");
//        }
        boolean alreadyDecremented = (oldTopic == ParallelTopicModel.UNASSIGNED_TOPIC);

        double topicTermMass = 0.0;

        while (index < currentTypeTopicCounts.size()
                && currentTypeTopicCounts.get(index) > 0) {

            currentTopic = currentTypeTopicCounts.get(index) & topicMask;
            currentValue = currentTypeTopicCounts.get(index) >> topicBits;

            if (!alreadyDecremented && currentTopic == oldTopic) {

                // We're decrementing and adding up the 
                //  sampling weights at the same time, but
                //  decrementing may require us to reorder
                //  the topics, so after we're done here,
                //  look at this cell in the array again.
                currentValue--;
                if (currentValue == 0) {
                    currentTypeTopicCounts.set(index, 0);
                } else {
                    currentTypeTopicCounts.set(index,
                            (currentValue << topicBits) + oldTopic);
                }

                // Shift the reduced value to the right, if necessary.
                int subIndex = index;
                while (subIndex < currentTypeTopicCounts.size() - 1
                        && currentTypeTopicCounts.get(subIndex) < currentTypeTopicCounts.get(subIndex + 1)) {
                    int temp = currentTypeTopicCounts.get(subIndex);
                    currentTypeTopicCounts.set(subIndex, currentTypeTopicCounts.get(subIndex + 1));
                    currentTypeTopicCounts.set(subIndex + 1, temp);

                    subIndex++;
                }

                alreadyDecremented = true;
            } else {

                // re scale topic term scores (probability mass related to token/label type)
                //types having large skew--> not ver discriminative. Thus I decrease their probability mass
                // skewWeight is used for normalization. Thus the total probability mass (topic term scores) related to types remains almost constant
                // but is share based on type skewness promoting types that are discriminative
                double skewInx = skewWeight[m] * (1 + termSkew); //1;
                //if (!ignoreSkewness) {
                //skewInx = skewWeight[m] * (1 + termSkew);
                // }

                //add normalized smoothingOnly coefficient 
                score = (cachedCoefficients.get(currentTopic) + smoothOnlyCachedCoefficients[m].get(currentTopic)) * currentValue * skewInx;

                topicTermMass += score;
                topicTermScores.set(index, score);

                index++;
            }
        }

        return topicTermMass;
    }

    protected int findNewTopicInTopicTermMass(
            TDoubleArrayList topicTermScores,
            TIntArrayList currentTypeTopicCounts,
            double sample) {

        int newTopic = -1;
        int currentValue;
        int i = -1;
        while (sample > 0) {
            i++;
            sample -= topicTermScores.get(i);
        }
        if (i >= 0) {
            newTopic = currentTypeTopicCounts.get(i) & topicMask;
            currentValue = currentTypeTopicCounts.get(i) >> topicBits;

            currentTypeTopicCounts.set(i, ((currentValue + 1) << topicBits) + newTopic);

            // Bubble the new value up, if necessary
            while (i > 0
                    && currentTypeTopicCounts.get(i) > currentTypeTopicCounts.get(i - 1)) {
                int temp = currentTypeTopicCounts.get(i);
                currentTypeTopicCounts.set(i, currentTypeTopicCounts.get(i - 1));
                currentTypeTopicCounts.set(i - 1, temp);

                i--;
            }
        }
        return newTopic;
    }

    protected int findNewTopicInBetaMass(
            TIntArrayList[] localTopicCounts,
            TIntArrayList localTopicIndex,
            TDoubleArrayList totalMassOtherModalities,
            int nonZeroTopics,
            byte m,
            final int[] docLength,
            double sample,
            double[][] p) {

        sample /= beta[m];
        int topic = -1;
        int denseIndex = 0;

        while (denseIndex < nonZeroTopics && sample > 0) {
            topic = localTopicIndex.get(denseIndex);
            //if (topic < numCommonTopics || (topic >= numCommonTopics + m * numIndependentTopics && topic < numCommonTopics + (m + 1) * numIndependentTopics)) {

            double normSumN = (localTopicCounts[m].get(topic) + totalMassOtherModalities.get(topic))
                    / (tokensPerTopic[m].get(topic) + betaSum[m]);

            sample -= normSumN;

            //}
            denseIndex++;
        }
//       
        if (sample > 0) {
            return -1; // error in rounding (?) I should check it again
        }
        return topic;
    }

    protected int findNewTopicInSmoothingMass(
            double sample,
            byte m,
            final int[] docLength) {

        int newTopic = -1;
        //sample *= docLength[m];
        //sample /= beta[m];

        int topic = 0;

        //while (sample > 0.0 && topic < numCommonTopics) {
        while (sample > 0.0 && topic < numTopics) {
            sample -= gamma[m] * alpha[m].get(topic) * beta[m]
                    / (tokensPerTopic[m].get(topic) + betaSum[m]);
            if (sample <= 0.0) {
                newTopic = topic;
            }
            topic++;
        }

//        //search independent topics
//        topic = 0;
//        int indTopic = 0;
//        while (sample > 0.0 && topic < numIndependentTopics) {
//
//            indTopic = numCommonTopics + m * numIndependentTopics + topic;
//            sample -= alpha[m].get(indTopic) * beta[m]
//                    / (tokensPerTopic[m].get(indTopic) + betaSum[m]);
//
//            if (sample <= 0.0) {
//                newTopic = indTopic;
//
//            }
//
//            topic++;
//        }
        if (newTopic == -1) {
            return -1;
            //          newTopic = numTopics - 1;
        }

        return newTopic;
    }

    protected int findNewTopic(
            TIntArrayList[] localTopicCounts,
            TIntArrayList localTopicIndex,
            TDoubleArrayList totalMassOtherModalities,
            double[] topicBetaMass,
            int nonZeroTopics,
            byte m,
            double topicTermMass,
            TDoubleArrayList topicTermScores,
            TIntArrayList currentTypeTopicCounts,
            int[] docLength,
            double sample,
            double[][] p,
            int oldTopic) {
        //	Make sure it actually gets set

        int newTopic = -1;
        //int index = 0;

        double origSample = sample;
        String samplingBucket = "";
        if (sample <= topicTermMass) {
            samplingBucket = "TermMass";
            newTopic = findNewTopicInTopicTermMass(
                    topicTermScores,
                    currentTypeTopicCounts,
                    sample);
        } else {
            sample -= topicTermMass;

            if (sample <= topicBetaMass[m]) {
                samplingBucket = "BetaMass";
                //betaTopicCount++;
                newTopic = findNewTopicInBetaMass(
                        localTopicCounts,
                        localTopicIndex,
                        totalMassOtherModalities,
                        nonZeroTopics,
                        m,
                        docLength,
                        sample,
                        p);

            } else {
                //smoothingOnlyCount++;
                //smoothingOnlyMass[i] += alpha[topic] * beta[i] / (tokensPerTopic[i][topic] + betaSum[i]);
                samplingBucket = "SmoothingMass";
                sample -= topicBetaMass[m];
                newTopic = findNewTopicInSmoothingMass(sample, m, docLength);

            }
        }

        if (newTopic == -1 || newTopic > numTopics - 1) {
            System.err.println("WorkerRunnable sampling error for modality: " + m + " in " + samplingBucket + ": Sample:" + origSample + " Smoothing:" + (smoothingOnlyMass[m] / docLength[m]) + " Beta:"
                    + topicBetaMass[m] + " TopicTerm:" + topicTermMass);
            newTopic = oldTopic; //numCommonTopics + (m + 1) * numIndependentTopics - 1; // TODO is this appropriate
            //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
        }

        rearrangeTypeTopicCounts(currentTypeTopicCounts, newTopic);

        return newTopic;
    }

    protected void rearrangeTypeTopicCounts(
            TIntArrayList currentTypeTopicCounts,
            int newTopic) {

        // Move to the position for the new topic,
        //  which may be the first empty position if this
        //  is a new topic for this word.
        int index = 0;
        while (currentTypeTopicCounts.get(index) > 0
                && (currentTypeTopicCounts.get(index) & topicMask) != newTopic) {
            index++;
            if (index == currentTypeTopicCounts.size()) { //TODO: Size is it OK
                System.err.println("error in findind new poisition for topic: " + newTopic);
                for (int k = 0; k < currentTypeTopicCounts.size(); k++) {
                    System.err.print((currentTypeTopicCounts.get(k) & topicMask) + ":"
                            + (currentTypeTopicCounts.get(k) >> topicBits) + " ");
                }
                System.err.println();

            }
        }

        // index should now be set to the position of the new topic,
        //  which may be an empty cell at the end of the list.
        int currentValue;
        if (currentTypeTopicCounts.get(index) == 0) {
            // inserting a new topic, guaranteed to be in
            //  order w.r.t. count, if not topic.
            currentTypeTopicCounts.set(index, (1 << topicBits) + newTopic);
        } else {
            currentValue = currentTypeTopicCounts.get(index) >> topicBits;
            currentTypeTopicCounts.set(index, ((currentValue + 1) << topicBits) + newTopic);

            // Bubble the increased value left, if necessary
            while (index > 0
                    && currentTypeTopicCounts.get(index) > currentTypeTopicCounts.get(index - 1)) {
                int temp = currentTypeTopicCounts.get(index);
                currentTypeTopicCounts.set(index, currentTypeTopicCounts.get(index - 1));
                currentTypeTopicCounts.set(index - 1, temp);

                index--;
            }
        }

    }

    protected int updateTopicCounts(
            int[][] oneDocTopics,
            int position,
            int newTopic,
            TDoubleArrayList cachedCoefficients,
            TDoubleArrayList totalMassOtherModalities,
            TIntArrayList[] localTopicCounts,
            TIntArrayList localTopicIndex,
            double[] topicBetaMass,
            int nonZeroTopics,
            final int[] docLength,
            byte m,
            double[][] p) {

        if (newTopic == numTopics) { //new topic in corpus
            for (byte i = 0; i < numModalities; i++) {

                alpha[i].set(numTopics, alpha[i].get(numTopics - 1));

                if (i != m) { //Update smoothing for all other modalities, current modality will be updated at the end
                    smoothingOnlyMass[i] += gamma[i] * alpha[i].get(newTopic) * beta[i]
                            / (tokensPerTopic[i].get(newTopic) + betaSum[i]);

                    smoothOnlyCachedCoefficients[i].set(newTopic, gamma[i] * alpha[i].get(newTopic) / (tokensPerTopic[i].get(newTopic) + betaSum[i]));
                    //ONLY Global counts should be updated
//                    double normSumN = (localTopicCounts[i].get(newTopic) + totalMassOtherModalities.get(newTopic))
//                            / (tokensPerTopic[i].get(newTopic) + betaSum[i]);
//
//                    topicBetaMass[i] += beta[i] * normSumN;
//                    cachedCoefficients.set(newTopic, normSumN);
                }
            }
            numTopics++;
        }

        //			Put that new topic into the counts
        oneDocTopics[m][position] = newTopic;

        // If this is a new topic for this document,
        //  add the topic to the dense index.
        boolean isNewTopic = (localTopicCounts[m].get(newTopic) == 0);
        byte jj = 0;
        while (isNewTopic && jj < numModalities) {
            //if (jj != m) { // every other topic should have zero counts
            isNewTopic = localTopicCounts[jj].get(newTopic) == 0;
            //}
            jj++;
        }

        if (isNewTopic) {

            // First find the point where we 
            //  should insert the new topic by going to
            //  the end (which is the only reason we're keeping
            //  track of the number of non-zero
            //  topics) and working backwards
            int denseIndex = nonZeroTopics;

            while (denseIndex > 0
                    && localTopicIndex.get(denseIndex - 1) > newTopic) {

                localTopicIndex.set(denseIndex,
                        localTopicIndex.get(denseIndex - 1));
                denseIndex--;
            }

            localTopicIndex.set(denseIndex, newTopic);
            nonZeroTopics++;
        }

        double normSumN = (localTopicCounts[m].get(newTopic) + totalMassOtherModalities.get(newTopic))
                / (tokensPerTopic[m].get(newTopic) + betaSum[m]);

        topicBetaMass[m] -= beta[m] * normSumN;

        smoothingOnlyMass[m] -= gamma[m] * alpha[m].get(newTopic) * beta[m]
                / (tokensPerTopic[m].get(newTopic) + betaSum[m]);

        // }
        localTopicCounts[m].set(newTopic, localTopicCounts[m].get(newTopic) + 1);
        tokensPerTopic[m].set(newTopic, tokensPerTopic[m].get(newTopic) + 1);

        //	update the coefficients for the non-zero topics
        smoothingOnlyMass[m] += gamma[m] * alpha[m].get(newTopic) * beta[m]
                / (tokensPerTopic[m].get(newTopic) + betaSum[m]);

        smoothOnlyCachedCoefficients[m].set(newTopic, gamma[m] * alpha[m].get(newTopic) / (tokensPerTopic[m].get(newTopic) + betaSum[m]));

        normSumN = (localTopicCounts[m].get(newTopic) + totalMassOtherModalities.get(newTopic))
                / (tokensPerTopic[m].get(newTopic) + betaSum[m]);

        topicBetaMass[m] += beta[m] * normSumN;
        cachedCoefficients.set(newTopic, normSumN);

        return nonZeroTopics;
    }

    protected void sampleTopicsForOneDoc(int docCnt) {

        MixTopicModelTopicAssignment doc = data.get(docCnt);

        TDoubleArrayList[] totalMassPerModalityAndTopic = new TDoubleArrayList[numModalities];

        //cachedCoefficients = new double[numModalities][numTopics];// Conservative allocation... [nonZeroTopics + 10]; //we want to avoid dynamic memory allocation , thus we think that we will not have more than ten new  topics in each run
        int[][] oneDocTopics = new int[numModalities][]; //token topics sequence for document
        FeatureSequence[] tokenSequence = new FeatureSequence[numModalities]; //tokens sequence

        int[] docLength = new int[numModalities];
        TIntArrayList[] localTopicCounts = new TIntArrayList[numModalities];
        TIntArrayList localTopicIndex = new TIntArrayList(numTopics); //dense topic index for all modalities
        localTopicIndex.fill(0, numTopics, 0);
        int type, oldTopic, newTopic;
        double[] topicBetaMass = new double[numModalities];

        //TObjectIntHashMap<Long> topicPerPrvTopic = new TObjectIntHashMap<Long>();
        //TObjectIntHashMap<MassValue> similarGroups = new TObjectIntHashMap<Integer>();
        //init modalities correlation
        double[][] p = new double[numModalities][numModalities];

        for (byte i = 0; i < numModalities; i++) {
            // Arrays.fill( p[i], 1);
            for (byte j = i; j < numModalities; j++) {
                double pRand = i == j ? 1.0 : p_a[i][j] == 0 ? 0 : ((double) Math.round(1000 * random.nextBeta(p_a[i][j], p_b[i][j])) / (double) 1000);

                p[i][j] = pRand;
                p[j][i] = pRand;
            }
        }

        //FeatureSequence tokens = (FeatureSequence) document.instance.getData();
        //FeatureSequence topicSequence =  (FeatureSequence) document.topicSequence;/
        int nonZeroTopics = initSampling(
                doc,
                totalMassPerModalityAndTopic,
                oneDocTopics,
                tokenSequence,
                docLength,
                localTopicCounts,
                localTopicIndex);

        //int[] topicTermIndices;
        //int[] topicTermValues;
        //int i;
        //int[] currentTypeTopicCounts;
        //	Iterate over the positions (words) in the document for each modality
        TDoubleArrayList cachedCoefficients = new TDoubleArrayList();
        TDoubleArrayList totalMassOtherModalities = new TDoubleArrayList();

        for (byte m = 0; m < numModalities; m++) // byte m = 0;
        {

            initSamplingForModality(
                    cachedCoefficients,
                    totalMassOtherModalities,
                    totalMassPerModalityAndTopic,
                    docLength,
                    localTopicCounts,
                    localTopicIndex,
                    topicBetaMass,
                    p,
                    m, //active modality
                    nonZeroTopics);

            FeatureSequence tokenSequenceCurMod = tokenSequence[m];
            for (int position = 0; position < docLength[m]; position++) {

                // if (tokenSequenceCurMod != null) { already checked in init sampling --> docLength[m] =0 if is null
                type = tokenSequenceCurMod.getIndexAtPosition(position);

                oldTopic = oneDocTopics[m][position];

                TIntArrayList currentTypeTopicCounts = typeTopicCounts[m][type];
                //int[] currentTypeTopicCounts = new int[typeTopicCounts[m][type].length]; //typeTopicCounts[m][type];
                //System.arraycopy(typeTopicCounts[m][type], 0, currentTypeTopicCounts, 0, typeTopicCounts[m][type].length-1);

                nonZeroTopics = removeOldTopicContribution(
                        cachedCoefficients,
                        totalMassOtherModalities,
                        localTopicCounts,
                        localTopicIndex,
                        topicBetaMass,
                        oldTopic,
                        nonZeroTopics,
                        // docLength,
                        m);
//,
                //                      p);

//                recalcBetaAndCachedCoefficients(
//                        cachedCoefficients,
//                        localTopicCounts,
//                        localTopicIndex,
//                        topicBetaMass,
//                        nonZeroTopics,
//                        docLength,
//                        m,
//                        p);
                TDoubleArrayList topicTermScores = new TDoubleArrayList(numTopics);
                topicTermScores.fill(0, numTopics, 0);
                double termSkew = typeSkewIndexes[m][type];

                double topicTermMass = calcTopicScores(
                        cachedCoefficients,
                        oldTopic,
                        m,
                        topicTermScores,
                        currentTypeTopicCounts,
                        docLength,
                        termSkew);

                //normalize smoothing mass. 
                //ThreadLocalRandom.current().nextDouble()
                assert (smoothingOnlyMass[m] >= 0) : "smoothing Mass " + smoothingOnlyMass[m] + " below 0";
                assert (topicBetaMass[m] >= 0) : "topicBeta Mass " + topicBetaMass[m] + " below 0";
                assert (topicTermMass >= 0) : "topicTerm Mass " + topicTermMass + " below 0";

                double sample = ThreadLocalRandom.current().nextDouble() * (smoothingOnlyMass[m]
                        + topicBetaMass[m] + topicTermMass);

//                if (sample < 0) {
//                    newTopic = 0;
//                }
                //random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);
                //double origSample = sample;
                newTopic = //random.nextInt(numTopics);
                        findNewTopic(
                                localTopicCounts,
                                localTopicIndex,
                                totalMassOtherModalities,
                                topicBetaMass,
                                nonZeroTopics,
                                m,
                                topicTermMass,
                                topicTermScores,
                                currentTypeTopicCounts,
                                docLength,
                                sample,
                                p,
                                oldTopic);

                long tmpPreviousTopics = doc.Assignments[m].prevTopicsSequence[position];
                tmpPreviousTopics = tmpPreviousTopics >> topicBits;
                long newTopicTmp = (long) newTopic << (63 - topicBits); //long is signed
                tmpPreviousTopics += newTopicTmp;
//
                doc.Assignments[m].prevTopicsSequence[position] = tmpPreviousTopics; //doc.Assignments[m].prevTopicsSequence[position] >> topicBits;

                //doc.Assignments[m].prevTopicsSequence[position] = doc.Assignments[m].prevTopicsSequence[position] + newTopicTmp;
                //assert(newTopic != -1);
                nonZeroTopics = updateTopicCounts(
                        oneDocTopics,
                        position,
                        newTopic,
                        cachedCoefficients,
                        totalMassOtherModalities,
                        localTopicCounts,
                        localTopicIndex,
                        topicBetaMass,
                        nonZeroTopics,
                        docLength,
                        m,
                        p);

                //statistics for p optimization
                for (byte i = (byte) (m - 1); i >= 0; i--) {
//                        
//                        if (localTopicCounts[i][newTopic] == 0)
//                        {
//                            System.out.println("Modality not related");
//                            
//                        }

                    pDistr_Mean[m][i][docCnt] += (localTopicCounts[i].get(newTopic) > 0 ? 1.0 : 0d) / (double) docLength[m];
                    pDistr_Mean[i][m][docCnt] = pDistr_Mean[m][i][docCnt];
                    //pDistr_Var[m][i][docCnt]+= localTopicCounts[i][newTopic]/docLength[m];
                }

                //}
            }

            if (shouldSaveState) {
                // Update the document-topic count histogram,
                //  for dirichlet estimation
                docLengthCounts[m][ docLength[m]]++;

                for (int denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
                    int topic = localTopicIndex.get(denseIndex);
                    topicDocCounts[m].get(topic)[localTopicCounts[m].get(topic)]++;

                }
            }

        }

//	Clean up our mess: reset the coefficients to values with only
//	smoothing. The next doc will update its own non-zero topics...
//not needed we have seperate smothOnlyCoefficients
//        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//            int topic = localTopicIndex[denseIndex];
//
//            cachedCoefficients[topic] =
//                    alpha[topic] / (tokensPerTopic[topic] + betaSum);
//        }
    }
//
//    protected void sampleTopicsForOneDoc(MixTopicModelTopicAssignment doc, boolean a /* fast */) {
//
//        final double[][] smoothOnlyCachedCoefficientsLcl = this.smoothOnlyCachedCoefficients;
//        double[][] cachedCoefficients;
//        int[][] oneDocTopics = new int[numModalities][];
//        FeatureSequence[] tokenSequence = new FeatureSequence[numModalities];
//
//        int[] docLength = new int[numModalities];
//        int[][] localTopicCounts = new int[numModalities][numTopics];
//        int[] localTopicIndex = new int[numTopics]; //dense topic index for all modalities
//        int type, oldTopic, newTopic;
//
//        //FeatureSequence tokens = (FeatureSequence) document.instance.getData();
//        //FeatureSequence topicSequence =  (FeatureSequence) document.topicSequence;/
//
//        for (byte i = 0; i < numModalities; i++) {
//            if (doc.Assignments[i] != null) {
//                oneDocTopics[i] = doc.Assignments[i].topicSequence.getFeatures();
//                tokenSequence[i] = ((FeatureSequence) doc.Assignments[i].instance.getData());
//
//                docLength[i] = tokenSequence[i].getLength();
//
//                //		populate topic counts
//                for (int position = 0; position < docLength[i]; position++) {
//                    if (oneDocTopics[i][position] == ParallelTopicModel.UNASSIGNED_TOPIC) {
//                        continue;
//                    }
//                    localTopicCounts[i][oneDocTopics[i][position]]++;
//                }
//            }
//        }
//        // Build an array that densely lists the topics that
//        //  have non-zero counts.
//        int denseIndex = 0;
//        for (int topic = 0; topic < numTopics; topic++) {
//            int i = 0;
//            boolean topicFound = false;
//            while (i < numModalities && !topicFound) {
//                if (localTopicCounts[i][topic] != 0) {
//                    localTopicIndex[denseIndex] = topic;
//                    denseIndex++;
//                    topicFound = true;
//                }
//                i++;
//            }
//        }
//
//        // Record the total number of non-zero topics
//        int nonZeroTopics = denseIndex;
//        cachedCoefficients = new double[numModalities][numTopics];// Conservative allocation... [nonZeroTopics + 10]; //we want to avoid dynamic memory allocation , thus we think that we will not have more than ten new  topics in each run
//        //		Initialize the topic count/beta sampling bucket
//        double[] topicBetaMass = new double[numModalities];
//        Arrays.fill(topicBetaMass, 0);
//        for (byte i = 0; i < numModalities; i++) {
//            Arrays.fill(cachedCoefficients[i], 0);
//
//        }
//
//        //test compile
//        // Initialize cached coefficients and the topic/beta 
//        //  normalizing constant.
//
//        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//            for (byte i = 0; i < numModalities; i++) {
//                int topic = localTopicIndex[denseIndex];
//                //double normSumN = 0;
//                for (byte j = 0; j < numModalities; j++) {
//                    if (docLength[j] > 0) {
//                        //	initialize the normalization constant for the (B * n_{t|d}) term
//                        double normSumN = p[i][j] * localTopicCounts[j][topic] // / docLength[j]
//                                / (tokensPerTopic[i][topic] + betaSum[i]);
//
//                        topicBetaMass[i] += beta[i] * normSumN;
//                        //	update the coefficients for the non-zero topics
//                        cachedCoefficients[i][topic] += normSumN;
//
//                    }
//                }
//
//                if (Double.isNaN(topicBetaMass[i])) {
//                    topicBetaMass[i] = 0;
//                }
//                if (Double.isNaN(cachedCoefficients[i][topic])) {
//                    cachedCoefficients[i][topic] = 0;
//                }
//            }
//        }
//
//        double topicTermMass = 0.0;
//
//        double[] topicTermScores = new double[numTopics];
//        //int[] topicTermIndices;
//        //int[] topicTermValues;
//        int i;
//        double score;
//
//        int[] currentTypeTopicCounts;
//        //	Iterate over the positions (words) in the document for each modality
//        //for (byte m = 0; m < numModalities; m++) 
//        byte m = 0;
//        {
//
//            for (int position = 0; position < docLength[m]; position++) {
//                if (tokenSequence[m] != null) {
//                    type = tokenSequence[m].getIndexAtPosition(position);
//
//                    oldTopic = oneDocTopics[m][position];
//
//                    currentTypeTopicCounts = typeTopicCounts[m][type];
//
//                    if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {
//                        //	Remove this token from all counts. 
//
//                        // Remove this topic's contribution to the 
//                        //  normalizing constants
//                        smoothingOnlyMass[m] -= alpha[oldTopic] * beta[m]
//                                / (tokensPerTopic[m][oldTopic] + betaSum[m]);
//
//                        double tmp = localTopicCounts[m][oldTopic] // / docLength[m]
//                                / (tokensPerTopic[m][oldTopic] + betaSum[m]);
//                        topicBetaMass[m] -= beta[m] * tmp;
//                        cachedCoefficients[m][oldTopic] -= tmp;
//                        // Decrement the local doc/topic counts
//
//                        localTopicCounts[m][oldTopic]--;
//
//                        // Maintain the dense index, if we are deleting
//                        //  the old topic
//                        boolean isDeletedTopic = true;
//                        for (byte j = 0; j < numModalities; j++) {
//
//                            if (localTopicCounts[j][oldTopic] != 0) {
//                                isDeletedTopic = false;
//                            }
//
//                        }
//
//                        if (isDeletedTopic) {
//
//                            // First get to the dense location associated with
//                            //  the old topic.
//
//                            denseIndex = 0;
//
//                            // We know it's in there somewhere, so we don't 
//                            //  need bounds checking.
//                            while (localTopicIndex[denseIndex] != oldTopic) {
//                                denseIndex++;
//                            }
//
//                            // shift all remaining dense indices to the left.
//                            while (denseIndex < nonZeroTopics) {
//                                if (denseIndex < localTopicIndex.length - 1) {
//                                    localTopicIndex[denseIndex] =
//                                            localTopicIndex[denseIndex + 1];
//                                }
//                                denseIndex++;
//                            }
//
//                            nonZeroTopics--;
//                        }
//
//                        // Decrement the global topic count totals
//                        tokensPerTopic[m][oldTopic]--;
//                        assert (tokensPerTopic[m][oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
//
//
//                        // Add the old topic's contribution back into the
//                        //  normalizing constants.
//                        smoothingOnlyMass[m] += alpha[oldTopic] * beta[m]
//                                / (tokensPerTopic[m][oldTopic] + betaSum[m]);
//
//                        topicBetaMass[m] += beta[m] * localTopicCounts[m][oldTopic] /// docLength[m]
//                                / (tokensPerTopic[m][oldTopic] + betaSum[m]);
//
//                        // Reset the cached coefficient for this topic
//                        cachedCoefficients[m][oldTopic] += localTopicCounts[m][oldTopic]// / docLength[m]
//                                / (tokensPerTopic[m][oldTopic] + betaSum[m]);
//
//                    }
//
//
//                    // Now go over the type/topic counts, decrementing
//                    //  where appropriate, and calculating the score
//                    //  for each topic at the same time.
//
//                    int index = 0;
//                    int currentTopic, currentValue;
//
//                    boolean alreadyDecremented = (oldTopic == ParallelTopicModel.UNASSIGNED_TOPIC);
//
//                    topicTermMass = 0.0;
//
//                    while (index < currentTypeTopicCounts.length
//                            && currentTypeTopicCounts[index] > 0) {
//
//                        currentTopic = currentTypeTopicCounts[index] & topicMask;
//                        currentValue = currentTypeTopicCounts[index] >> topicBits;
//
//                        if (!alreadyDecremented
//                                && currentTopic == oldTopic) {
//
//                            // We're decrementing and adding up the 
//                            //  sampling weights at the same time, but
//                            //  decrementing may require us to reorder
//                            //  the topics, so after we're done here,
//                            //  look at this cell in the array again.
//
//                            currentValue--;
//                            if (currentValue == 0) {
//                                currentTypeTopicCounts[index] = 0;
//                            } else {
//                                currentTypeTopicCounts[index] =
//                                        (currentValue << topicBits) + oldTopic;
//                            }
//
//                            // Shift the reduced value to the right, if necessary.
//
//                            int subIndex = index;
//                            while (subIndex < currentTypeTopicCounts.length - 1
//                                    && currentTypeTopicCounts[subIndex] < currentTypeTopicCounts[subIndex + 1]) {
//                                int temp = currentTypeTopicCounts[subIndex];
//                                currentTypeTopicCounts[subIndex] = currentTypeTopicCounts[subIndex + 1];
//                                currentTypeTopicCounts[subIndex + 1] = temp;
//
//                                subIndex++;
//                            }
//
//                            alreadyDecremented = true;
//                        } else {
//                            //add normalized smoothingOnly coefficient 
//                            score =
//                                    (cachedCoefficients[m][currentTopic] + smoothOnlyCachedCoefficientsLcl[m][currentTopic] /// docLength[m])
//                                    ) * currentValue;
//                            topicTermMass += score;
//                            topicTermScores[index] = score;
//
//                            index++;
//                        }
//                    }
//
//
//                    //normalize smoothing mass. 
//                    //ThreadLocalRandom.current().nextDouble()
//                    double sample = random.nextUniform() * (smoothingOnlyMass[m] //  / docLength[m] 
//                            + topicBetaMass[m] + topicTermMass);
//                    //random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);
//
//                    double origSample = sample;
//
//                    //	Make sure it actually gets set
//                    newTopic = -1;
//
//                    if (sample < topicTermMass) {
//                        //topicTermCount++;
//
//                        i = -1;
//                        while (sample > 0) {
//                            i++;
//                            sample -= topicTermScores[i];
//                        }
//
//                        newTopic = currentTypeTopicCounts[i] & topicMask;
//                        currentValue = currentTypeTopicCounts[i] >> topicBits;
//
//                        currentTypeTopicCounts[i] = ((currentValue + 1) << topicBits) + newTopic;
//
//                        // Bubble the new value up, if necessary
//
//                        while (i > 0
//                                && currentTypeTopicCounts[i] > currentTypeTopicCounts[i - 1]) {
//                            int temp = currentTypeTopicCounts[i];
//                            currentTypeTopicCounts[i] = currentTypeTopicCounts[i - 1];
//                            currentTypeTopicCounts[i - 1] = temp;
//
//                            i--;
//                        }
//
//                    } else {
//                        sample -= topicTermMass;
//
//                        if (sample < topicBetaMass[m]) {
//                            //betaTopicCount++;
//
//                            //sample /= beta[m];
//
//
//                            for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//
//                                int topic = localTopicIndex[denseIndex];
//
//                                for (byte j = 0; j < numModalities; j++) {
//                                    if (docLength[j] > 0) {
//                                        double normSumN = p[m][j] * localTopicCounts[j][topic] /// docLength[j]
//                                                / (tokensPerTopic[m][topic] + betaSum[m]);
//
//                                        sample -= beta[m] * normSumN;
//                                        //sample -= p[m][j] * beta[j] * localTopicCounts[j][topic] / docLength[j]
//                                        //        / (tokensPerTopic[m][topic] + betaSum[m]);
//
//                                        //normSumN += p[m][j] * localTopicCounts[j][topic] / docLength[j];
//                                    }
//                                }
////                                sample -= normSumN
////                                        / (tokensPerTopic[m][topic] + betaSum[m]);
//
//                                if (sample <= 0) {
//                                    newTopic = topic;
//                                    break;
//                                }
//                            }
//                            if (sample > 0) {
//                                newTopic = -1;
//                            }
//
//
//                        } else {
//                            //smoothingOnlyCount++;
//                            //smoothingOnlyMass[i] += alpha[topic] * beta[i] / (tokensPerTopic[i][topic] + betaSum[i]);
//
//                            sample -= topicBetaMass[m];
//
//                            // sample *= docLength[m];
//                            sample /= beta[m];
//
//                            int topic = 0;
//                            sample -= alpha[topic]
//                                    / (tokensPerTopic[m][topic] + betaSum[m]);
//
//                            while (sample > 0.0 && topic < numTopics - 1) {
//                                topic++;
//                                sample -= alpha[topic]
//                                        / (tokensPerTopic[m][topic] + betaSum[m]);
//                            }
//
//                            if (sample <= 0.0) {
//                                newTopic = topic;
//                                // break;
//                            } else {
//                                newTopic = -1;
//                            }
//
//                        }
//
//                        // Move to the position for the new topic,
//                        //  which may be the first empty position if this
//                        //  is a new topic for this word.
//
//                        index = 0;
//                        while (currentTypeTopicCounts[index] > 0
//                                && (currentTypeTopicCounts[index] & topicMask) != newTopic) {
//                            index++;
//                            if (index == currentTypeTopicCounts.length) {
//                                System.err.println("type: " + type + " new topic: " + newTopic);
//                                for (int k = 0; k < currentTypeTopicCounts.length; k++) {
//                                    System.err.print((currentTypeTopicCounts[k] & topicMask) + ":"
//                                            + (currentTypeTopicCounts[k] >> topicBits) + " ");
//                                }
//                                System.err.println();
//
//                            }
//                        }
//
//
//                        // index should now be set to the position of the new topic,
//                        //  which may be an empty cell at the end of the list.
//
//                        if (currentTypeTopicCounts[index] == 0) {
//                            // inserting a new topic, guaranteed to be in
//                            //  order w.r.t. count, if not topic.
//                            currentTypeTopicCounts[index] = (1 << topicBits) + newTopic;
//                        } else {
//                            currentValue = currentTypeTopicCounts[index] >> topicBits;
//                            currentTypeTopicCounts[index] = ((currentValue + 1) << topicBits) + newTopic;
//
//                            // Bubble the increased value left, if necessary
//                            while (index > 0
//                                    && currentTypeTopicCounts[index] > currentTypeTopicCounts[index - 1]) {
//                                int temp = currentTypeTopicCounts[index];
//                                currentTypeTopicCounts[index] = currentTypeTopicCounts[index - 1];
//                                currentTypeTopicCounts[index - 1] = temp;
//
//                                index--;
//                            }
//                        }
//
//                    }
//
//                    if (newTopic == -1) {
//                        System.err.println("WorkerRunnable sampling error: " + origSample + " " + sample + " " + smoothingOnlyMass[m] + " "
//                                + topicBetaMass[m] + " " + topicTermMass);
//                        newTopic = numTopics - 1; // TODO is this appropriate
//                        //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
//                    }
//                    //assert(newTopic != -1);
//
//                    //			Put that new topic into the counts
//                    oneDocTopics[m][position] = newTopic;
//
//                    smoothingOnlyMass[m] -= alpha[newTopic] * beta[m]
//                            / (tokensPerTopic[m][newTopic] + betaSum[m]);
//
//                    topicBetaMass[m] -= beta[m] * localTopicCounts[m][newTopic]// / docLength[m]
//                            / (tokensPerTopic[m][newTopic] + betaSum[m]);
//
//
//                    cachedCoefficients[m][newTopic] -= localTopicCounts[m][newTopic]// / docLength[m]
//                            / (tokensPerTopic[m][newTopic] + betaSum[m]);
//
//                    localTopicCounts[m][newTopic]++;
//
//                    // If this is a new topic for this document,
//                    //  add the topic to the dense index.
//
//                    boolean isNewTopic = false;
//                    if (localTopicCounts[m][newTopic] == 1) {
//                        isNewTopic = true;
//                        for (byte j = 0; j < numModalities; j++) {
//                            if (j != m) {
//                                if (localTopicCounts[j][newTopic] != 0) {
//                                    isNewTopic = false;
//                                }
//                            }
//                        }
//                    }
//
//
//                    if (isNewTopic) {
//
//                        // First find the point where we 
//                        //  should insert the new topic by going to
//                        //  the end (which is the only reason we're keeping
//                        //  track of the number of non-zero
//                        //  topics) and working backwards
//
//                        denseIndex = nonZeroTopics;
//
//                        while (denseIndex > 0
//                                && localTopicIndex[denseIndex - 1] > newTopic) {
//
//                            localTopicIndex[denseIndex] =
//                                    localTopicIndex[denseIndex - 1];
//                            denseIndex--;
//                        }
//
//                        localTopicIndex[denseIndex] = newTopic;
//                        nonZeroTopics++;
//                    }
//
//                    tokensPerTopic[m][newTopic]++;
//
//                    //	update the coefficients for the non-zero topics
//                    smoothingOnlyMass[m] += alpha[newTopic] * beta[m]
//                            / (tokensPerTopic[m][newTopic] + betaSum[m]);
//
//                    topicBetaMass[m] += beta[m] * localTopicCounts[m][newTopic]// / docLength[m]
//                            / (tokensPerTopic[m][newTopic] + betaSum[m]);
//
//
//                    cachedCoefficients[m][newTopic] += localTopicCounts[m][newTopic]// / docLength[m]
//                            / (tokensPerTopic[m][newTopic] + betaSum[m]);
//
//                }
//
//            }
//            if (shouldSaveState) {
//                // Update the document-topic count histogram,
//                //  for dirichlet estimation
//                docLengthCounts[ docLength[m]]++;
//
//                for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//                    int topic = localTopicIndex[denseIndex];
//                    topicDocCounts[topic][ localTopicCounts[m][topic]]++;
//                }
//            }
//        }
////	Clean up our mess: reset the coefficients to values with only
////	smoothing. The next doc will update its own non-zero topics...
////not needed we have seperate smothOnlyCoefficients
////        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
////            int topic = localTopicIndex[denseIndex];
////
////            cachedCoefficients[topic] =
////                    alpha[topic] / (tokensPerTopic[topic] + betaSum);
////        }
//    }
//    protected void sampleTopicsForOneDoc(MixTopicModelTopicAssignment doc, String a /* not fast any more */) {
//
//        final double[][] smoothOnlyCachedCoefficientsLcl = this.smoothOnlyCachedCoefficients;
//        double[][] cachedCoefficients;
//        int[][] oneDocTopics = new int[numModalities][];
//        FeatureSequence[] tokenSequence = new FeatureSequence[numModalities];
//
//        int[] docLength = new int[numModalities];
//        int[][] localTopicCounts = new int[numModalities][numTopics];
//        int[] localTopicIndex = new int[numTopics]; //dense topic index for all modalities
//        int type, oldTopic, newTopic;
//
//        //FeatureSequence tokens = (FeatureSequence) document.instance.getData();
//        //FeatureSequence topicSequence =  (FeatureSequence) document.topicSequence;/
//
//        for (byte i = 0; i < numModalities; i++) {
//            if (doc.Assignments[i] != null) {
//                oneDocTopics[i] = doc.Assignments[i].topicSequence.getFeatures();
//                tokenSequence[i] = ((FeatureSequence) doc.Assignments[i].instance.getData());
//
//                docLength[i] = tokenSequence[i].getLength();
//
//                //		populate topic counts
//                for (int position = 0; position < docLength[i]; position++) {
//                    if (oneDocTopics[i][position] == ParallelTopicModel.UNASSIGNED_TOPIC) {
//                        continue;
//                    }
//                    localTopicCounts[i][oneDocTopics[i][position]]++;
//                }
//            }
//        }
//        // Build an array that densely lists the topics that
//        //  have non-zero counts.
//        int denseIndex = 0;
//        for (int topic = 0; topic < numTopics; topic++) {
//            int i = 0;
//            boolean topicFound = false;
//            while (i < numModalities && !topicFound) {
//                if (localTopicCounts[i][topic] != 0) {
//                    localTopicIndex[denseIndex] = topic;
//                    denseIndex++;
//                    topicFound = true;
//                }
//                i++;
//            }
//        }
//
//        // Record the total number of non-zero topics
//        int nonZeroTopics = denseIndex;
//        cachedCoefficients = new double[numModalities][numTopics];// Conservative allocation... [nonZeroTopics + 10]; //we want to avoid dynamic memory allocation , thus we think that we will not have more than ten new  topics in each run
//        //		Initialize the topic count/beta sampling bucket
//        double[] topicBetaMass = new double[numModalities];
//        Arrays.fill(topicBetaMass, 0);
//        for (byte i = 0; i < numModalities; i++) {
//            Arrays.fill(cachedCoefficients[i], 0);
//
//        }
//
//        //test compile
//        // Initialize cached coefficients and the topic/beta 
//        //  normalizing constant.
//
//        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//            for (byte i = 0; i < numModalities; i++) {
//                int topic = localTopicIndex[denseIndex];
//                //double normSumN = 0;
//                for (byte j = 0; j < numModalities; j++) {
//                    if (docLength[j] > 0) {
//                        //	initialize the normalization constant for the (B * n_{t|d}) term
//                        double normSumN = p[i][j] * localTopicCounts[j][topic] // / docLength[j]
//                                / (docLength[j] * (tokensPerTopic[i][topic] + betaSum[i]));
//
//                        topicBetaMass[i] += beta[i] * normSumN;
//                        //	update the coefficients for the non-zero topics
//                        cachedCoefficients[i][topic] += normSumN;
//
//                    }
//                }
//
//                if (Double.isNaN(topicBetaMass[i])) {
//                    topicBetaMass[i] = 0;
//                }
//                if (Double.isNaN(cachedCoefficients[i][topic])) {
//                    cachedCoefficients[i][topic] = 0;
//                }
//            }
//        }
//
//        double topicTermMass = 0.0;
//
//        double[] topicTermScores = new double[numTopics];
//        //int[] topicTermIndices;
//        //int[] topicTermValues;
//        int i;
//        double score;
//
//        int[] currentTypeTopicCounts;
//        //	Iterate over the positions (words) in the document for each modality
//        //for (byte m = 0; m < numModalities; m++) 
//        byte m = 0;
//        {
//
//            for (int position = 0; position < docLength[m]; position++) {
//                if (tokenSequence[m] != null) {
//                    type = tokenSequence[m].getIndexAtPosition(position);
//
//                    oldTopic = oneDocTopics[m][position];
//
//                    currentTypeTopicCounts = typeTopicCounts[m][type];
//
//                    if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {
//                        //	Remove this token from all counts. 
//
//                        // Remove this topic's contribution to the 
//                        //  normalizing constants
//                        smoothingOnlyMass[m] -= alpha[oldTopic] * beta[m]
//                                / (tokensPerTopic[m][oldTopic] + betaSum[m]);
//
//                        double tmp = localTopicCounts[m][oldTopic] // / docLength[m]
//                                / (docLength[m] * (tokensPerTopic[m][oldTopic] + betaSum[m]));
//                        topicBetaMass[m] -= beta[m] * tmp;
//                        cachedCoefficients[m][oldTopic] -= tmp;
//                        // Decrement the local doc/topic counts
//
//                        localTopicCounts[m][oldTopic]--;
//
//                        // Maintain the dense index, if we are deleting
//                        //  the old topic
//                        boolean isDeletedTopic = true;
//                        for (byte j = 0; j < numModalities; j++) {
//
//                            if (localTopicCounts[j][oldTopic] != 0) {
//                                isDeletedTopic = false;
//                            }
//
//                        }
//
//                        if (isDeletedTopic) {
//
//                            // First get to the dense location associated with
//                            //  the old topic.
//
//                            denseIndex = 0;
//
//                            // We know it's in there somewhere, so we don't 
//                            //  need bounds checking.
//                            while (localTopicIndex[denseIndex] != oldTopic) {
//                                denseIndex++;
//                            }
//
//                            // shift all remaining dense indices to the left.
//                            while (denseIndex < nonZeroTopics) {
//                                if (denseIndex < localTopicIndex.length - 1) {
//                                    localTopicIndex[denseIndex] =
//                                            localTopicIndex[denseIndex + 1];
//                                }
//                                denseIndex++;
//                            }
//
//                            nonZeroTopics--;
//                        }
//
//                        // Decrement the global topic count totals
//                        tokensPerTopic[m][oldTopic]--;
//                        assert (tokensPerTopic[m][oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
//
//
//                        // Add the old topic's contribution back into the
//                        //  normalizing constants.
//                        smoothingOnlyMass[m] += alpha[oldTopic] * beta[m]
//                                / (tokensPerTopic[m][oldTopic] + betaSum[m]);
//
//                        topicBetaMass[m] += beta[m] * localTopicCounts[m][oldTopic] /// docLength[m]
//                                / (docLength[m] * (tokensPerTopic[m][oldTopic] + betaSum[m]));
//
//                        // Reset the cached coefficient for this topic
//                        cachedCoefficients[m][oldTopic] += localTopicCounts[m][oldTopic]// / docLength[m]
//                                / (docLength[m] * (tokensPerTopic[m][oldTopic] + betaSum[m]));
//
//                    }
//
//
//                    // Now go over the type/topic counts, decrementing
//                    //  where appropriate, and calculating the score
//                    //  for each topic at the same time.
//
//                    int index = 0;
//                    int currentTopic, currentValue;
//
//                    boolean alreadyDecremented = (oldTopic == ParallelTopicModel.UNASSIGNED_TOPIC);
//
//                    topicTermMass = 0.0;
//
//                    while (index < currentTypeTopicCounts.length
//                            && currentTypeTopicCounts[index] > 0) {
//
//                        currentTopic = currentTypeTopicCounts[index] & topicMask;
//                        currentValue = currentTypeTopicCounts[index] >> topicBits;
//
//                        if (!alreadyDecremented
//                                && currentTopic == oldTopic) {
//
//                            // We're decrementing and adding up the 
//                            //  sampling weights at the same time, but
//                            //  decrementing may require us to reorder
//                            //  the topics, so after we're done here,
//                            //  look at this cell in the array again.
//
//                            currentValue--;
//                            if (currentValue == 0) {
//                                currentTypeTopicCounts[index] = 0;
//                            } else {
//                                currentTypeTopicCounts[index] =
//                                        (currentValue << topicBits) + oldTopic;
//                            }
//
//                            // Shift the reduced value to the right, if necessary.
//
//                            int subIndex = index;
//                            while (subIndex < currentTypeTopicCounts.length - 1
//                                    && currentTypeTopicCounts[subIndex] < currentTypeTopicCounts[subIndex + 1]) {
//                                int temp = currentTypeTopicCounts[subIndex];
//                                currentTypeTopicCounts[subIndex] = currentTypeTopicCounts[subIndex + 1];
//                                currentTypeTopicCounts[subIndex + 1] = temp;
//
//                                subIndex++;
//                            }
//
//                            alreadyDecremented = true;
//                        } else {
//                            //add normalized smoothingOnly coefficient 
//                            score =
//                                    (cachedCoefficients[m][currentTopic] + (smoothOnlyCachedCoefficientsLcl[m][currentTopic] / docLength[m])) * currentValue;
//                            topicTermMass += score;
//                            topicTermScores[index] = score;
//
//                            index++;
//                        }
//                    }
//
//
//                    //normalize smoothing mass. 
//                    //ThreadLocalRandom.current().nextDouble()
//                    double sample = random.nextUniform() * ((smoothingOnlyMass[m] / docLength[m])
//                            + topicBetaMass[m] + topicTermMass);
//                    //random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);
//
//                    double origSample = sample;
//
//                    //	Make sure it actually gets set
//                    newTopic = -1;
//
//                    if (sample < topicTermMass) {
//                        //topicTermCount++;
//
//                        i = -1;
//                        while (sample > 0) {
//                            i++;
//                            sample -= topicTermScores[i];
//                        }
//
//                        newTopic = currentTypeTopicCounts[i] & topicMask;
//                        currentValue = currentTypeTopicCounts[i] >> topicBits;
//
//                        currentTypeTopicCounts[i] = ((currentValue + 1) << topicBits) + newTopic;
//
//                        // Bubble the new value up, if necessary
//
//                        while (i > 0
//                                && currentTypeTopicCounts[i] > currentTypeTopicCounts[i - 1]) {
//                            int temp = currentTypeTopicCounts[i];
//                            currentTypeTopicCounts[i] = currentTypeTopicCounts[i - 1];
//                            currentTypeTopicCounts[i - 1] = temp;
//
//                            i--;
//                        }
//
//                    } else {
//                        sample -= topicTermMass;
//
//                        if (sample < topicBetaMass[m]) {
//                            //betaTopicCount++;
//
//                            sample /= beta[m];
//
//                            int topic = -1;
//                            for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//
//                                topic = localTopicIndex[denseIndex];
//
//                                for (byte j = 0; j < numModalities; j++) {
//                                    if (docLength[j] > 0) {
//                                        double normSumN = p[m][j] * localTopicCounts[j][topic] /// docLength[j]
//                                                / (docLength[j] * (tokensPerTopic[m][topic] + betaSum[m]));
//
//                                        sample -= normSumN;
//                                        //sample -= p[m][j] * beta[j] * localTopicCounts[j][topic] / docLength[j]
//                                        //        / (tokensPerTopic[m][topic] + betaSum[m]);
//
//                                        //normSumN += p[m][j] * localTopicCounts[j][topic] / docLength[j];
//                                    }
//                                }
////                                sample -= normSumN
////                                        / (tokensPerTopic[m][topic] + betaSum[m]);
//
//                                if (sample <= 0) {
//                                    newTopic = topic;
//                                    break;
//                                }
//                            }
//                            if (sample > 0) {
//                                newTopic = topic; // rounding error sometimes TODO: find a solution
//                                //newTopic = -1;
//                            }
//
//
//                        } else {
//                            //smoothingOnlyCount++;
//                            //smoothingOnlyMass[i] += alpha[topic] * beta[i] / (tokensPerTopic[i][topic] + betaSum[i]);
//
//                            sample -= topicBetaMass[m];
//
//                            sample *= docLength[m];
//                            sample /= beta[m];
//
//                            int topic = 0;
//                            sample -= alpha[topic]
//                                    / (tokensPerTopic[m][topic] + betaSum[m]);
//
//                            while (sample > 0.0 && topic < numTopics - 1) {
//                                topic++;
//                                sample -= alpha[topic]
//                                        / (tokensPerTopic[m][topic] + betaSum[m]);
//                            }
//
//                            if (sample <= 0.0) {
//                                newTopic = topic;
//                                // break;
//                            } else {
//                                newTopic = -1;
//                            }
//
//                        }
//
//                        // Move to the position for the new topic,
//                        //  which may be the first empty position if this
//                        //  is a new topic for this word.
//
//                        index = 0;
//                        while (currentTypeTopicCounts[index] > 0
//                                && (currentTypeTopicCounts[index] & topicMask) != newTopic) {
//                            index++;
//                            if (index == currentTypeTopicCounts.length) {
//                                System.err.println("type: " + type + " new topic: " + newTopic);
//                                for (int k = 0; k < currentTypeTopicCounts.length; k++) {
//                                    System.err.print((currentTypeTopicCounts[k] & topicMask) + ":"
//                                            + (currentTypeTopicCounts[k] >> topicBits) + " ");
//                                }
//                                System.err.println();
//
//                            }
//                        }
//
//
//                        // index should now be set to the position of the new topic,
//                        //  which may be an empty cell at the end of the list.
//
//                        if (currentTypeTopicCounts[index] == 0) {
//                            // inserting a new topic, guaranteed to be in
//                            //  order w.r.t. count, if not topic.
//                            currentTypeTopicCounts[index] = (1 << topicBits) + newTopic;
//                        } else {
//                            currentValue = currentTypeTopicCounts[index] >> topicBits;
//                            currentTypeTopicCounts[index] = ((currentValue + 1) << topicBits) + newTopic;
//
//                            // Bubble the increased value left, if necessary
//                            while (index > 0
//                                    && currentTypeTopicCounts[index] > currentTypeTopicCounts[index - 1]) {
//                                int temp = currentTypeTopicCounts[index];
//                                currentTypeTopicCounts[index] = currentTypeTopicCounts[index - 1];
//                                currentTypeTopicCounts[index - 1] = temp;
//
//                                index--;
//                            }
//                        }
//
//                    }
//
//                    if (newTopic == -1) {
//                        System.err.println("WorkerRunnable sampling error: " + origSample + " " + sample + " " + smoothingOnlyMass[m] + " "
//                                + topicBetaMass[m] + " " + topicTermMass);
//                        newTopic = numTopics - 1; // TODO is this appropriate
//                        //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
//                    }
//                    //assert(newTopic != -1);
//
//                    //			Put that new topic into the counts
//                    oneDocTopics[m][position] = newTopic;
//
//                    smoothingOnlyMass[m] -= alpha[newTopic] * beta[m]
//                            / (tokensPerTopic[m][newTopic] + betaSum[m]);
//
//                    topicBetaMass[m] -= beta[m] * localTopicCounts[m][newTopic]// / docLength[m]
//                            / (docLength[m] * (tokensPerTopic[m][newTopic] + betaSum[m]));
//
//
//                    cachedCoefficients[m][newTopic] -= localTopicCounts[m][newTopic]// / docLength[m]
//                            / (docLength[m] * (tokensPerTopic[m][newTopic] + betaSum[m]));
//
//                    localTopicCounts[m][newTopic]++;
//
//                    // If this is a new topic for this document,
//                    //  add the topic to the dense index.
//
//                    boolean isNewTopic = false;
//                    if (localTopicCounts[m][newTopic] == 1) {
//                        isNewTopic = true;
//                        for (byte j = 0; j < numModalities; j++) {
//                            if (j != m) {
//                                if (localTopicCounts[j][newTopic] != 0) {
//                                    isNewTopic = false;
//                                }
//                            }
//                        }
//                    }
//
//
//                    if (isNewTopic) {
//
//                        // First find the point where we 
//                        //  should insert the new topic by going to
//                        //  the end (which is the only reason we're keeping
//                        //  track of the number of non-zero
//                        //  topics) and working backwards
//
//                        denseIndex = nonZeroTopics;
//
//                        while (denseIndex > 0
//                                && localTopicIndex[denseIndex - 1] > newTopic) {
//
//                            localTopicIndex[denseIndex] =
//                                    localTopicIndex[denseIndex - 1];
//                            denseIndex--;
//                        }
//
//                        localTopicIndex[denseIndex] = newTopic;
//                        nonZeroTopics++;
//                    }
//
//                    tokensPerTopic[m][newTopic]++;
//
//                    //	update the coefficients for the non-zero topics
//                    smoothingOnlyMass[m] += alpha[newTopic] * beta[m]
//                            / (tokensPerTopic[m][newTopic] + betaSum[m]);
//
//                    topicBetaMass[m] += beta[m] * localTopicCounts[m][newTopic]// / docLength[m]
//                            / (docLength[m] * (tokensPerTopic[m][newTopic] + betaSum[m]));
//
//
//                    cachedCoefficients[m][newTopic] += localTopicCounts[m][newTopic]// / docLength[m]
//                            / (docLength[m] * (tokensPerTopic[m][newTopic] + betaSum[m]));
//
//                }
//
//            }
//            if (shouldSaveState) {
//                // Update the document-topic count histogram,
//                //  for dirichlet estimation
//                docLengthCounts[ docLength[m]]++;
//
//                for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//                    int topic = localTopicIndex[denseIndex];
//                    topicDocCounts[topic][ localTopicCounts[m][topic]]++;
//                }
//            }
//        }
////	Clean up our mess: reset the coefficients to values with only
////	smoothing. The next doc will update its own non-zero topics...
////not needed we have seperate smothOnlyCoefficients
////        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
////            int topic = localTopicIndex[denseIndex];
////
////            cachedCoefficients[topic] =
////                    alpha[topic] / (tokensPerTopic[topic] + betaSum);
////        }
//    }
////
//    protected void sampleTopicsForOneDoc(FeatureSequence tokenSequence,
//            FeatureSequence topicSequence) {
//
//        //double[][] cachedCoefficients;
//        int[] oneDocTopics = topicSequence.getFeatures();
//        //FeatureSequence[] tokenSequence = new FeatureSequence[numModalities];
//
//        //int docLength = new int[numModalities];
//        int[] localTopicCounts = new int[numTopics];
//        int[] localTopicIndex = new int[numTopics]; //dense topic index for all modalities
//        int type, oldTopic, newTopic;
//
//        //FeatureSequence tokens = (FeatureSequence) document.instance.getData();
//        //FeatureSequence topicSequence =  (FeatureSequence) document.topicSequence;/
//
//        //for (byte i = 0; i < numModalities; i++) {
//        //if (doc.Assignments[i] != null) {
//        //oneDocTopics = topicSequence.getFeatures();
//        //tokenSequence[i] = ((FeatureSequence) doc.Assignments[i].instance.getData());
//
//        int docLength = tokenSequence.getLength();
//
//        //		populate topic counts
//        for (int position = 0; position < docLength; position++) {
//            if (oneDocTopics[position] == ParallelTopicModel.UNASSIGNED_TOPIC) {
//                continue;
//            }
//            localTopicCounts[oneDocTopics[position]]++;
//        }
//        // }
//        //}
//        // Build an array that densely lists the topics that
//        //  have non-zero counts.
//        int denseIndex = 0;
//        for (int topic = 0; topic < numTopics; topic++) {
//            if (localTopicCounts[topic] != 0) {
//                localTopicIndex[denseIndex] = topic;
//                denseIndex++;
//            }
//        }
//
//        // Record the total number of non-zero topics
//        int nonZeroTopics = denseIndex;
//        // cachedCoefficients = new double[numModalities][numTopics];// Conservative allocation... [nonZeroTopics + 10]; //we want to avoid dynamic memory allocation , thus we think that we will not have more than ten new  topics in each run
//        //		Initialize the topic count/beta sampling bucket
//        double topicBetaMass = 0.0;//new double[numModalities];
//        //Arrays.fill(topicBetaMass, 0);
//        // for (byte i = 0; i < numModalities; i++) {
//        // Arrays.fill(cachedCoefficients[m], 0);
//
//        //}
//        // Initialize cached coefficients and the topic/beta 
//        //  normalizing constant.
//
//        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//            int topic = localTopicIndex[denseIndex];
//            int n = localTopicCounts[topic];
//
//            //	initialize the normalization constant for the (B * n_{t|d}) term
//            topicBetaMass += beta[0] * n / (tokensPerTopic[0][topic] + betaSum[0]);
//
//            //	update the coefficients for the non-zero topics
//            //cachedCoefficients[topic] =	(alpha[topic] + n) / (tokensPerTopic[topic] + betaSum);
//        }
////
////                if (Double.isNaN(topicBetaMass[i])) {
////                    topicBetaMass[i] = 0;
////                }
////                if (Double.isNaN(cachedCoefficients[i][topic])) {
////                    cachedCoefficients[i][topic] = 0;
////                }
//
//
//
//        double topicTermMass = 0.0;
//
//        double[] topicTermScores = new double[numTopics];
//        //int[] topicTermIndices;
//        //int[] topicTermValues;
//        int i;
//        double score;
//
//        int[] currentTypeTopicCounts;
//        //	Iterate over the positions (words) in the document for each modality
//        //for (byte m = 0; m < numModalities; m++) 
//
//        for (int position = 0; position < docLength; position++) {
//            //if (tokenSequence[0] != null) {
//            type = tokenSequence.getIndexAtPosition(position);
//
//            oldTopic = oneDocTopics[position];
//
//            currentTypeTopicCounts = typeTopicCounts[0][type];
//
//            if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {
//                //	Remove this token from all counts. 
//
//                // Remove this topic's contribution to the 
//                //  normalizing constants
//                smoothingOnlyMass[0] -= alpha[oldTopic] * beta[0]
//                        / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//
//                double tmp = localTopicCounts[oldTopic] /// docLength[0]
//                        / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//                topicBetaMass -= beta[0] * tmp;
//                // cachedCoefficients[0][oldTopic] -= tmp;
//                // Decrement the local doc/topic counts
//
//                localTopicCounts[oldTopic]--;
//
//                // Maintain the dense index, if we are deleting
//                //  the old topic
////                        boolean isDeletedTopic = true;
////                        for (byte j = 0; j < numModalities; j++) {
////
////                            if (localTopicCounts[oldTopic] != 0) {
////                                isDeletedTopic = false;
////                            }
////
////                        }
//
//                if (localTopicCounts[oldTopic] == 0) {
//
//                    // First get to the dense location associated with
//                    //  the old topic.
//
//                    denseIndex = 0;
//
//                    // We know it's in there somewhere, so we don't 
//                    //  need bounds checking.
//                    while (localTopicIndex[denseIndex] != oldTopic) {
//                        denseIndex++;
//                    }
//
//                    // shift all remaining dense indices to the left.
//                    while (denseIndex < nonZeroTopics) {
//                        if (denseIndex < localTopicIndex.length - 1) {
//                            localTopicIndex[denseIndex] =
//                                    localTopicIndex[denseIndex + 1];
//                        }
//                        denseIndex++;
//                    }
//
//                    nonZeroTopics--;
//                }
//
//                // Decrement the global topic count totals
//                tokensPerTopic[0][oldTopic]--;
//                assert (tokensPerTopic[0][oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
//
//
//                // Add the old topic's contribution back into the
//                //  normalizing constants.
//                smoothingOnlyMass[0] += alpha[oldTopic] * beta[0]
//                        / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//
//                topicBetaMass += beta[0] * localTopicCounts[oldTopic] /// docLength[0]
//                        / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//
//                // Reset the cached coefficient for this topic
//                // cachedCoefficients[0][oldTopic] += localTopicCounts[oldTopic] /// docLength[0]
//                //         / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//
//            }
//
//
//            // Now go over the type/topic counts, decrementing
//            //  where appropriate, and calculating the score
//            //  for each topic at the same time.
//
//            int index = 0;
//            int currentTopic, currentValue;
//
//            boolean alreadyDecremented = (oldTopic == ParallelTopicModel.UNASSIGNED_TOPIC);
//
//            topicTermMass = 0.0;
//
//            while (index < currentTypeTopicCounts.length
//                    && currentTypeTopicCounts[index] > 0) {
//
//                currentTopic = currentTypeTopicCounts[index] & topicMask;
//                currentValue = currentTypeTopicCounts[index] >> topicBits;
//
//                if (!alreadyDecremented
//                        && currentTopic == oldTopic) {
//
//                    // We're decrementing and adding up the 
//                    //  sampling weights at the same time, but
//                    //  decrementing may require us to reorder
//                    //  the topics, so after we're done here,
//                    //  look at this cell in the array again.
//
//                    currentValue--;
//                    if (currentValue == 0) {
//                        currentTypeTopicCounts[index] = 0;
//                    } else {
//                        currentTypeTopicCounts[index] =
//                                (currentValue << topicBits) + oldTopic;
//                    }
//
//                    // Shift the reduced value to the right, if necessary.
//
//                    int subIndex = index;
//                    while (subIndex < currentTypeTopicCounts.length - 1
//                            && currentTypeTopicCounts[subIndex] < currentTypeTopicCounts[subIndex + 1]) {
//                        int temp = currentTypeTopicCounts[subIndex];
//                        currentTypeTopicCounts[subIndex] = currentTypeTopicCounts[subIndex + 1];
//                        currentTypeTopicCounts[subIndex + 1] = temp;
//
//                        subIndex++;
//                    }
//
//                    alreadyDecremented = true;
//                } else {
//                    //add normalized smoothingOnly coefficient 
//                    score =
//                            (smoothOnlyCachedCoefficients[0][currentTopic]// / docLength[0]
//                            ) * currentValue;
//                    topicTermMass += score;
//                    topicTermScores[index] = score;
//
//                    index++;
//                }
//            }
//
//
//            //normalize smoothing mass. 
//            //ThreadLocalRandom.current().nextDouble()
//            double sample = random.nextUniform() * (smoothingOnlyMass[0] // / docLength[0] 
//                    + topicBetaMass + topicTermMass);
//            //random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);
//
//            double origSample = sample;
//
//            //	Make sure it actually gets set
//            newTopic = -1;
//
//            if (sample < topicTermMass) {
//                //topicTermCount++;
//
//                i = -1;
//                while (sample > 0) {
//                    i++;
//                    sample -= topicTermScores[i];
//                }
//
//                newTopic = currentTypeTopicCounts[i] & topicMask;
//                currentValue = currentTypeTopicCounts[i] >> topicBits;
//
//                currentTypeTopicCounts[i] = ((currentValue + 1) << topicBits) + newTopic;
//
//                // Bubble the new value up, if necessary
//
//                while (i > 0
//                        && currentTypeTopicCounts[i] > currentTypeTopicCounts[i - 1]) {
//                    int temp = currentTypeTopicCounts[i];
//                    currentTypeTopicCounts[i] = currentTypeTopicCounts[i - 1];
//                    currentTypeTopicCounts[i - 1] = temp;
//
//                    i--;
//                }
//
//            } else {
//                sample -= topicTermMass;
//
//                if (sample < topicBetaMass) {
//                    //betaTopicCount++;
//
//                    sample /= beta[0];
//
//
//                    for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//
//                        int topic = localTopicIndex[denseIndex];
//
//                        //for (byte j = 0; j < numModalities; j++) {
//                        // if (docLength[j] > 0) {
//                        double normSumN = localTopicCounts[topic] /// docLength[j]
//                                / (tokensPerTopic[0][topic] + betaSum[0]);
//
//                        sample -= normSumN;
//                        //sample -= p[0][j] * beta[j] * localTopicCounts[j][topic] / docLength[j]
//                        //        / (tokensPerTopic[0][topic] + betaSum[0]);
//
//                        //normSumN += p[0][j] * localTopicCounts[j][topic] / docLength[j];
//                        //}
//                        //}
////                                sample -= normSumN
////                                        / (tokensPerTopic[0][topic] + betaSum[0]);
//
//                        if (sample <= 0) {
//                            newTopic = topic;
//                            break;
//                        }
//                    }
//                    if (sample > 0) {
//                        newTopic = -1;
//                    }
//
//
//                } else {
//                    //smoothingOnlyCount++;
//                    //smoothingOnlyMass[i] += alpha[topic] * beta[i] / (tokensPerTopic[i][topic] + betaSum[i]);
//
//                    sample -= topicBetaMass;
//
//                    //sample *= docLength[0];
//                    sample /= beta[0];
//
//                    int topic = 0;
//                    sample -= alpha[topic]
//                            / (tokensPerTopic[0][topic] + betaSum[0]);
//
//                    while (sample > 0.0 && topic < numTopics - 1) {
//                        topic++;
//                        sample -= alpha[topic]
//                                / (tokensPerTopic[0][topic] + betaSum[0]);
//                    }
//
//                    if (sample <= 0.0) {
//                        newTopic = topic;
//                        break;
//                    } else {
//                        newTopic = -1;
//                    }
//
//                }
//
//                // Move to the position for the new topic,
//                //  which may be the first empty position if this
//                //  is a new topic for this word.
//
//                index = 0;
//                while (currentTypeTopicCounts[index] > 0
//                        && (currentTypeTopicCounts[index] & topicMask) != newTopic) {
//                    index++;
//                    if (index == currentTypeTopicCounts.length) {
//                        System.err.println("type: " + type + " new topic: " + newTopic);
//                        for (int k = 0; k < currentTypeTopicCounts.length; k++) {
//                            System.err.print((currentTypeTopicCounts[k] & topicMask) + ":"
//                                    + (currentTypeTopicCounts[k] >> topicBits) + " ");
//                        }
//                        System.err.println();
//
//                    }
//                }
//
//
//                // index should now be set to the position of the new topic,
//                //  which may be an empty cell at the end of the list.
//
//                if (currentTypeTopicCounts[index] == 0) {
//                    // inserting a new topic, guaranteed to be in
//                    //  order w.r.t. count, if not topic.
//                    currentTypeTopicCounts[index] = (1 << topicBits) + newTopic;
//                } else {
//                    currentValue = currentTypeTopicCounts[index] >> topicBits;
//                    currentTypeTopicCounts[index] = ((currentValue + 1) << topicBits) + newTopic;
//
//                    // Bubble the increased value left, if necessary
//                    while (index > 0
//                            && currentTypeTopicCounts[index] > currentTypeTopicCounts[index - 1]) {
//                        int temp = currentTypeTopicCounts[index];
//                        currentTypeTopicCounts[index] = currentTypeTopicCounts[index - 1];
//                        currentTypeTopicCounts[index - 1] = temp;
//
//                        index--;
//                    }
//                }
//
//            }
//
//            if (newTopic == -1) {
//                System.err.println("WorkerRunnable sampling error: " + origSample + " " + sample + " " + smoothingOnlyMass[0] + " "
//                        + topicBetaMass + " " + topicTermMass);
//                newTopic = numTopics - 1; // TODO is this appropriate
//                //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
//            }
//            //assert(newTopic != -1);
//
//            //			Put that new topic into the counts
//            oneDocTopics[position] = newTopic;
//
//            smoothingOnlyMass[0] -= alpha[newTopic] * beta[0]
//                    / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//            topicBetaMass -= beta[0] * localTopicCounts[newTopic]// / docLength[0]
//                    / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//
//            // cachedCoefficients[0][newTopic] -= localTopicCounts[newTopic]// / docLength[0]
//            //         / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//            localTopicCounts[newTopic]++;
//
//            // If this is a new topic for this document,
//            //  add the topic to the dense index.
////
////                    boolean isNewTopic = false;
////                    if (localTopicCounts[newTopic] == 1) {
////                        isNewTopic = true;
////                        for (byte j = 0; j < numModalities; j++) {
////                            if (j != 0) {
////                                if (localTopicCounts[newTopic] != 0) {
////                                    isNewTopic = false;
////                                }
////                            }
////                        }
////                    }
//
//
//            if (localTopicCounts[newTopic] == 1) {
//
//                // First find the point where we 
//                //  should insert the new topic by going to
//                //  the end (which is the only reason we're keeping
//                //  track of the number of non-zero
//                //  topics) and working backwards
//
//                denseIndex = nonZeroTopics;
//
//                while (denseIndex > 0
//                        && localTopicIndex[denseIndex - 1] > newTopic) {
//
//                    localTopicIndex[denseIndex] =
//                            localTopicIndex[denseIndex - 1];
//                    denseIndex--;
//                }
//
//                localTopicIndex[denseIndex] = newTopic;
//                nonZeroTopics++;
//            }
//
//            tokensPerTopic[0][newTopic]++;
//
//            //	update the coefficients for the non-zero topics
//            smoothingOnlyMass[0] += alpha[newTopic] * beta[0]
//                    / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//            topicBetaMass += beta[0] * localTopicCounts[newTopic]// / docLength[0]
//                    / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//
//            // cachedCoefficients[0][newTopic] += localTopicCounts[newTopic]// / docLength[0]
//            //         / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//
//
//        }
//        if (shouldSaveState) {
//            // Update the document-topic count histogram,
//            //  for dirichlet estimation
//            docLengthCounts[ docLength]++;
//
//            for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//                int topic = localTopicIndex[denseIndex];
//                topicDocCounts[topic][ localTopicCounts[topic]]++;
//            }
//        }
//
////	Clean up our mess: reset the coefficients to values with only
////	smoothing. The next doc will update its own non-zero topics...
////not needed we have seperate smothOnlyCoefficients
////        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
////            int topic = localTopicIndex[denseIndex];
////
////            cachedCoefficients[topic] =
////                    alpha[topic] / (tokensPerTopic[topic] + betaSum);
////        }
//    }
    //    protected void sampleTopicsForOneDoc(FeatureSequence tokenSequence,
//            FeatureSequence topicSequence,
//            boolean readjustTopicsAndStats /* currently ignored */) {
//
//        int[] oneDocTopics = topicSequence.getFeatures();
//
//        int[] currentTypeTopicCounts;
//        int type, oldTopic, newTopic;
//        double topicWeightsSum;
//        int docLength = tokenSequence.getLength();
//
//        int[] localTopicCounts = new int[numTopics];
//        int[] localTopicIndex = new int[numTopics];
//
//        //		populate topic counts
//        for (int position = 0; position < docLength; position++) {
//            if (oneDocTopics[position] == ParallelTopicModel.UNASSIGNED_TOPIC) {
//                continue;
//            }
//            localTopicCounts[oneDocTopics[position]]++;
//        }
//
//        // Build an array that densely lists the topics that
//        //  have non-zero counts.
//        int denseIndex = 0;
//        for (int topic = 0; topic < numTopics; topic++) {
//            if (localTopicCounts[topic] != 0) {
//                localTopicIndex[denseIndex] = topic;
//                denseIndex++;
//            }
//        }
//
//        // Record the total number of non-zero topics
//        int nonZeroTopics = denseIndex;
//
//        //		Initialize the topic count/beta sampling bucket
//        double topicBetaMass = 0.0;
//
//        // Initialize cached coefficients and the topic/beta 
//        //  normalizing constant.
//
//        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//            int topic = localTopicIndex[denseIndex];
//            int n = localTopicCounts[topic];
//
//            //	initialize the normalization constant for the (B * n_{t|d}) term
//            topicBetaMass += beta[0] * n / (tokensPerTopic[0][topic] + betaSum[0]);
//
//            //	update the coefficients for the non-zero topics
//            smoothOnlyCachedCoefficients[0][topic] = (alpha[topic] + n) / (tokensPerTopic[0][topic] + betaSum[0]);
//        }
//
//        double topicTermMass = 0.0;
//
//        double[] topicTermScores = new double[numTopics];
//        int[] topicTermIndices;
//        int[] topicTermValues;
//        int i;
//        double score;
//
//        //	Iterate over the positions (words) in the document 
//        for (int position = 0; position < docLength; position++) {
//            type = tokenSequence.getIndexAtPosition(position);
//            oldTopic = oneDocTopics[position];
//
//            currentTypeTopicCounts = typeTopicCounts[0][type];
//
//            if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {
//                //	Remove this token from all counts. 
//
//                // Remove this topic's contribution to the 
//                //  normalizing constants
//                smoothingOnlyMass[0] -= alpha[oldTopic] * beta[0]
//                        / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//                topicBetaMass -= beta[0] * localTopicCounts[oldTopic]
//                        / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//
//                // Decrement the local doc/topic counts
//
//                localTopicCounts[oldTopic]--;
//
//                // Maintain the dense index, if we are deleting
//                //  the old topic
//                if (localTopicCounts[oldTopic] == 0) {
//
//                    // First get to the dense location associated with
//                    //  the old topic.
//
//                    denseIndex = 0;
//
//                    // We know it's in there somewhere, so we don't 
//                    //  need bounds checking.
//                    while (localTopicIndex[denseIndex] != oldTopic) {
//                        denseIndex++;
//                    }
//
//                    // shift all remaining dense indices to the left.
//                    while (denseIndex < nonZeroTopics) {
//                        if (denseIndex < localTopicIndex.length - 1) {
//                            localTopicIndex[denseIndex] =
//                                    localTopicIndex[denseIndex + 1];
//                        }
//                        denseIndex++;
//                    }
//
//                    nonZeroTopics--;
//                }
//
//                // Decrement the global topic count totals
//                tokensPerTopic[0][oldTopic]--;
//                assert (tokensPerTopic[0][oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
//
//
//                // Add the old topic's contribution back into the
//                //  normalizing constants.
//                smoothingOnlyMass[0] += alpha[oldTopic] * beta[0]
//                        / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//                topicBetaMass += beta[0] * localTopicCounts[oldTopic]
//                        / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//
//                // Reset the cached coefficient for this topic
//                smoothOnlyCachedCoefficients[0][oldTopic] =
//                        (alpha[oldTopic] + localTopicCounts[oldTopic])
//                        / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//            }
//
//
//            // Now go over the type/topic counts, decrementing
//            //  where appropriate, and calculating the score
//            //  for each topic at the same time.
//
//            int index = 0;
//            int currentTopic, currentValue;
//
//            boolean alreadyDecremented = (oldTopic == ParallelTopicModel.UNASSIGNED_TOPIC);
//
//            topicTermMass = 0.0;
//
//            while (index < currentTypeTopicCounts.length
//                    && currentTypeTopicCounts[index] > 0) {
//                currentTopic = currentTypeTopicCounts[index] & topicMask;
//                currentValue = currentTypeTopicCounts[index] >> topicBits;
//
//                if (!alreadyDecremented
//                        && currentTopic == oldTopic) {
//
//                    // We're decrementing and adding up the 
//                    //  sampling weights at the same time, but
//                    //  decrementing may require us to reorder
//                    //  the topics, so after we're done here,
//                    //  look at this cell in the array again.
//
//                    currentValue--;
//                    if (currentValue == 0) {
//                        currentTypeTopicCounts[index] = 0;
//                    } else {
//                        currentTypeTopicCounts[index] =
//                                (currentValue << topicBits) + oldTopic;
//                    }
//
//                    // Shift the reduced value to the right, if necessary.
//
//                    int subIndex = index;
//                    while (subIndex < currentTypeTopicCounts.length - 1
//                            && currentTypeTopicCounts[subIndex] < currentTypeTopicCounts[subIndex + 1]) {
//                        int temp = currentTypeTopicCounts[subIndex];
//                        currentTypeTopicCounts[subIndex] = currentTypeTopicCounts[subIndex + 1];
//                        currentTypeTopicCounts[subIndex + 1] = temp;
//
//                        subIndex++;
//                    }
//
//                    alreadyDecremented = true;
//                } else {
//                    score =
//                            smoothOnlyCachedCoefficients[0][currentTopic] * currentValue;
//                    topicTermMass += score;
//                    topicTermScores[index] = score;
//
//                    index++;
//                }
//            }
//
//            double sample = random.nextUniform() * (smoothingOnlyMass[0] + topicBetaMass + topicTermMass);
//            double origSample = sample;
//
//            //	Make sure it actually gets set
//            newTopic = -1;
//
//            if (sample < topicTermMass) {
//                //topicTermCount++;
//
//                i = -1;
//                while (sample > 0) {
//                    i++;
//                    sample -= topicTermScores[i];
//                }
//
//                newTopic = currentTypeTopicCounts[i] & topicMask;
//                currentValue = currentTypeTopicCounts[i] >> topicBits;
//
//                currentTypeTopicCounts[i] = ((currentValue + 1) << topicBits) + newTopic;
//
//                // Bubble the new value up, if necessary
//
//                while (i > 0
//                        && currentTypeTopicCounts[i] > currentTypeTopicCounts[i - 1]) {
//                    int temp = currentTypeTopicCounts[i];
//                    currentTypeTopicCounts[i] = currentTypeTopicCounts[i - 1];
//                    currentTypeTopicCounts[i - 1] = temp;
//
//                    i--;
//                }
//
//            } else {
//                sample -= topicTermMass;
//
//                if (sample < topicBetaMass) {
//                    //betaTopicCount++;
//
//                    sample /= beta[0];
//
//                    for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//                        int topic = localTopicIndex[denseIndex];
//
//                        sample -= localTopicCounts[topic]
//                                / (tokensPerTopic[0][topic] + betaSum[0]);
//
//                        if (sample <= 0.0) {
//                            newTopic = topic;
//                            break;
//                        }
//                    }
//
//                } else {
//                    //smoothingOnlyCount++;
//
//                    sample -= topicBetaMass;
//
//                    sample /= beta[0];
//
//                    newTopic = 0;
//                    sample -= alpha[newTopic]
//                            / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//                    while (sample > 0.0) {
//                        newTopic++;
//                        sample -= alpha[newTopic]
//                                / (tokensPerTopic[0][newTopic] + betaSum[0]);
//                    }
//
//                }
//
//                // Move to the position for the new topic,
//                //  which may be the first empty position if this
//                //  is a new topic for this word.
//
//                index = 0;
//                while (currentTypeTopicCounts[index] > 0
//                        && (currentTypeTopicCounts[index] & topicMask) != newTopic) {
//                    index++;
//                    if (index == currentTypeTopicCounts.length) {
//                        System.err.println("type: " + type + " new topic: " + newTopic);
//                        for (int k = 0; k < currentTypeTopicCounts.length; k++) {
//                            System.err.print((currentTypeTopicCounts[k] & topicMask) + ":"
//                                    + (currentTypeTopicCounts[k] >> topicBits) + " ");
//                        }
//                        System.err.println();
//
//                    }
//                }
//
//
//                // index should now be set to the position of the new topic,
//                //  which may be an empty cell at the end of the list.
//
//                if (currentTypeTopicCounts[index] == 0) {
//                    // inserting a new topic, guaranteed to be in
//                    //  order w.r.t. count, if not topic.
//                    currentTypeTopicCounts[index] = (1 << topicBits) + newTopic;
//                } else {
//                    currentValue = currentTypeTopicCounts[index] >> topicBits;
//                    currentTypeTopicCounts[index] = ((currentValue + 1) << topicBits) + newTopic;
//
//                    // Bubble the increased value left, if necessary
//                    while (index > 0
//                            && currentTypeTopicCounts[index] > currentTypeTopicCounts[index - 1]) {
//                        int temp = currentTypeTopicCounts[index];
//                        currentTypeTopicCounts[index] = currentTypeTopicCounts[index - 1];
//                        currentTypeTopicCounts[index - 1] = temp;
//
//                        index--;
//                    }
//                }
//
//            }
//
//            if (newTopic == -1) {
//                System.err.println("WorkerRunnable sampling error: " + origSample + " " + sample + " " + smoothingOnlyMass + " "
//                        + topicBetaMass + " " + topicTermMass);
//                newTopic = numTopics - 1; // TODO is this appropriate
//                //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
//            }
//            //assert(newTopic != -1);
//
//            //			Put that new topic into the counts
//            oneDocTopics[position] = newTopic;
//
//            smoothingOnlyMass[0] -= alpha[newTopic] * beta[0]
//                    / (tokensPerTopic[0][newTopic] + betaSum[0]);
//            topicBetaMass -= beta[0] * localTopicCounts[newTopic]
//                    / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//            localTopicCounts[newTopic]++;
//
//            // If this is a new topic for this document,
//            //  add the topic to the dense index.
//            if (localTopicCounts[newTopic] == 1) {
//
//                // First find the point where we 
//                //  should insert the new topic by going to
//                //  the end (which is the only reason we're keeping
//                //  track of the number of non-zero
//                //  topics) and working backwards
//
//                denseIndex = nonZeroTopics;
//
//                while (denseIndex > 0
//                        && localTopicIndex[denseIndex - 1] > newTopic) {
//
//                    localTopicIndex[denseIndex] =
//                            localTopicIndex[denseIndex - 1];
//                    denseIndex--;
//                }
//
//                localTopicIndex[denseIndex] = newTopic;
//                nonZeroTopics++;
//            }
//
//            tokensPerTopic[0][newTopic]++;
//
//            //	update the coefficients for the non-zero topics
//            smoothOnlyCachedCoefficients[0][newTopic] =
//                    (alpha[newTopic] + localTopicCounts[newTopic])
//                    / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//            smoothingOnlyMass[0] += alpha[newTopic] * beta[0]
//                    / (tokensPerTopic[0][newTopic] + betaSum[0]);
//            topicBetaMass += beta[0] * localTopicCounts[newTopic]
//                    / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//        }
//
//        if (shouldSaveState) {
//            // Update the document-topic count histogram,
//            //  for dirichlet estimation
//            docLengthCounts[ docLength]++;
//
//            for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//                int topic = localTopicIndex[denseIndex];
//
//                topicDocCounts[topic][ localTopicCounts[topic]]++;
//            }
//        }
//
//        //	Clean up our mess: reset the coefficients to values with only
//        //	smoothing. The next doc will update its own non-zero topics...
//
//        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//            int topic = localTopicIndex[denseIndex];
//
//            smoothOnlyCachedCoefficients[0][topic] =
//                    alpha[topic] / (tokensPerTopic[0][topic] + betaSum[0]);
//        }
//
//    }
}
