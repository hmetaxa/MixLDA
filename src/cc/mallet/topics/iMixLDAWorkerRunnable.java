/* Copyright (C) 2013 Omiros Metaxas */
package cc.mallet.topics;

import java.util.Arrays;
import java.util.ArrayList;

//import java.util.zip.*;
//import java.io.*;
//import java.text.NumberFormat;
import cc.mallet.types.*;
import cc.mallet.util.Randoms;
import java.util.HashSet;
//import gnu.trove.list.array.double[];
//import gnu.trove.list.array.int[];
//gnu.trove.map.hash.TIntIntHashMap
//import gnu.trove.map.hash.TIntIntHashMap;
//import gnu.trove.map.hash.TObjectIntHashMap;
//import java.util.HashMap;
import java.util.concurrent.ThreadLocalRandom;
//import org.knowceans.util.Samplers;
import org.knowceans.util.Vectors;

/**
 * A parallel semi supervised topic model runnable task.
 *
 * @author Omiros Metaxas extending MALLET Parallel topic model of David Mimno,
 * Andrew McCallum
 *
 * TODO: See if using the "minimal path" assumption to reduce bookkeeping gives
 * the same results. Huge Memory consumption due to topicDocCounts (*
 * NumThreads), and striling number of first kind allss double[][] Also 2x
 * slower than the parametric version due to UpdateAlphaAndSmoothing
 *
 */
public class iMixLDAWorkerRunnable implements Runnable {
//
//    public class MassValue {
//
//        public double topicTermMass;
//        public double topicBetaMass;
//        public double smoothingOnlyMass;
//        //public int nonZeroTopics;
//        //public double few;//frequency exclusivity weight we have an array for that
//    }

    public static final int UNASSIGNED_TOPIC = -1;
    boolean isFinished = true;
    boolean checkConvergenceRate = false;
    //boolean ignoreLabels = false;
    //boolean ignoreSkewness = false;
    ArrayList<MixTopicModelTopicAssignment> data;
    int startDoc, numDocs;
    protected int numTopics; // Number of topics to be fit
    protected int maxNumTopics; // Number of topics to be fit
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
    protected double gammaRoot;
    protected double[] gamma;
    protected double[][] alpha;
    protected double[] alphaSum;
    protected double[] beta;   // Prior on per-topic multinomial distribution over words
    protected double[] betaSum;
    public static final double DEFAULT_BETA = 0.01;
    //homer 
    protected double[] smoothingOnlyMass;// = 0.0;
    protected double[][] smoothOnlyCachedCoefficients;
    protected int[][][] typeTopicCounts; //per modality // indexed by <modalityIndex, feature index, topic index>
    protected int[][] tokensPerTopic; //per modality// indexed by <modalityIndex,topic index>
    //protected int[][][] topicsPerDoc; //per modality// indexed by <modalityIndex,Doc, topic index>

    //protected int[][][] typeTopicCounts; // indexed by <modality index, feature index, topic index>
    //protected int[][] tokensPerTopic; // indexed by <modality index, topic index>
    // for dirichlet estimation
    //protected int[][] docLengthCounts; // histogram of document sizes
    protected int[][][] topicDocCounts; // histogram of document/topic counts, indexed by <topic index, sequence position index>
    boolean shouldSaveState = false;
    boolean shouldBuildLocalCounts = true;
    protected Randoms random;
    protected double[] tablesPerModality; // num of tables per modality
    // The skew index of eachType
    //protected final double[][] typeSkewIndexes;
    //protected double[] skewWeight;// = 1;
    protected double[][] p_a; // a for beta prior for modalities correlation
    protected double[][] p_b; // b for beta prir for modalities correlation
    protected boolean fastSampling = false; // b for beta prir for modalities correlation
    double[][][] pDistr_Mean; // modalities correlation distribution accross documents (used in a, b beta params optimization)
    //double[][][] pDistr_Var; // modalities correlation distribution accross documents (used in a, b beta params optimization)
    //double avgSkew = 0;

    HashSet<Integer> inActiveTopicIndex = new HashSet<Integer>(); //inactive topic index for all modalities

    public iMixLDAWorkerRunnable(int numTopics, int maxNumTopics,
            double[][] alpha, double[] alphaSum,
            double[] beta, Randoms random,
            final ArrayList<MixTopicModelTopicAssignment> data,
            int[][][] typeTopicCounts,
            int[][] tokensPerTopic,
            int[][][] topicDocCounts,
            int startDoc, int numDocs, byte numModalities,
            //double[][] typeSkewIndexes, iMixParallelTopicModelFixTopics.SkewType skewOn, double[] skewWeight,
            double[][] p_a, double[][] p_b, boolean checkConvergenceRate, double[] gamma, double gammaRoot) {

        this.data = data;
        this.checkConvergenceRate = checkConvergenceRate;
        this.numTopics = numTopics;
        this.maxNumTopics = maxNumTopics;
        //this.numIndependentTopics = numIndependentTopics;
        this.numModalities = numModalities;
        //this.numCommonTopics = numTopics - numIndependentTopics * numModalities;
        this.numTypes = new int[numModalities];
        this.betaSum = new double[numModalities];
        this.tablesPerModality = new double[numModalities];
        //this.skewWeight = skewWeight;
        this.p_a = p_a;  //new double[numModalities][numModalities];
        this.p_b = p_b;
        this.smoothingOnlyMass = new double[numModalities];
        this.smoothOnlyCachedCoefficients = new double[numModalities][];
        //this.typeSkewIndexes = typeSkewIndexes;
        this.gamma = gamma;
        this.gammaRoot = gammaRoot;

        this.alpha = new double[numModalities][maxNumTopics];
        this.alphaSum = new double[numModalities];

        for (byte i = 0; i < numModalities; i++) {
            this.numTypes[i] = typeTopicCounts[i].length;
            this.betaSum[i] = beta[i] * numTypes[i];
            this.smoothOnlyCachedCoefficients[i] = new double[maxNumTopics];
            Arrays.fill(this.smoothOnlyCachedCoefficients[i], 0);
            Arrays.fill(this.alpha[i], 0, numTopics, 1.0 / numTopics);
        }

        //this.alphaSum = alphaSum;
        if (Integer.bitCount(maxNumTopics) == 1) {
            // exact power of 2
            topicMask = maxNumTopics - 1;
            topicBits = Integer.bitCount(topicMask);
        } else {
            // otherwise add an extra bit
            topicMask = Integer.highestOneBit(maxNumTopics) * 2 - 1;
            topicBits = Integer.bitCount(topicMask);
        }

        this.typeTopicCounts = typeTopicCounts;
        this.tokensPerTopic = tokensPerTopic;
        this.topicDocCounts = topicDocCounts;

        //this.alpha = alpha; not the global alpha anymore.. 
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

    public int getNumTopics() {
        return numTopics;
    }

    public double[] getTablesPerModality() {
        return tablesPerModality;
    }

    public double[][] getAlpha() {
        return alpha;
    }

    public int[][] getTokensPerTopic() {
        return tokensPerTopic;
    }

    public int[][][] getTypeTopicCounts() {
        return typeTopicCounts;
    }

//    public int[][] getDocLengthCounts() {
//        return docLengthCounts;
//    }
    public int[][][] getTopicDocCounts() {
        return topicDocCounts;
    }

    public double[][][] getPDistr_Mean() {
        return pDistr_Mean;
    }

//    public double[][][] getPDistr_Var() {
//        return pDistr_Var;
//    }
//    public void initializeAlphaStatistics(int size) {
//        docLengthCounts = new int[numModalities][size];
//        topicDocCounts = new int[numModalities][maxNumTopics][];
//        for (byte i = 0; i < numModalities; i++) {
//            //topicDocCounts[i] = new TIntObjectHashMap<int[]>(numTopics);
//            for (int topic = 0; topic < numTopics; topic++) {
//                topicDocCounts[i][topic] = new int[docLengthCounts[i].length];
//            }
//        }
//        //  [size];
//    }
    public void collectAlphaStatistics() {
        shouldSaveState = true;
    }

    public void resetNumTopics(int numTopics) {
        this.numTopics = numTopics;
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

//    public void resetSkewWeight(double[] skewWeight) {
//        this.skewWeight = skewWeight;
//    }
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
            //tokensPerTopic[i].fill(0, numTopics, 0);
            Arrays.fill(tokensPerTopic[i], 0);

            for (int type = 0; type < typeTopicCounts[i].length; type++) {

                //typeTopicCounts[i][type].fill(0, numTopics, 0);//.reset();
                int[] topicCounts = typeTopicCounts[i][type];

                int position = 0;
                while (position < topicCounts.length
                        && topicCounts[position] > 0) {
                    topicCounts[position] = 0;
                    position++;
                }
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

                        if (topic == UNASSIGNED_TOPIC) {
                            System.err.println(" buildLocalTypeTopicCounts UNASSIGNED_TOPIC");
                            continue;
                        }

                        tokensPerTopic[i][topic]++; //, tokensPerTopic[i][topic] + 1);

                        // The format for these arrays is 
                        //  the topic in the rightmost bits
                        //  the count in the remaining (left) bits.
                        // Since the count is in the high bits, sorting (desc)
                        //  by the numeric value of the int guarantees that
                        //  higher counts will be before the lower counts.
                        int type = tokens.getIndexAtPosition(position);

                        int[] currentTypeTopicCounts = typeTopicCounts[i][type];

                        // Start by assuming that the array is either empty
                        //  or is in sorted (descending) order.
                        // Here we are only adding counts, so if we find 
                        //  an existing location with the topic, we only need
                        //  to ensure that it is not larger than its left neighbor.
                        int index = 0;
                        int currentTopic = currentTypeTopicCounts[index] & topicMask;
                        int currentValue;

                        while (currentTypeTopicCounts[index] > 0 && currentTopic != topic) {
                            index++;
                            if (index == currentTypeTopicCounts.length) {
                                System.out.println("overflow on type " + type);
                            }
                            currentTopic = currentTypeTopicCounts[index] & topicMask;
                        }
                        currentValue = currentTypeTopicCounts[index] >> topicBits;

                        if (currentValue == 0) {
                            // new value is 1, so we don't have to worry about sorting
                            //  (except by topic suffix, which doesn't matter)

                            currentTypeTopicCounts[index] = (1 << topicBits) + topic;
                        } else {
                            currentTypeTopicCounts[index]
                                    = ((currentValue + 1) << topicBits) + topic;

                            // Now ensure that the array is still sorted by 
                            //  bubbling this value up.
                            while (index > 0
                                    && currentTypeTopicCounts[index] > currentTypeTopicCounts[index - 1]) {
                                int temp = currentTypeTopicCounts[index];
                                currentTypeTopicCounts[index] = currentTypeTopicCounts[index - 1];
                                currentTypeTopicCounts[index - 1] = temp;

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

            updateAlphaAndSmoothing();
            // Initialize the smoothing-only sampling bucket
//            Arrays.fill(smoothingOnlyMass, 0d);
//
//            for (byte i = 0; i < numModalities; i++) {
//
//                // Initialize the cached coefficients, using only smoothing.
//                //  These values will be selectively replaced in documents with
//                //  non-zero counts in particular topics.
//                //for (int topic = 0; topic < numCommonTopics; topic++) {
//                for (int topic = 0; topic < numTopics; topic++) {
//                    smoothingOnlyMass[i] += alpha[i][topic] * beta[i] / (tokensPerTopic[i][topic] + betaSum[i]);
//                    smoothOnlyCachedCoefficients[i][topic] = alpha[i][topic] / (tokensPerTopic[i][topic] + betaSum[i]);
//                }
//
//            }

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
            //double[][] totalMassPerModalityAndTopic,
            int[][] oneDocTopics,
            FeatureSequence[] tokenSequence,
            int[] docLength,
            int[][] localTopicCounts,
            int[] localTopicIndex) {

        for (byte i = 0; i < numModalities; i++) {
            docLength[i] = 0;
            //totalMassPerModalityAndTopic[i] = new double[numTopics];
            // Arrays.fill(totalMassPerModalityAndTopic[i], 0);
            //totalMassPerModalityAndTopic[i].fill(0, , 0);

            localTopicCounts[i] = new int[maxNumTopics];
            //Arrays.fill( localTopicCounts[i], 0);
            //localTopicCounts[i].fill(0, numTopics, 0);

            if (doc.Assignments[i] != null) {
                //TODO can I order by tokens/topics??
                oneDocTopics[i] = doc.Assignments[i].topicSequence.getFeatures();

                //System.arraycopy(oneDocTopics[i], 0, doc.Assignments[i].topicSequence.getFeatures(), 0, doc.Assignments[i].topicSequence.getFeatures().length-1);
                tokenSequence[i] = ((FeatureSequence) doc.Assignments[i].instance.getData());

                docLength[i] = tokenSequence[i].getLength(); //size is the same??

                //		populate topic counts
                for (int position = 0; position < docLength[i]; position++) {
                    if (oneDocTopics[i][position] == UNASSIGNED_TOPIC) {
                        System.err.println(" Init Sampling UNASSIGNED_TOPIC");
                        continue;
                    }
                    localTopicCounts[i][oneDocTopics[i][position]]++; //, localTopicCounts[i][oneDocTopics[i][position]] + 1);

                }
            }

        }
        // Build an array that densely lists the topics that
        //  have non-zero counts.
        int denseIndex = 0;
        for (int topic = 0; topic < numTopics; topic++) {
            int i = 0;
            boolean topicFound = false;
            while (i < numModalities && !topicFound) {
                if (localTopicCounts[i][topic] != 0) {
                    localTopicIndex[denseIndex] = topic;
                    denseIndex++;
                    topicFound = true;
                }
                i++;
            }
        }

        // Record the total number of non-zero topics
        int nonZeroTopics = denseIndex;

        return nonZeroTopics;
    }

    protected void initSamplingForModality(
            double[] cachedCoefficients,
            double[] totalMassOtherModalities,
            //double[][] totalMassPerModalityAndTopic,
            int[] docLength,
            int[][] localTopicCounts,
            int[] localTopicIndex,
            double[] topicBetaMass,
            double[][] p,
            int m, //active modality
            int nonZeroTopics) {

        //cachedCoefficients.reset();
        Arrays.fill(cachedCoefficients, 0);
        //cachedCoefficients.fill(0, numTopics, 0);

        //totalMassOtherModalities.reset();
        Arrays.fill(totalMassOtherModalities, 0);
        //totalMassOtherModalities.fill(0, numTopics, 0);
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

            topic = localTopicIndex[denseIndex];

            //totalMassOtherModalities.set(topic, 0);
//if (topic < numCommonTopics || (topic >= numCommonTopics + m * numIndependentTopics && topic < numCommonTopics + (m + 1) * numIndependentTopics)) {
            //  if (topic < numTopics) {
            for (byte i = 0; i < numModalities; i++) {
                if (i != m && docLength[i] != 0) {
                    totalMassOtherModalities[topic] += p[m][i] * localTopicCounts[i][topic] / docLength[i];
                }
            }

            totalMassOtherModalities[topic] = totalMassOtherModalities[topic] * (docLength[m] + alphaSum[m]);
            //	initialize the normalization constant for the (B * n_{t|d}) term
            double normSumN = (localTopicCounts[m][topic] + totalMassOtherModalities[topic])
                    / (tokensPerTopic[m][topic] + betaSum[m]);

            topicBetaMass[m] += beta[m] * normSumN;
            cachedCoefficients[topic] = normSumN;

            // }
        }

    }

    protected int removeOldTopicContribution(
            double[] cachedCoefficients,
            double[] totalMassOtherModalities,
            int[][] localTopicCounts,
            int[] localTopicIndex,
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
        smoothingOnlyMass[m] -= gamma[m] * alpha[m][oldTopic] * beta[m]
                / (tokensPerTopic[m][oldTopic] + betaSum[m]);

        double normSumN = (localTopicCounts[m][oldTopic] + totalMassOtherModalities[oldTopic])
                / (tokensPerTopic[m][oldTopic] + betaSum[m]);

        topicBetaMass[m] -= beta[m] * normSumN;
        //cachedCoefficients.set(oldTopic, normSumN);

        //decrement local histogram
        topicDocCounts[m][oldTopic][localTopicCounts[m][oldTopic]]--;

        // Decrement the local doc/topic counts
        localTopicCounts[m][oldTopic]--;

        // Decrement the global topic count totals
        tokensPerTopic[m][oldTopic]--;

        assert (tokensPerTopic[m][oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";

        // Add the old topic's contribution back into the
        //  normalizing constants.
        smoothingOnlyMass[m] += gamma[m] * alpha[m][oldTopic] * beta[m]
                / (tokensPerTopic[m][oldTopic] + betaSum[m]);

        smoothOnlyCachedCoefficients[m][oldTopic] = gamma[m] * alpha[m][oldTopic] / (tokensPerTopic[m][oldTopic] + betaSum[m]);

        normSumN = (localTopicCounts[m][oldTopic] + totalMassOtherModalities[oldTopic])
                / (tokensPerTopic[m][oldTopic] + betaSum[m]);

        topicBetaMass[m] += beta[m] * normSumN;
        cachedCoefficients[oldTopic] = normSumN;

        if (localTopicCounts[m][oldTopic] > 0) {
            topicDocCounts[m][oldTopic][localTopicCounts[m][oldTopic]]++;
        }
        // Maintain the dense index, if we are deleting
        //  the old topic
        boolean isDeletedTopic = localTopicCounts[m][oldTopic] == 0;
        byte jj = 0;
        while (isDeletedTopic && jj < numModalities) {
            // if (jj != m) { //do not check m twice
            isDeletedTopic = localTopicCounts[jj][oldTopic] == 0;
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
            while (localTopicIndex[denseIndex] != oldTopic) {
                denseIndex++;
            }

            // shift all remaining dense indices to the left.
            while (denseIndex < nonZeroTopics) {
                if (denseIndex < localTopicIndex.length - 1) {
                    localTopicIndex[denseIndex] = localTopicIndex[denseIndex + 1];
                }
                denseIndex++;
            }

            nonZeroTopics--;
        }

        return nonZeroTopics;

    }

    //TODO: I recalc them every time because sometimes I had a sampling error in FindTopicIn Beta Mass.. 
    //I shouldn't need it, thus I should check it again
//    protected void recalcBetaAndCachedCoefficients(
//            double[][] cachedCoefficients,
//            int[][] localTopicCounts,
//            int[] localTopicIndex,
//            double[] topicBetaMass,
//            int nonZeroTopics,
//            final int[] docLength,
//            byte m, //modality
//            double[][] p) {
//
//        Arrays.fill(topicBetaMass, 0);
//        for (byte i = 0; i < numModalities; i++) {
//            
//            Arrays.fill(cachedCoefficients[i], 0);
//        }
//
//        for (int denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//            int topic = localTopicIndex[denseIndex];
//            //if (topic < numCommonTopics || (topic >= numCommonTopics + m * numIndependentTopics && topic < numCommonTopics + (m + 1) * numIndependentTopics)) {
//
//            for (byte j = 0; j < numModalities; j++) {
//                if (docLength[j] > 0) {
//                    double normSumN = p[m][j] * localTopicCounts[j][topic]
//                            / (docLength[j] * (tokensPerTopic[m][topic] + betaSum[m]));
//
//                    topicBetaMass[m] += beta[m] * normSumN;
//                    cachedCoefficients[m][topic] += normSumN;
//                }
//            }
//            //}
//
//        }
//    }
    protected double calcTopicScores(
            double[] cachedCoefficients,
            int oldTopic,
            byte m,
            double[] topicTermScores,
            int[] currentTypeTopicCounts,
            int[] docLength
    ) {
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
        boolean alreadyDecremented = false; //(oldTopic == UNASSIGNED_TOPIC);

        double topicTermMass = 0.0;

        while (index < currentTypeTopicCounts.length
                && currentTypeTopicCounts.length > 0) {

            currentTopic = currentTypeTopicCounts[index] & topicMask;
            currentValue = currentTypeTopicCounts[index] >> topicBits;

            if (!alreadyDecremented && currentTopic == oldTopic) {

                // We're decrementing and adding up the 
                //  sampling weights at the same time, but
                //  decrementing may require us to reorder
                //  the topics, so after we're done here,
                //  look at this cell in the array again.
                currentValue--;
                if (currentValue == 0) {
                    currentTypeTopicCounts[index] = 0;
                } else {
                    currentTypeTopicCounts[index]
                            = (currentValue << topicBits) + oldTopic;
                }

                // Shift the reduced value to the right, if necessary.
                int subIndex = index;
                while (subIndex < currentTypeTopicCounts.length - 1
                        && currentTypeTopicCounts[subIndex] < currentTypeTopicCounts[subIndex + 1]) {
                    int temp = currentTypeTopicCounts[subIndex];
                    currentTypeTopicCounts[subIndex] = currentTypeTopicCounts[subIndex + 1];
                    currentTypeTopicCounts[subIndex + 1] = temp;

                    subIndex++;
                }

                alreadyDecremented = true;
            } else {

                score = (cachedCoefficients[currentTopic] + smoothOnlyCachedCoefficients[m][currentTopic]) * currentValue;

                topicTermMass += score;
                topicTermScores[index] = score;

                index++;
            }
        }

        return topicTermMass;
    }

    protected int findNewTopicInTopicTermMass(
            double[] topicTermScores,
            int[] currentTypeTopicCounts,
            double sample) {

        int newTopic = -1;
        int currentValue;
        int i = -1;
        while (sample > 0) {
            i++;
            sample -= topicTermScores[i];
        }
        //if (i >= 0) { // Omiros normally not needed
        newTopic = currentTypeTopicCounts[i] & topicMask;
        currentValue = currentTypeTopicCounts[i] >> topicBits;

        currentTypeTopicCounts[i] = ((currentValue + 1) << topicBits) + newTopic;

        // Bubble the new value up, if necessary
        while (i > 0
                && currentTypeTopicCounts[i] > currentTypeTopicCounts[i - 1]) {
            int temp = currentTypeTopicCounts[i];
            currentTypeTopicCounts[i] = currentTypeTopicCounts[i - 1];
            currentTypeTopicCounts[i - 1] = temp;

            i--;
        }
        //}
        return newTopic;
    }

    protected int findNewTopicInBetaMass(
            int[][] localTopicCounts,
            int[] localTopicIndex,
            double[] totalMassOtherModalities,
            int nonZeroTopics,
            byte m,
            //int[] docLength,
            double sample,
            double[][] p) {

        sample /= beta[m];
        int topic = -1;
        int denseIndex = 0;

        while (denseIndex < nonZeroTopics && sample > 0) {
            topic = localTopicIndex[denseIndex];
            //if (topic < numCommonTopics || (topic >= numCommonTopics + m * numIndependentTopics && topic < numCommonTopics + (m + 1) * numIndependentTopics)) {

            double normSumN = (localTopicCounts[m][topic] + totalMassOtherModalities[topic])
                    / (tokensPerTopic[m][topic] + betaSum[m]);

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
            byte m
    //        int[] docLength
    ) {

        int newTopic = -1;
        //sample *= docLength[m];
        //sample /= beta[m];

        int topic = 0;

        //while (sample > 0.0 && topic < numCommonTopics) {
        while (sample > 0.0 && topic < numTopics) {
            sample -= gamma[m] * alpha[m][topic] * beta[m]
                    / (tokensPerTopic[m][topic] + betaSum[m]);
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
            int[][] localTopicCounts,
            int[] localTopicIndex,
            double[] totalMassOtherModalities,
            double[] topicBetaMass,
            int nonZeroTopics,
            byte m,
            double topicTermMass,
            double[] topicTermScores,
            int[] currentTypeTopicCounts,
            //int[] docLength,
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
                        //docLength,
                        sample,
                        p);

            } else {
                //smoothingOnlyCount++;
                //smoothingOnlyMass[i] += alpha[topic] * beta[i] / (tokensPerTopic[i][topic] + betaSum[i]);
                samplingBucket = "SmoothingMass";
                sample -= topicBetaMass[m];
                if (sample < smoothingOnlyMass[m]) {
                    newTopic = findNewTopicInSmoothingMass(sample, m);
                } else {
                    //totally new topic
                    //synchronized (LOCK) {
                    if (inActiveTopicIndex.isEmpty()) {
                        newTopic = numTopics;
                    } else {
                        newTopic = inActiveTopicIndex.iterator().next();
                    }
//Update topics count will take care of them
//                        for (m = 0; m < numModalities; m++) {
//                            alpha[m].set(numTopics + 1, alpha[m].get(numTopics));
//                        }
                    // }

                }
            }
        }

        if (newTopic == -1 || newTopic > numTopics) {
            System.err.println("WorkerRunnable sampling error for modality: " + m + " in " + samplingBucket + ": Sample:" + origSample + " Smoothing:" + (smoothingOnlyMass[m]) + " Beta:"
                    + topicBetaMass[m] + " TopicTerm:" + topicTermMass);
            newTopic = oldTopic; //numCommonTopics + (m + 1) * numIndependentTopics - 1; // TODO is this appropriate
            //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
        }

        rearrangeTypeTopicCounts(currentTypeTopicCounts, newTopic);

        return newTopic;
    }

    protected void rearrangeTypeTopicCounts(
            int[] currentTypeTopicCounts,
            int newTopic) {

        // Move to the position for the new topic,
        //  which may be the first empty position if this
        //  is a new topic for this word.
        int index = 0;
        while (currentTypeTopicCounts[index] > 0
                && (currentTypeTopicCounts[index] & topicMask) != newTopic) {
            index++;
            if (index == currentTypeTopicCounts.length) { //TODO: Size is it OK
                System.err.println("error in findind new position for topic: " + newTopic);
                for (int k = 0; k < currentTypeTopicCounts.length; k++) {
                    System.err.print((currentTypeTopicCounts[k] & topicMask) + ":"
                            + (currentTypeTopicCounts[k] >> topicBits) + " ");
                }
                System.err.println();

            }
        }

        // index should now be set to the position of the new topic,
        //  which may be an empty cell at the end of the list.
        int currentValue;
        if (currentTypeTopicCounts[index] == 0) {
            // inserting a new topic, guaranteed to be in
            //  order w.r.t. count, if not topic.
            currentTypeTopicCounts[index] = (1 << topicBits) + newTopic;
        } else {
            currentValue = currentTypeTopicCounts[index] >> topicBits;
            currentTypeTopicCounts[index] = ((currentValue + 1) << topicBits) + newTopic;

            // Bubble the increased value left, if necessary
            while (index > 0
                    && currentTypeTopicCounts[index] > currentTypeTopicCounts[index - 1]) {
                int temp = currentTypeTopicCounts[index];
                currentTypeTopicCounts[index] = currentTypeTopicCounts[index - 1];
                currentTypeTopicCounts[index - 1] = temp;

                index--;
            }
        }

    }

    protected int updateTopicCounts(
            int[][] oneDocTopics,
            int position,
            int newTopic,
            double[] cachedCoefficients,
            double[] totalMassOtherModalities,
            int[][] localTopicCounts,
            int[] localTopicIndex,
            double[] topicBetaMass,
            int nonZeroTopics,
            final int[] docLength,
            byte m,
            double[][] p) {

        if (newTopic == numTopics || inActiveTopicIndex.contains(newTopic)) { //new topic in corpus

            //totalMassOtherModalities.add(0);
            //cachedCoefficients.add(0);
            for (Byte i = 0; i < numModalities; i++) {
                localTopicCounts[i][newTopic] = 0;
                //alpha[i][numTopics] = 1; //???

                tokensPerTopic[i][newTopic] = i == m ? 1 : 0;
                //smoothOnlyCachedCoefficients[i].add(0);

            }

            localTopicIndex[nonZeroTopics] = newTopic;
            nonZeroTopics++;
            topicDocCounts[m][newTopic][1] = 1;

            updateAlphaAndSmoothing();

            //	update the coefficients for the non-zero topics
//            smoothingOnlyMass[m] += gamma[m] * alpha[m][newTopic] * beta[m]
//                    / (tokensPerTopic[m][newTopic] + betaSum[m]);
//
//            smoothOnlyCachedCoefficients[m][newTopic] = gamma[m] * alpha[m][newTopic] / (tokensPerTopic[m][newTopic] + betaSum[m]);
//******* updateAlphaAndSmoothing at the end of the doc
            double normSumN = (localTopicCounts[m][newTopic] + totalMassOtherModalities[newTopic])
                    / (tokensPerTopic[m][newTopic] + betaSum[m]);

            topicBetaMass[m] += beta[m] * normSumN;
            cachedCoefficients[newTopic] = normSumN;

            oneDocTopics[m][position] = newTopic;
            if (newTopic == numTopics) {
                numTopics++;
            }
            //else { //Already done in UpdateAlphaAndSmoothing
            //    inActiveTopicIndex.remove(newTopic);
            // }

        } else {
            //			Put that new topic into the counts
            oneDocTopics[m][position] = newTopic;

            if (localTopicCounts[m][newTopic] > 0) {
                topicDocCounts[m][newTopic][localTopicCounts[m][newTopic]]--;
            }
            // If this is a new topic for this document,
            //  add the topic to the dense index.
            boolean isNewTopic = (localTopicCounts[m][newTopic] == 0);
            byte jj = 0;
            while (isNewTopic && jj < numModalities) {
                //if (jj != m) { // every other topic should have zero counts
                isNewTopic = localTopicCounts[jj][newTopic] == 0;
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
                        && localTopicIndex[denseIndex - 1] > newTopic) {

                    localTopicIndex[denseIndex]
                            = localTopicIndex[denseIndex - 1];
                    denseIndex--;
                }

                localTopicIndex[denseIndex] = newTopic;
                nonZeroTopics++;
            }
            //TODO check here for totally new topic... 

            double normSumN = (localTopicCounts[m][newTopic] + totalMassOtherModalities[newTopic])
                    / (tokensPerTopic[m][newTopic] + betaSum[m]);

            topicBetaMass[m] -= beta[m] * normSumN;

            smoothingOnlyMass[m] -= gamma[m] * alpha[m][newTopic] * beta[m]
                    / (tokensPerTopic[m][newTopic] + betaSum[m]);

            // }
            localTopicCounts[m][newTopic]++;
            tokensPerTopic[m][newTopic]++;

            topicDocCounts[m][newTopic][localTopicCounts[m][newTopic]]++;

            //	update the coefficients for the non-zero topics
            smoothingOnlyMass[m] += gamma[m] * alpha[m][newTopic] * beta[m]
                    / (tokensPerTopic[m][newTopic] + betaSum[m]);

            smoothOnlyCachedCoefficients[m][newTopic] = gamma[m] * alpha[m][newTopic] / (tokensPerTopic[m][newTopic] + betaSum[m]);

            normSumN = (localTopicCounts[m][newTopic] + totalMassOtherModalities[newTopic])
                    / (tokensPerTopic[m][newTopic] + betaSum[m]);

            topicBetaMass[m] += beta[m] * normSumN;
            cachedCoefficients[newTopic] = normSumN;
        }

        return nonZeroTopics;
    }

    protected void sampleTopicsForOneDoc(int docCnt) {

        MixTopicModelTopicAssignment doc = data.get(docCnt);

        //double[][] totalMassPerModalityAndTopic = new double[numModalities][];
        //cachedCoefficients = new double[numModalities][numTopics];// Conservative allocation... [nonZeroTopics + 10]; //we want to avoid dynamic memory allocation , thus we think that we will not have more than ten new  topics in each run
        int[][] oneDocTopics = new int[numModalities][]; //token topics sequence for document
        FeatureSequence[] tokenSequence = new FeatureSequence[numModalities]; //tokens sequence

        int[] docLength = new int[numModalities];
        int[][] localTopicCounts = new int[numModalities][];
        int[] localTopicIndex = new int[maxNumTopics]; //dense topic index for all modalities
        Arrays.fill(localTopicIndex, 0);
        //localTopicIndex.fill(0, numTopics, 0);
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
                //totalMassPerModalityAndTopic,
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
        double[] cachedCoefficients = new double[maxNumTopics];
        double[] totalMassOtherModalities = new double[maxNumTopics];
        double[] topicTermScores = new double[maxNumTopics];

//        TObjectIntHashMap<String> boostTopicSelection = new TObjectIntHashMap<String>();
        for (byte m = 0; m < numModalities; m++) // byte m = 0;
        {
            //      boostTopicSelection.clear();

            initSamplingForModality(
                    cachedCoefficients,
                    totalMassOtherModalities,
                    //totalMassPerModalityAndTopic,
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
                //long tmpPreviousTopics =0;

//                String boostId = type + "_" + tmpPreviousTopics;
//                long minTopic = (long) 1 << (63 - topicBits);
//                minTopic = tmpPreviousTopics >> topicBits;
//                minTopic += (long) 1 << (63 - topicBits);
//                minTopic = tmpPreviousTopics >> topicBits;
//                minTopic += (long) 1 << (63 - topicBits);
//                if (!(tmpPreviousTopics > minTopic && boostTopicSelection.containsKey(boostId))) {
                // if (true) {
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
                //TOCheck: probably not needed Arrays.fill(topicTermScores, 0);
                //topicTermScores.fill(0, numTopics, 0);
                // double termSkew = typeSkewIndexes[m][type];
                int[] currentTypeTopicCounts = typeTopicCounts[m][type];
                //int[] currentTypeTopicCounts = new int[typeTopicCounts[m][type].length]; //typeTopicCounts[m][type];
                //System.arraycopy(typeTopicCounts[m][type], 0, currentTypeTopicCounts, 0, typeTopicCounts[m][type].length - 1);

                double topicTermMass = calcTopicScores(
                        cachedCoefficients,
                        oldTopic,
                        m,
                        topicTermScores,
                        currentTypeTopicCounts,
                        docLength);

                //normalize smoothing mass. 
                //ThreadLocalRandom.current().nextDouble()
                // assert (smoothingOnlyMass[m] >= 0) : "smoothing Mass " + smoothingOnlyMass[m] + " below 0";
                // assert (topicBetaMass[m] >= 0) : "topicBeta Mass " + topicBetaMass[m] + " below 0";
                // assert (topicTermMass >= 0) : "topicTerm Mass " + topicTermMass + " below 0";
//                    MassValue tmpValue = new MassValue();
//                    tmpValue.smoothingOnlyMass = smoothingOnlyMass[m];
//                    tmpValue.topicBetaMass = topicBetaMass[m];
//                    tmpValue.topicTermMass = topicTermMass;
//                double sample
//                        = //random.nextUniform() *
//                        ThreadLocalRandom.current().nextDouble()
//                        * (smoothingOnlyMass[m] + topicBetaMass[m] + topicTermMass);
                //double newTopicMass = gamma[m] * alpha[m].get(numTopics) / (docLength[m] * numTypes[m]);
                double newTopicMass = numTopics + 1 == maxNumTopics ? 0 : gamma[m] * alpha[m][numTopics] / (numTypes[m]);

                double sample = ThreadLocalRandom.current().nextDouble()
                        * (newTopicMass + smoothingOnlyMass[m] + topicBetaMass[m] + topicTermMass);

//                if (topicBetaMass[m]==0)
//                {
//                      newTopic = 0;
//                }
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
                                //docLength,
                                sample,
                                p,
                                oldTopic);

//                    boostTopicSelection.put(boostId, newTopic);
//
//                } else {
//                    newTopic = boostTopicSelection.get(boostId);
//                }
                if (checkConvergenceRate) {
                    long tmpPreviousTopics = doc.Assignments[m].prevTopicsSequence[position];

                    tmpPreviousTopics = tmpPreviousTopics >> topicBits;
                    long newTopicTmp = (long) newTopic << (63 - topicBits); //long is signed
                    tmpPreviousTopics += newTopicTmp;
                    doc.Assignments[m].prevTopicsSequence[position] = tmpPreviousTopics; //doc.Assignments[m].prevTopicsSequence[position] >> topicBits;
                }
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
                    pDistr_Mean[m][i][docCnt] += (localTopicCounts[i][newTopic] > 0 ? 1.0 : 0d) / (double) docLength[m];
                    pDistr_Mean[i][m][docCnt] = pDistr_Mean[m][i][docCnt];
                    //pDistr_Var[m][i][docCnt]+= localTopicCounts[i][newTopic]/docLength[m];
                }

                //}
            }

//            if (shouldSaveState) {
            // Update the document-topic count histogram,
            //  for dirichlet estimation
            //docLengthCounts[m][docLength[m]]++;
//            for (int denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
//                int topic = localTopicIndex[denseIndex];
//                topicDocCounts[m][topic][localTopicCounts[m][topic]]++;
//            }
            // }
        }

    }

    private void updateAlphaAndSmoothing() {
        double[][] mk = new double[numModalities][numTopics + 1];

//        //double[] tt = new double[maxTopic + 2];
//        for (int t = 0; t < numTopics; t++) {
//
//            //int k = kactive.get(kk);
//            for (int doc = startDoc;
//                    doc < data.size() && doc < startDoc + numDocs;
//                    doc++) {
//                //for (int j = 0; j < numDocuments; j++) {
//                for (byte m = 0; m < numModalities; m++) {
//                    if (tokensPerTopic[m][t] > 1) {
//                        //sample number of tables
//                        mk[m][t] += Samplers.randAntoniak(gamma[m] * alpha[m][t], tokensPerTopic[m][t]);
//                        //mk[m][t] += 1;//direct minimal path assignment Samplers.randAntoniak(gamma[m] * alpha[m].get(t),  tokensPerTopic[m].get(t));
//                        // nmk[m].get(k));
//                    } else //nmk[m].get(k) = 0 or 1
//                    {
//                        mk[m][t] += tokensPerTopic[m][t];
//                    }
//                }
//            }
//        }// end outter for loop
        for (int t = 0; t < numTopics; t++) {
            inActiveTopicIndex.add(t); //inActive by default and activate if found 
        }

        for (byte m = 0; m < numModalities; m++) {
            for (int t = 0; t < numTopics; t++) {

                //int k = kactive.get(kk);
                for (int i = 0; i < topicDocCounts[m][t].length; i++) {
                    //for (int j = 0; j < numDocuments; j++) {

                    if (topicDocCounts[m][t][i] > 0 && i > 1) {
                        inActiveTopicIndex.remove(t);
                        //sample number of tables
                        // number of tables a CRP(alpha tau) produces for nmk items
                        //TODO: See if  using the "minimal path" assumption  to reduce bookkeeping gives the same results. 
                        //Huge Memory consumption due to  topicDocCounts (* NumThreads), and striling number of first kind allss double[][] 
                        //Also 2x slower than the parametric version due to UpdateAlphaAndSmoothing

                        int curTbls = 0;
                        try {
                            curTbls = random.nextAntoniak(gamma[m] * alpha[m][t], i);

                        } catch (Exception e) {
                            curTbls = 1;
                        }

                        mk[m][t] += (topicDocCounts[m][t][i] * curTbls);
                        //mk[m][t] += 1;//direct minimal path assignment Samplers.randAntoniak(gamma[m] * alpha[m].get(t),  tokensPerTopic[m].get(t));
                        // nmk[m].get(k));
                    } else if (topicDocCounts[m][t][i] > 0 && i == 1) //nmk[m].get(k) = 0 or 1
                    {
                        inActiveTopicIndex.remove(t);
                        mk[m][t] += topicDocCounts[m][t][i];
                    }
                }
            }
        }// end outter for loop

        for (byte m = 0; m < numModalities; m++) {
            //alpha[m].fill(0, numTopics, 0);

            alphaSum[m] = 0;
            mk[m][numTopics] = gammaRoot;
            tablesPerModality[m] = Vectors.sum(mk[m]);

            double[] tt = sampleDirichlet(mk[m]);

            for (int kk = 0; kk <= numTopics; kk++) {
                //int k = kactive.get(kk);
                alpha[m][kk] = tt[kk];
                alphaSum[m] += gamma[m] * tt[kk];
                //tau.set(k, tt[kk]);
            }

//            if (alpha[m].size() < numTopics + 1) {
//                alpha[m].add(tt[numTopics]);
//            } else {
//                alpha[m].set(numTopics, tt[numTopics]);
//            }
            //tau.set(K, tt[K]);
        }

        // Initialize the smoothing-only sampling bucket
        Arrays.fill(smoothingOnlyMass, 0d);

        for (byte i = 0; i < numModalities; i++) {
            for (int topic = 0; topic < numTopics; topic++) {
                smoothingOnlyMass[i] += gamma[i] * alpha[i][topic] * beta[i] / (tokensPerTopic[i][topic] + betaSum[i]);
                smoothOnlyCachedCoefficients[i][topic] = gamma[i] * alpha[i][topic] / (tokensPerTopic[i][topic] + betaSum[i]);
            }
            //not needed new mass is a seperate mass  smoothingOnlyMass[i] += gamma[i] * alpha[i][numTopics] * beta[i] / (betaSum[i]);
            //  smoothOnlyCachedCoefficients[i][numTopics] = gamma[i] * alpha[i][numTopics] / (betaSum[i]);

        }

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

//    
//    protected int MAXSTIRLING = 20000;
//
//    /**
//     * maximum stirling number in allss
//     */
//    protected int maxnn = 1;
//
//    /**
//     * contains all stirling number iteratively calculated so far
//     */
//    protected double[][] allss = new double[MAXSTIRLING][];
//
//    /**
//     *
//     */
//    //protected double[] logmaxss = new double[MAXSTIRLING];
//
//    //protected double lmss = 0;
//
//    /**
//     * [ss lmss] = stirling(nn) Gives unsigned Stirling numbers of the first
//     * kind s(nn,*) in ss. ss(i) = s(nn,i-1). ss is normalized so that maximum
//     * value is 1, and the log of normalization is given in lmss (static
//     * variable). After Teh (npbayes).
//     *
//     * @param nn
//     * @return
//     */
//    public double[] stirling(int nn) {
//        if (allss[0] == null) {
//            allss[0] = new double[1];
//            allss[0][0] = 1;
//            //logmaxss[0] = 0;
//        }
//
//        if (nn > maxnn) {
//            for (int mm = maxnn; mm < nn; mm++) {
//                int len = allss[mm - 1].length + 1;
//                if (allss[mm] == null) {
//                    allss[mm] = new double[len];
//                }
//                Arrays.fill(allss[mm], 0);
//
//                for (int xx = 0; xx < len; xx++) {
//                    // allss{mm} = [allss{mm-1}*(mm-1) 0] + ...
//                    allss[mm][xx] += (xx < len - 1) ? allss[mm - 1][xx] * mm
//                            : 0;
//                    // [0 allss{mm-1}];
//                    allss[mm][xx] += (xx == 0) ? 0 : allss[mm - 1][xx - 1];
//                }
//                double mss = Vectors.max(allss[mm]);
//                Vectors.mult(allss[mm], 1 / mss);
//                //logmaxss[mm] = logmaxss[mm - 1] + Math.log(mss);
//            }
//            maxnn = nn;
//        }
//        //lmss = logmaxss[nn - 1];
//        return allss[nn - 1];
//    }
//
//    /**
//     * sample number of components m that a DP(alpha, G0) has after n samples.
//     * This was first published by Antoniak (1974). TODO: another check, as
//     * direct simulation of CRP tables produces higher results
//     *
//     * @param alpha
//     * @param n
//     * @return
//     */
//    public int randAntoniak(double alpha, int n) {
////        double[] p = stirling(n);
////        double aa = 1;
////        for (int m = 0; m < p.length; m++) {
////            p[m] *= aa;
////            aa *= alpha;
////        }
////        
//        //alternatively using direct simulation of CRP// OMIROS: too SLOWWW
//        int R = 20;
//        double ainv = 1 / (alpha);
//        double nt = 0;
//        double[] p = new double[n];
//        for (int r = 0; r < R; r++) {
//            for (int m = 0; m < n; m++) {
//                for (int t = 0; t < n; t++, nt += ainv) {
//                    p[m] +=   random.nextBernoulli(1 / (nt + 1));
//                }
//            }
//        }
//        Vectors.mult(p, 1. / R);
//
//        return random.nextDiscrete(p) + 1;
//    }
}
