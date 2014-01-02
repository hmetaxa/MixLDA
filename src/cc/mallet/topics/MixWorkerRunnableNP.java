/* Copyright (C) 2013 Omiros Metaxas */
package cc.mallet.topics;

import java.util.Arrays;
import java.util.ArrayList;

//import java.util.zip.*;

//import java.io.*;
//import java.text.NumberFormat;

import cc.mallet.types.*;
import cc.mallet.util.Randoms;
//import java.util.HashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Queue;
import org.knowceans.util.Samplers;
import org.knowceans.util.Vectors;

/**
 * A parallel semi supervised topic model runnable task.
 *
 * @author Omiros Metaxas extending MALLET Parallel topic model of author David
 * Mimno, Andrew McCallum test BOX sync
 *
 */
public class MixWorkerRunnableNP implements Runnable {

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
    protected int maxNumTopics; // Max Number of topics to be fit
    //protected int numActiveTopics; // Number of active topics to be fit
    protected byte numModalities;
    // These values are used to encode type/topic counts as
    //  count/topic pairs in a single int.
    protected int topicMask;
    protected int topicBits;
    protected int[] numTypes;
    protected double[] avgTypeCount; //not used for now
    protected int[][] typeTotals; //not used for now
    //homer
    //protected final double[] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
    protected double alpha;
    protected double gamma;
    protected double[] tau;
    protected double[] beta;   // Prior on per-topic multinomial distribution over words
    protected double[] betaSum;
    public static final double DEFAULT_BETA = 0.01;
    //homer 
    protected double[] smoothingOnlyMass;// = 0.0;
    protected double[][] smoothOnlyCachedCoefficients;
    protected int[][][] typeTopicCounts; // indexed by <modality index, feature index, topic index>
    protected int[][] tokensPerTopic; // indexed by <modality index, topic index>
    // for dirichlet estimation
    protected int[] docLengthCounts; // histogram of document sizes
    protected int[][] topicDocCounts; // histogram of document/topic counts, indexed by <topic index, sequence position index>
    boolean shouldSaveState = false;
    boolean shouldBuildLocalCounts = true;
    protected Randoms random;
    // The skew index of eachType
    protected final double[][] typeSkewIndexes;
    protected double[] skewWeight;// = 1;
    protected double[][] p_a; // a for beta prior for modalities correlation
    protected double[][] p_b; // b for beta prir for modalities correlation
    protected boolean fastSampling = false; // b for beta prir for modalities correlation
    protected double[][][] pDistr_Mean; // modalities correlation distribution accross documents (used in a, b beta params optimization)
    //double[][][] pDistr_Var; // modalities correlation distribution accross documents (used in a, b beta params optimization)
    //double avgSkew = 0;
    private Queue<Integer> nonActiveTopics;
    protected int totalTables;
    protected int[] tablesPerTopic;// indexed by <topic index>
    protected int maxTopic;
    boolean fixedNumTopics = false;
    //protected int[] nonActiveTopics;

    public MixWorkerRunnableNP(int maxNumTopics,
            double alpha, double gamma,
            double[] beta, Randoms random,
            final ArrayList<MixTopicModelTopicAssignment> data,
            int[][][] typeTopicCounts, int[][] tokensPerTopic,
            int maxTopic, int totalTables, int[] tablesPerTopic,
            int startDoc, int numDocs, boolean ignoreLabels, byte numModalities,
            double[][] typeSkewIndexes, double[] skewWeight, double[][] p_a, double[][] p_b) {

        this.data = data;

        this.maxNumTopics = maxNumTopics;
        //this.numActiveTopics = initialNumTopics;
        //this.fixedNumTopics = numActiveTopics == this.maxNumTopics;

        this.numModalities = numModalities;

        this.numTypes = new int[numModalities];
        this.betaSum = new double[numModalities];
        this.skewWeight = skewWeight;
        this.p_a = p_a;  //new double[numModalities][numModalities];
        this.p_b = p_b;
        this.smoothingOnlyMass = new double[numModalities];
        this.smoothOnlyCachedCoefficients = new double[numModalities][maxNumTopics];
        //this.nonActiveTopics = new int[maxNumTopics];
        //this.nonActiveTopics = new PriorityQueue<Integer>(maxNumTopics / 3, topicComparator); go to global
        this.typeSkewIndexes = typeSkewIndexes;


        this.totalTables = totalTables;
        this.tablesPerTopic = tablesPerTopic;
        this.maxTopic = maxTopic;

        for (byte i = 0; i < numModalities; i++) {
            this.numTypes[i] = typeTopicCounts[i].length;
            this.betaSum[i] = beta[i] * numTypes[i];
            //Arrays.fill(this.p[i], 1d);
        }

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

        this.alpha = alpha;

        this.beta = beta;
        this.gamma = gamma;

        this.random = random;

        this.startDoc = startDoc;
        this.numDocs = numDocs;






        //System.err.println("WorkerRunnable Thread: " + maxNumTopics + " topics, " + topicBits + " topic bits, " + 
        //				   Integer.toBinaryString(topicMask) + " topic mask");

    }
    //Comparator for sorting topics
    public static Comparator<Integer> topicComparator = new Comparator<Integer>() {
        @Override
        public int compare(Integer c1, Integer c2) {
            return (int) (c1 - c2);
        }
    };

    /**
     * If there is only one thread, we don't need to go through communication
     * overhead. This method asks this worker not to prepare local type-topic
     * counts. The method should be called when we are using this code in a
     * non-threaded environment.
     */
    public void makeOnlyThread() {
        shouldBuildLocalCounts = false;
    }

    public int[][] getTokensPerTopic() {
        return tokensPerTopic;
    }

    public int[][][] getTypeTopicCounts() {
        return typeTopicCounts;
    }

    public int[] getTablesPerTopic() {
        return tablesPerTopic;
    }

    public int[] getDocLengthCounts() {
        return docLengthCounts;
    }

    public int[][] getTopicDocCounts() {
        return topicDocCounts;
    }

    public double[][][] getPDistr_Mean() {
        return pDistr_Mean;
    }

    public int getMaxTopic() {
        return maxTopic;
    }

//    public double[][][] getPDistr_Var() {
//        return pDistr_Var;
//    }
    public void initializeAlphaStatistics(int size) {
        docLengthCounts = new int[size];
        topicDocCounts = new int[maxNumTopics][size];
    }

    public void collectAlphaStatistics() {
        shouldSaveState = true;
    }

    //Todo: Check if this need.. --> pass by reference @ constructor 
//    public void resetBeta(double[] beta, double[] betaSum) {
//        this.beta = beta;
//        this.betaSum = betaSum;
//    }
    public void resetGamma(double gamma) {
        this.gamma = gamma;

    }

    public void resetAlpha(double alpha) {
        this.alpha = alpha;

    }

    //Todo: Check if this need.. --> pass by reference @ constructor 
    public void resetP_a(double[][] p_a) {
        this.p_a = p_a;
    }

    public void resetP_b(double[][] p_b) {
        this.p_b = p_b;
    }

    public void resetSkewWeight(double[] skewWeight) {
        this.skewWeight = skewWeight;
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
        //numActiveTopics = 0;

        Arrays.fill(tablesPerTopic, 0);
        maxTopic = 0;

        for (byte i = 0; i < numModalities; i++) {

            // Clear the topic totals
            Arrays.fill(tokensPerTopic[i], 0);


            for (int type = 0; type < typeTopicCounts[i].length; type++) {

                int[] topicCounts = typeTopicCounts[i][type];

                int position = 0;
                while (position < topicCounts.length
                        && topicCounts[position] > 0) {
                    topicCounts[position] = 0;
                    position++;
                }
            }
        }

        int[][] localTopicCounts = new int[numModalities][maxTopic];

        for (int doc = startDoc;
                doc < data.size() && doc < startDoc + numDocs;
                doc++) {



            for (byte i = 0; i < numModalities; i++) {
                Arrays.fill(localTopicCounts[i], 0);

                TopicAssignment document = data.get(doc).Assignments[i];
                if (document != null) {
                    FeatureSequence tokens = (FeatureSequence) document.instance.getData();
                    FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;

                    int[] topics = topicSequence.getFeatures();
                    for (int position = 0; position < tokens.size(); position++) {

                        int topic = topics[position];


                        if (topic == ParallelTopicModel.UNASSIGNED_TOPIC) {
                            continue;
                        }

                        tokensPerTopic[i][topic]++;

                        localTopicCounts[i][topic]++;
                        if (localTopicCounts[i][topic] == 1) {
                            tablesPerTopic[topic]++;
                            maxTopic = Math.max(maxTopic, topic);
                        }


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

                            currentTypeTopicCounts[index] =
                                    (1 << topicBits) + topic;
                        } else {
                            currentTypeTopicCounts[index] =
                                    ((currentValue + 1) << topicBits) + topic;

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


        // Record the total number of non-zero topics
        if (!fixedNumTopics) {
            // numActiveTopics = findNumOfActiveTopics();
        }

    }

//    //check all modalities to see whether a topic exists
//    private boolean topicExist(int topic, int[][] tokensPerTopic) {
//        int i = 0;
//        boolean topicFound = false;
//        while (i < numModalities && !topicFound) {
//            if (tokensPerTopic[i][topic] != 0) {
//
//                topicFound = true;
//            }
//            i++;
//
//
//        }
//        return topicFound;
//    }
//    private int findActiveTopics() {
//        int denseIndex = 0;
//
//        for (int topic = 0; topic < maxTopic; topic++) {
//            if (topicExist(topic)) {
//                activeTopics[denseIndex] = topic;
//                denseIndex++;
//            } else {
//                nonActiveTopics.add(topic);
//
//            }
//        }
//        // Record the total number of non-zero topics
//        return denseIndex;
//    }
    //we should also update coeffifients and smoothing 
    private void updateTauAndSmoothing() {
        double[] mk = new double[maxTopic];

        for (int t = 0; t <= maxTopic; t++) {

            //direct assignment scheme: one table per topic per doc per modality
            mk[t] = (double) tablesPerTopic[t];

//			int k = kactive.get(kk);
//			
//			for(int m=0 ; m < numDocuments ; m++){
//				
//				if(nmk[m].get(k) > 1){
//					//sample number of tables
//					mk[kk] +=Samplers.randAntoniak(alpha * tau.get(k), 
//							nmk[m].get(k));
//				}
//				else //nmk[m].get(k) = 0 or 1
//				{   
//					mk[kk] +=nmk[m].get(k);
//				}
//				
//			}
        }// end outter for loop

        //get number of tables
        //tables = Vectors.sum(mk);
        mk[maxTopic + 1] = gamma;

        double[] tt = Samplers.randDir(mk);

        // Initialize the smoothing-only sampling bucket
        Arrays.fill(smoothingOnlyMass, 0d);
        nonActiveTopics.clear();

        for (int topic = 0; topic <= maxTopic + 1; topic++) {

            if (mk[topic] == 0) {
                nonActiveTopics.add(topic);
                tau[topic] = 0;
                for (byte m = 0; m < numModalities; m++) {
                    smoothOnlyCachedCoefficients[m][topic] = 0;
                }

            } else {
                tau[topic] = tt[topic];
                for (byte m = 0; m < numModalities; m++) {
                    // Initialize the cached coefficients, using only smoothing.
                    //  These values will be selectively replaced in documents with
                    //  non-zero counts in particular topics.
                    smoothingOnlyMass[m] += alpha * tau[topic] * beta[m] / (tokensPerTopic[m][topic] + betaSum[m]);
                    smoothOnlyCachedCoefficients[m][topic] = alpha * tau[topic] / (tokensPerTopic[m][topic] + betaSum[m]);

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

            updateTauAndSmoothing();

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

    protected int initSampling(
            MixTopicModelTopicAssignment doc,
            double[][] cachedCoefficients,
            int[][] oneDocTopics,
            FeatureSequence[] tokenSequence,
            int[] docLength,
            int[][] localTopicCounts,
            int[] localTopicIndex,
            double[] topicBetaMass,
            double[][] p) {

        for (byte i = 0; i < numModalities; i++) {
            if (doc.Assignments[i] != null) {
                oneDocTopics[i] = doc.Assignments[i].topicSequence.getFeatures();

                //System.arraycopy(oneDocTopics[i], 0, doc.Assignments[i].topicSequence.getFeatures(), 0, doc.Assignments[i].topicSequence.getFeatures().length-1);


                tokenSequence[i] = ((FeatureSequence) doc.Assignments[i].instance.getData());


                docLength[i] = tokenSequence[i].getLength();

                //		populate topic counts
                for (int position = 0; position < docLength[i]; position++) {
                    if (oneDocTopics[i][position] == ParallelTopicModel.UNASSIGNED_TOPIC) {
                        continue;
                    }
                    localTopicCounts[i][oneDocTopics[i][position]]++;


                }
            }
        }
        // Build an array that densely lists the topics that
        //  have non-zero counts.
        int denseIndex = 0;
        for (int topic = 0; topic < maxTopic; topic++) {
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

        //		Initialize the topic count/beta sampling bucket

        Arrays.fill(topicBetaMass, 0);
        for (byte i = 0; i < numModalities; i++) {
            Arrays.fill(cachedCoefficients[i], 0);

        }
        // Initialize cached coefficients and the topic/beta 
        //  normalizing constant.

        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
            for (byte i = 0; i < numModalities; i++) {
                int topic = localTopicIndex[denseIndex];
                //double normSumN = 0;
                for (byte j = 0; j < numModalities; j++) {
                    if (docLength[j] > 0) {
                        //	initialize the normalization constant for the (B * n_{t|d}) term
                        double normSumN = p[i][j] * localTopicCounts[j][topic] /// docLength[j] * 1000
                                / (docLength[j] * (tokensPerTopic[i][topic] + betaSum[i]));
                        /// (tokensPerTopic[i][topic] + betaSum[i]);

                        topicBetaMass[i] += beta[i] * normSumN;
                        //	update the coefficients for the non-zero topics
                        cachedCoefficients[i][topic] += normSumN;

                    }
                }

                if (Double.isNaN(topicBetaMass[i])) {
                    topicBetaMass[i] = 0;
                }
                if (Double.isNaN(cachedCoefficients[i][topic])) {
                    cachedCoefficients[i][topic] = 0;
                }
            }
        }
        return nonZeroTopics;
    }

    protected int removeOldTopicContribution(
            double[][] cachedCoefficients,
            int[][] localTopicCounts,
            int[] localTopicIndex,
            double[] topicBetaMass,
            int oldTopic,
            int nonZeroTopics,
            final int[] docLength,
            byte m //modality
            ) {

        if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {
            //	Remove this token from all counts. 

            // Remove this topic's contribution to the 
            //  normalizing constants
            smoothingOnlyMass[m] -= alpha * tau[oldTopic] * beta[m]
                    / (tokensPerTopic[m][oldTopic] + betaSum[m]);

            double tmp = localTopicCounts[m][oldTopic]// / docLength[m] * 1000
                    / (docLength[m] * (tokensPerTopic[m][oldTopic] + betaSum[m]));
            /// (tokensPerTopic[m][oldTopic] + betaSum[m]);
            topicBetaMass[m] -= beta[m] * tmp;
            cachedCoefficients[m][oldTopic] -= tmp;
            // Decrement the local doc/topic counts

            localTopicCounts[m][oldTopic]--;

            // Maintain the dense index, if we are deleting
            //  the old topic
            boolean isDeletedTopic = localTopicCounts[m][oldTopic] == 0;

            if (isDeletedTopic) {
                //we have a table per modality thus we should decrease related tables num
                totalTables--;
                tablesPerTopic[oldTopic]--;

                //check if this topic should be deleted from document
                byte j = 0;
                while (isDeletedTopic && j < numModalities) {

                    if (j != m) { //do not check m twice
                        if (localTopicCounts[j][oldTopic] > 0) {
                            isDeletedTopic = false;

                        }
                    }
                    j++;
                }

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
                            localTopicIndex[denseIndex] =
                                    localTopicIndex[denseIndex + 1];
                        }
                        denseIndex++;
                    }
                    nonZeroTopics--;
                }


            }

            // Decrement the global topic count totals
            tokensPerTopic[m][oldTopic]--;
            assert (tokensPerTopic[m][oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";

            if (tokensPerTopic[m][oldTopic] == 0) {
                if (tablesPerTopic[oldTopic] == 0) { //globally last topic

                    nonActiveTopics.add(oldTopic);
                    tau[oldTopic] = 0;
                    maxTopic = Math.max(oldTopic - 1, maxTopic);
                    for (byte mm = 0; mm < numModalities; mm++) {
                        smoothOnlyCachedCoefficients[mm][oldTopic] = 0;
                    }

                }
            }

            // Add the old topic's contribution back into the
            //  normalizing constants.
            smoothingOnlyMass[m] += alpha * tau[oldTopic] * beta[m]
                    / (tokensPerTopic[m][oldTopic] + betaSum[m]);

            topicBetaMass[m] += beta[m] * localTopicCounts[m][oldTopic] /// docLength[m] * 1000
                    / (docLength[m] * (tokensPerTopic[m][oldTopic] + betaSum[m]));
            /// (tokensPerTopic[m][oldTopic] + betaSum[m]);

            // Reset the cached coefficient for this topic
            cachedCoefficients[m][oldTopic] += localTopicCounts[m][oldTopic]// / docLength[m]  * 1000
                    / (docLength[m] * (tokensPerTopic[m][oldTopic] + betaSum[m]));
            /// (tokensPerTopic[m][oldTopic] + betaSum[m]);

        }
        return nonZeroTopics;

    }

    protected double calcTopicScores(
            double[][] cachedCoefficients,
            int oldTopic,
            byte m,
            double[] topicTermScores,
            int[] currentTypeTopicCounts,
            int[] docLength,
            double termSkew) {
        // Now go over the type/topic counts, decrementing
        //  where appropriate, and calculating the score
        //  for each topic at the same time.

        final double[][] smoothOnlyCachedCoefficientsLcl = this.smoothOnlyCachedCoefficients;
        int index = 0;
        int currentTopic, currentValue;
        double score;

        boolean alreadyDecremented = (oldTopic == ParallelTopicModel.UNASSIGNED_TOPIC);

        double topicTermMass = 0.0;

        while (index < currentTypeTopicCounts.length
                && currentTypeTopicCounts[index] > 0) {

            currentTopic = currentTypeTopicCounts[index] & topicMask;
            currentValue = currentTypeTopicCounts[index] >> topicBits;

            if (!alreadyDecremented
                    && currentTopic == oldTopic) {

                // We're decrementing and adding up the 
                //  sampling weights at the same time, but
                //  decrementing may require us to reorder
                //  the topics, so after we're done here,
                //  look at this cell in the array again.

                currentValue--;
                if (currentValue == 0) {
                    currentTypeTopicCounts[index] = 0;
                } else {
                    currentTypeTopicCounts[index] =
                            (currentValue << topicBits) + oldTopic;
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

                // re scale topic term scores (probability mass related to token/label type)
                //types having large skew--> not ver discriminative. Thus I decrease their probability mass
                // skewWeight is used for normalization. Thus the total probability mass (topic term scores) related to types remains almost constant
                // but is share based on type skewness promoting types that are discriminative
                double skewInx = skewWeight[m] * (1 + termSkew); //1;
                //if (!ignoreSkewness) {
                //skewInx = skewWeight[m] * (1 + termSkew);
                // }

                //add normalized smoothingOnly coefficient 
                score =
                        (cachedCoefficients[m][currentTopic] + (smoothOnlyCachedCoefficientsLcl[m][currentTopic] / docLength[m])) * currentValue * skewInx;

                topicTermMass += score;
                topicTermScores[index] = score;

                index++;
            }
        }

        return topicTermMass;
    }

    protected int findNewTopicInTopicTermMass(
            final double[] topicTermScores,
            int[] currentTypeTopicCounts,
            double sample) {

        int newTopic = -1;
        int currentValue;
        int i = -1;
        while (sample > 0) {
            i++;
            sample -= topicTermScores[i];
        }

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
        return newTopic;
    }

    protected int findNewTopicInBetaMass(
            final int[][] localTopicCounts,
            final int[] localTopicIndex,
            int nonZeroTopics,
            byte m,
            final int[] docLength,
            double sample,
            double[][] p) {

        // sample /= beta[m];

        int topic = -1;
        for (int denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {

            topic = localTopicIndex[denseIndex];

            for (byte j = 0; j < numModalities; j++) {
                if (docLength[j] > 0) {
                    sample -= p[m][j] * beta[m] * localTopicCounts[j][topic] /// docLength[j]  * 1000
                            / (docLength[j] * (tokensPerTopic[m][topic] + betaSum[m]));
                    /// (tokensPerTopic[m][topic] + betaSum[m]);

                }
            }
//                                sample -= normSumN
//                                        / (tokensPerTopic[m][topic] + betaSum[m]);

            if (sample <= 0) {

                break;
            }
        }
//       
        if (sample > 0) {
            return topic; // error in rounding (?) I should find a solution 
        }
        return topic;
    }

    protected int findNewTopicInSmoothingMass(
            double sample,
            byte m,
            final int[] docLength) {

        int newTopic = -1;
        sample *= docLength[m];
        //sample /= beta[m];

        int topic = 0;
        sample -= alpha * tau[topic] * beta[m]
                / (tokensPerTopic[m][topic] + betaSum[m]);


        while (sample > 0.0 && topic < maxTopic) {

            topic++;
            sample -= alpha * tau[topic] * beta[m]
                    / (tokensPerTopic[m][topic] + betaSum[m]);


            if (sample <= 0.0) {
                newTopic = topic;
                break;
            }
        }



        return newTopic;
    }

    protected int findNewTopic(
            final int[][] localTopicCounts,
            final int[] localTopicIndex,
            final double[] topicBetaMass,
            int nonZeroTopics,
            byte m,
            double topicTermMass,
            final double[] topicTermScores,
            int[] currentTypeTopicCounts,
            final int[] docLength,
            double sample,
            double[][] p) {
        //	Make sure it actually gets set

        int newTopic = -1;
        //int index = 0;

        double origSample = sample;
        if (sample < topicTermMass) {
            newTopic = findNewTopicInTopicTermMass(
                    topicTermScores,
                    currentTypeTopicCounts,
                    sample);
        } else {
            sample -= topicTermMass;

            if (sample < topicBetaMass[m]) {
                //betaTopicCount++;
                newTopic = findNewTopicInBetaMass(
                        localTopicCounts,
                        localTopicIndex,
                        nonZeroTopics,
                        m,
                        docLength,
                        sample,
                        p);

            } else {
                //smoothingOnlyCount++;
                //smoothingOnlyMass[i] += alpha*tau[topic] * beta[i] / (tokensPerTopic[i][topic] + betaSum[i]);

                sample -= topicBetaMass[m];
                if (sample < smoothingOnlyMass[m]) {
                    newTopic = findNewTopicInSmoothingMass(sample, m, docLength);
                } else {

                    if (nonActiveTopics.peek() != null) {
                        newTopic = nonActiveTopics.poll();
                        maxTopic = Math.max(newTopic, maxTopic);
                    } else {
                        newTopic = maxTopic + 1;
                        maxTopic = newTopic;
                    }

                }
            }
        }

        if (newTopic == -1 || newTopic > maxTopic - 1) {
            System.err.println("WorkerRunnable sampling error: " + origSample + " " + sample + " " + smoothingOnlyMass[m] + " "
                    + topicBetaMass[m] + " " + topicTermMass);
            newTopic = maxTopic - 1; // TODO is this appropriate
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
            if (index == currentTypeTopicCounts.length) {
                System.err.println("error in findind new poisition for topic: " + newTopic);
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
            double[][] cachedCoefficients,
            int[][] localTopicCounts,
            int[] localTopicIndex,
            double[] topicBetaMass,
            int nonZeroTopics,
            final int[] docLength,
            byte m) {
        //			Put that new topic into the counts
        oneDocTopics[m][position] = newTopic;

        smoothingOnlyMass[m] -= alpha * tau[newTopic] * beta[m]
                / (tokensPerTopic[m][newTopic] + betaSum[m]);

        topicBetaMass[m] -= beta[m] * localTopicCounts[m][newTopic]// / docLength[m]  * 1000
                / (docLength[m] * (tokensPerTopic[m][newTopic] + betaSum[m]));
        /// (tokensPerTopic[m][newTopic] + betaSum[m]);

        cachedCoefficients[m][newTopic] -= localTopicCounts[m][newTopic] /// docLength[m]  * 1000
                / (docLength[m] * (tokensPerTopic[m][newTopic] + betaSum[m]));
        /// (tokensPerTopic[m][newTopic] + betaSum[m]);

        localTopicCounts[m][newTopic]++;

        // If this is a new topic for this document,
        //  add the topic to the dense index.

        boolean isNewTopic = (localTopicCounts[m][newTopic] == 1); //new in Document / modality
        if (isNewTopic) {

            totalTables++;
            tablesPerTopic[newTopic]++;

            byte j = 0;
            while (isNewTopic && j < numModalities) {
                if (j != m) {
                    if (localTopicCounts[j][newTopic] > 0) {
                        isNewTopic = false;
                    }
                }
                j++;
            }


            if (isNewTopic) { //new in Document

                // First find the point where we 
                //  should insert the new topic by going to
                //  the end (which is the only reason we're keeping
                //  track of the number of non-zero
                //  topics) and working backwards

                int denseIndex = nonZeroTopics;

                while (denseIndex > 0
                        && localTopicIndex[denseIndex - 1] > newTopic) {

                    localTopicIndex[denseIndex] =
                            localTopicIndex[denseIndex - 1];
                    denseIndex--;
                }

                localTopicIndex[denseIndex] = newTopic;
                nonZeroTopics++;
            }
        }

        tokensPerTopic[m][newTopic]++;

        if (tablesPerTopic[newTopic] == 1) //new topic in corpus 
        {
            updateTauAndSmoothing();
        } else //	update the coefficients for the non-zero topics
        {
            smoothingOnlyMass[m] += alpha * tau[newTopic] * beta[m]
                    / (tokensPerTopic[m][newTopic] + betaSum[m]);
        }

        topicBetaMass[m] += beta[m] * localTopicCounts[m][newTopic]// / docLength[m]  * 1000
                / (docLength[m] * (tokensPerTopic[m][newTopic] + betaSum[m]));
        /// (tokensPerTopic[m][newTopic] + betaSum[m]);

        cachedCoefficients[m][newTopic] += localTopicCounts[m][newTopic]// / docLength[m]  * 1000
                / (docLength[m] * (tokensPerTopic[m][newTopic] + betaSum[m]));
        /// (tokensPerTopic[m][newTopic] + betaSum[m]);

        return nonZeroTopics;
    }

    protected void sampleTopicsForOneDoc(int docCnt) {

        MixTopicModelTopicAssignment doc = data.get(docCnt);
        double[][] cachedCoefficients;
        cachedCoefficients = new double[numModalities][maxNumTopics];// Conservative allocation... [nonZeroTopics + 10]; //we want to avoid dynamic memory allocation , thus we think that we will not have more than ten new  topics in each run
        int[][] oneDocTopics = new int[numModalities][];
        FeatureSequence[] tokenSequence = new FeatureSequence[numModalities];

        int[] docLength = new int[numModalities];
        int[][] localTopicCounts = new int[numModalities][maxNumTopics];
        int[] localTopicIndex = new int[maxNumTopics]; //dense topic index for all modalities
        int type, oldTopic, newTopic;
        double[] topicBetaMass = new double[numModalities];

        //gnu.trove.TObjectIntHashMap<Long> topicPerPrvTopic = new gnu.trove.TObjectIntHashMap<Long>();

        //gnu.trove.TObjectIntHashMap<MassValue> similarGroups = new gnu.trove.TObjectIntHashMap<Integer>();



        //init modalities correlation
        double[][] p = new double[numModalities][numModalities];

        for (byte i = 0; i < numModalities; i++) {
            // Arrays.fill( p[i], 1);
            for (byte j = i; j < numModalities; j++) {

                double pRand = i == j ? 1.0 : ((double) Math.round(1000 * random.nextBeta(p_a[i][j], p_b[i][j])) / (double) 1000);
                p[i][j] = pRand;
                p[j][i] = pRand;

            }


        }

        //FeatureSequence tokens = (FeatureSequence) document.instance.getData();
        //FeatureSequence topicSequence =  (FeatureSequence) document.topicSequence;/
        int nonZeroTopics = initSampling(
                doc,
                cachedCoefficients,
                oneDocTopics,
                tokenSequence,
                docLength,
                localTopicCounts,
                localTopicIndex,
                topicBetaMass,
                p);


        //int[] topicTermIndices;
        //int[] topicTermValues;
        //int i;


        //int[] currentTypeTopicCounts;
        //	Iterate over the positions (words) in the document for each modality
        for (byte m = 0; m < numModalities; m++) // byte m = 0;
        {
            final FeatureSequence tokenSequenceCurMod = tokenSequence[m];
            for (int position = 0; position < docLength[m]; position++) {

                if (tokenSequenceCurMod != null) {

                    //long tmpPreviousTopics = doc.Assignments[m].prevTopicsSequence[position];

                    type = tokenSequenceCurMod.getIndexAtPosition(position);

                    oldTopic = oneDocTopics[m][position];

                    int[] currentTypeTopicCounts = typeTopicCounts[m][type];
                    //int[] currentTypeTopicCounts = new int[typeTopicCounts[m][type].length]; //typeTopicCounts[m][type];
                    //System.arraycopy(typeTopicCounts[m][type], 0, currentTypeTopicCounts, 0, typeTopicCounts[m][type].length-1);

                    nonZeroTopics = removeOldTopicContribution(
                            cachedCoefficients,
                            localTopicCounts,
                            localTopicIndex,
                            topicBetaMass,
                            oldTopic,
                            nonZeroTopics,
                            docLength,
                            m //modality)
                            );


//                    if (topicPerPrvTopic.contains(tmpPreviousTopics) && tmpPreviousTopics > 0 && fastSampling) {
//                        
//                        newTopic = topicPerPrvTopic.get(tmpPreviousTopics);
//                        //System.out.println("common topic sequence found");
//                    } else 
                    {

                        double[] topicTermScores = new double[maxNumTopics];
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

                        double newTopicMass = alpha * tau[maxTopic + 1] / (docLength[m] * numTypes[m]);

                        double sample = ThreadLocalRandom.current().nextDouble() * (newTopicMass + (smoothingOnlyMass[m] / docLength[m])
                                + topicBetaMass[m] + topicTermMass);


                        //random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);

                        //double origSample = sample;

                        newTopic = //random.nextInt(maxNumTopics);

                                findNewTopic(
                                localTopicCounts,
                                localTopicIndex,
                                topicBetaMass,
                                nonZeroTopics,
                                m,
                                topicTermMass,
                                topicTermScores,
                                currentTypeTopicCounts,
                                docLength,
                                sample,
                                p);
                    }


//                    if (!topicPerPrvTopic.contains(tmpPreviousTopics)) {
//                        topicPerPrvTopic.put(tmpPreviousTopics, newTopic);
//                    }
//                    tmpPreviousTopics = tmpPreviousTopics >> topicBits;
//                    long newTopicTmp = (long) newTopic << (63 - topicBits); //long is signed
//                    tmpPreviousTopics += newTopicTmp;
//
//                    doc.Assignments[m].prevTopicsSequence[position] = tmpPreviousTopics; //doc.Assignments[m].prevTopicsSequence[position] >> topicBits;

                    //doc.Assignments[m].prevTopicsSequence[position] = doc.Assignments[m].prevTopicsSequence[position] + newTopicTmp;

                    //assert(newTopic != -1);
                    nonZeroTopics = updateTopicCounts(
                            oneDocTopics,
                            position,
                            newTopic,
                            cachedCoefficients,
                            localTopicCounts,
                            localTopicIndex,
                            topicBetaMass,
                            nonZeroTopics,
                            docLength,
                            m);

                    //statistics for p optimization


                    for (byte i = (byte) (m - 1); i >= 0; i--) {
//                        
//                        if (localTopicCounts[i][newTopic] == 0)
//                        {
//                            System.out.println("Modality not related");
//                            
//                        }

                        pDistr_Mean[m][i][docCnt] += (localTopicCounts[i][newTopic] > 0 ? 1.0 : 0d) / (double) docLength[m];
                        pDistr_Mean[i][m][docCnt] = pDistr_Mean[m][i][docCnt];
                        //pDistr_Var[m][i][docCnt]+= localTopicCounts[i][newTopic]/docLength[m];
                    }


                }


            }


            if (shouldSaveState) {
                // Update the document-topic count histogram,
                //  for dirichlet estimation
                docLengthCounts[ docLength[m]]++;

                for (int denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
                    int topic = localTopicIndex[denseIndex];
                    topicDocCounts[topic][ localTopicCounts[m][topic]]++;
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
//                    alpha*tau[topic] / (tokensPerTopic[topic] + betaSum);
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
//        int[][] localTopicCounts = new int[numModalities][maxNumTopics];
//        int[] localTopicIndex = new int[maxNumTopics]; //dense topic index for all modalities
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
//        for (int topic = 0; topic < maxNumTopics; topic++) {
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
//        cachedCoefficients = new double[numModalities][maxNumTopics];// Conservative allocation... [nonZeroTopics + 10]; //we want to avoid dynamic memory allocation , thus we think that we will not have more than ten new  topics in each run
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
//        double[] topicTermScores = new double[maxNumTopics];
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
//                        smoothingOnlyMass[m] -= alpha*tau[oldTopic] * beta[m]
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
//                        smoothingOnlyMass[m] += alpha*tau[oldTopic] * beta[m]
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
//                            //smoothingOnlyMass[i] += alpha*tau[topic] * beta[i] / (tokensPerTopic[i][topic] + betaSum[i]);
//
//                            sample -= topicBetaMass[m];
//
//                            // sample *= docLength[m];
//                            sample /= beta[m];
//
//                            int topic = 0;
//                            sample -= alpha*tau[topic]
//                                    / (tokensPerTopic[m][topic] + betaSum[m]);
//
//                            while (sample > 0.0 && topic < maxNumTopics - 1) {
//                                topic++;
//                                sample -= alpha*tau[topic]
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
//                        newTopic = maxNumTopics - 1; // TODO is this appropriate
//                        //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
//                    }
//                    //assert(newTopic != -1);
//
//                    //			Put that new topic into the counts
//                    oneDocTopics[m][position] = newTopic;
//
//                    smoothingOnlyMass[m] -= alpha*tau[newTopic] * beta[m]
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
//                    smoothingOnlyMass[m] += alpha*tau[newTopic] * beta[m]
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
////                    alpha*tau[topic] / (tokensPerTopic[topic] + betaSum);
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
//        int[][] localTopicCounts = new int[numModalities][maxNumTopics];
//        int[] localTopicIndex = new int[maxNumTopics]; //dense topic index for all modalities
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
//        for (int topic = 0; topic < maxNumTopics; topic++) {
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
//        cachedCoefficients = new double[numModalities][maxNumTopics];// Conservative allocation... [nonZeroTopics + 10]; //we want to avoid dynamic memory allocation , thus we think that we will not have more than ten new  topics in each run
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
//        double[] topicTermScores = new double[maxNumTopics];
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
//                        smoothingOnlyMass[m] -= alpha*tau[oldTopic] * beta[m]
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
//                        smoothingOnlyMass[m] += alpha*tau[oldTopic] * beta[m]
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
//                            //smoothingOnlyMass[i] += alpha*tau[topic] * beta[i] / (tokensPerTopic[i][topic] + betaSum[i]);
//
//                            sample -= topicBetaMass[m];
//
//                            sample *= docLength[m];
//                            sample /= beta[m];
//
//                            int topic = 0;
//                            sample -= alpha*tau[topic]
//                                    / (tokensPerTopic[m][topic] + betaSum[m]);
//
//                            while (sample > 0.0 && topic < maxNumTopics - 1) {
//                                topic++;
//                                sample -= alpha*tau[topic]
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
//                        newTopic = maxNumTopics - 1; // TODO is this appropriate
//                        //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
//                    }
//                    //assert(newTopic != -1);
//
//                    //			Put that new topic into the counts
//                    oneDocTopics[m][position] = newTopic;
//
//                    smoothingOnlyMass[m] -= alpha*tau[newTopic] * beta[m]
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
//                    smoothingOnlyMass[m] += alpha*tau[newTopic] * beta[m]
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
////                    alpha*tau[topic] / (tokensPerTopic[topic] + betaSum);
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
//        int[] localTopicCounts = new int[maxNumTopics];
//        int[] localTopicIndex = new int[maxNumTopics]; //dense topic index for all modalities
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
//        for (int topic = 0; topic < maxNumTopics; topic++) {
//            if (localTopicCounts[topic] != 0) {
//                localTopicIndex[denseIndex] = topic;
//                denseIndex++;
//            }
//        }
//
//        // Record the total number of non-zero topics
//        int nonZeroTopics = denseIndex;
//        // cachedCoefficients = new double[numModalities][maxNumTopics];// Conservative allocation... [nonZeroTopics + 10]; //we want to avoid dynamic memory allocation , thus we think that we will not have more than ten new  topics in each run
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
//            //cachedCoefficients[topic] =	(alpha*tau[topic] + n) / (tokensPerTopic[topic] + betaSum);
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
//        double[] topicTermScores = new double[maxNumTopics];
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
//                smoothingOnlyMass[0] -= alpha*tau[oldTopic] * beta[0]
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
//                smoothingOnlyMass[0] += alpha*tau[oldTopic] * beta[0]
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
//                    //smoothingOnlyMass[i] += alpha*tau[topic] * beta[i] / (tokensPerTopic[i][topic] + betaSum[i]);
//
//                    sample -= topicBetaMass;
//
//                    //sample *= docLength[0];
//                    sample /= beta[0];
//
//                    int topic = 0;
//                    sample -= alpha*tau[topic]
//                            / (tokensPerTopic[0][topic] + betaSum[0]);
//
//                    while (sample > 0.0 && topic < maxNumTopics - 1) {
//                        topic++;
//                        sample -= alpha*tau[topic]
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
//                newTopic = maxNumTopics - 1; // TODO is this appropriate
//                //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
//            }
//            //assert(newTopic != -1);
//
//            //			Put that new topic into the counts
//            oneDocTopics[position] = newTopic;
//
//            smoothingOnlyMass[0] -= alpha*tau[newTopic] * beta[0]
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
//            smoothingOnlyMass[0] += alpha*tau[newTopic] * beta[0]
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
////                    alpha*tau[topic] / (tokensPerTopic[topic] + betaSum);
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
//        int[] localTopicCounts = new int[maxNumTopics];
//        int[] localTopicIndex = new int[maxNumTopics];
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
//        for (int topic = 0; topic < maxNumTopics; topic++) {
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
//            smoothOnlyCachedCoefficients[0][topic] = (alpha*tau[topic] + n) / (tokensPerTopic[0][topic] + betaSum[0]);
//        }
//
//        double topicTermMass = 0.0;
//
//        double[] topicTermScores = new double[maxNumTopics];
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
//                smoothingOnlyMass[0] -= alpha*tau[oldTopic] * beta[0]
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
//                smoothingOnlyMass[0] += alpha*tau[oldTopic] * beta[0]
//                        / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//                topicBetaMass += beta[0] * localTopicCounts[oldTopic]
//                        / (tokensPerTopic[0][oldTopic] + betaSum[0]);
//
//                // Reset the cached coefficient for this topic
//                smoothOnlyCachedCoefficients[0][oldTopic] =
//                        (alpha*tau[oldTopic] + localTopicCounts[oldTopic])
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
//                    sample -= alpha*tau[newTopic]
//                            / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//                    while (sample > 0.0) {
//                        newTopic++;
//                        sample -= alpha*tau[newTopic]
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
//                newTopic = maxNumTopics - 1; // TODO is this appropriate
//                //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
//            }
//            //assert(newTopic != -1);
//
//            //			Put that new topic into the counts
//            oneDocTopics[position] = newTopic;
//
//            smoothingOnlyMass[0] -= alpha*tau[newTopic] * beta[0]
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
//                    (alpha*tau[newTopic] + localTopicCounts[newTopic])
//                    / (tokensPerTopic[0][newTopic] + betaSum[0]);
//
//            smoothingOnlyMass[0] += alpha*tau[newTopic] * beta[0]
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
//                    alpha*tau[topic] / (tokensPerTopic[0][topic] + betaSum[0]);
//        }
//
//    }
}
