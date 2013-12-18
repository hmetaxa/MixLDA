/* Copyright (C) 2013 Omiros Metaxas */
package cc.mallet.topics;

import java.util.Arrays;
import java.util.ArrayList;

import java.util.zip.*;

import java.io.*;
import java.text.NumberFormat;

import cc.mallet.types.*;
import cc.mallet.util.Randoms;

/**
 * A parallel semi supervised topic model runnable task.
 *
 * @author Omiros Metaxas Based on MALLET Parallel topic model of author David
 * Mimno, Andrew McCallum test BOX sync
 *
 */
public class MirrorWorkerRunnable implements Runnable {

    public class MassValue {

        public double topicTermMass;
        public double topicBetaMass;
        public double smoothingOnlyMass;
        //public int nonZeroTopics;
        //public double few;//frequency exclusivity weight we have an array for that
    }
    boolean isFinished = true;
    boolean ignoreLabels = false;
    boolean ignoreSkewness = false;
    ArrayList<TopicAssignment> data;
    int startDoc, numDocs;
    protected int numTopics; // Number of topics to be fit
    // These values are used to encode type/topic counts as
    //  count/topic pairs in a single int.
    protected int topicMask;
    protected int topicBits;
    protected int numTypes;
    protected double avgTypeCount; //not used for now
    protected int[] typeTotals; //not used for now
    //homer
    protected int numLblTypes;
    protected double avgLblTypeCount; //not used for now
    protected int[] lblTypeTotals;  //not used for now
    protected double[] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
    protected double alphaSum;
    protected double beta;   // Prior on per-topic multinomial distribution over words
    protected double betaSum;
    public static final double DEFAULT_BETA = 0.01;
    //homer 
    protected double gamma;   // Prior on per-topic multinomial distribution over labels
    protected double gammaSum;
    public static final double DEFAULT_GAMMA = 0.01;
    protected double smoothingOnlyMass = 0.0;
    protected double[] cachedCoefficients;
    protected double smoothingOnlyLabelMass = 0.0;
    protected double[] cachedLabelCoefficients;
    protected int[][] typeTopicCounts; // indexed by <feature index, topic index>
    protected int[] tokensPerTopic; // indexed by <topic index>
    //homer 
    protected int[][] lbltypeTopicCounts; // indexed by <label index, topic index>
    protected int[] labelsPerTopic; // indexed by <topic index>
    // for dirichlet estimation
    protected int[] docLengthCounts; // histogram of document sizes
    protected int[][] topicDocCounts; // histogram of document/topic counts, indexed by <topic index, sequence position index>
    protected int[] docLblLengthCounts; // histogram of document sizes
    protected int[][] topicLblDocCounts; // histogram of document/topic counts, indexed by <topic index, sequence position index>
    boolean shouldSaveState = false;
    boolean shouldBuildLocalCounts = true;
    protected Randoms random;
    // The skew index of eachType
    protected double[] typeSkewIndexes;
    // The skew index of each Lbl Type
    protected double[] lblTypeSkewIndexes;
    protected double skewWeight = 1;
    protected double lblSkewWeight = 1;
    double lblWeight = 1;
    //double avgSkew = 0;

    public MirrorWorkerRunnable(int numTopics,
            double[] alpha, double alphaSum,
            double beta, double gamma, Randoms random,
            ArrayList<TopicAssignment> data,
            int[][] typeTopicCounts,
            int[] tokensPerTopic,
            int[][] lbltypeTopicCounts,
            int[] labelsPerTopic,
            int startDoc, int numDocs, boolean ignoreLabels,
            double avgTypeCount, int[] typeTotals,
            double avgLblTypeCount, int[] lblTypeTotals,
            double[] typeSkewIndexes, double[] lblTypeSkewIndexes, boolean ignoreSkewness, double skewWeight, double lblSkewWeight) {

        this.data = data;

        this.numTopics = numTopics;
        this.numTypes = typeTopicCounts.length;
        this.numLblTypes = lbltypeTopicCounts.length;
        this.ignoreSkewness = ignoreSkewness;
        this.skewWeight = skewWeight;
        this.lblSkewWeight = lblSkewWeight;

        lblWeight = numTypes / numLblTypes;

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

        this.lbltypeTopicCounts = lbltypeTopicCounts;
        this.labelsPerTopic = labelsPerTopic;


        this.alphaSum = alphaSum;
        this.alpha = alpha;
        this.beta = beta;
        this.betaSum = beta * numTypes;
        this.gamma = gamma;
        this.gammaSum = gamma * numLblTypes;
        this.random = random;

        this.startDoc = startDoc;
        this.numDocs = numDocs;
        this.ignoreLabels = ignoreLabels;

        this.avgTypeCount = avgTypeCount;
        this.typeTotals = typeTotals;
        this.avgLblTypeCount = avgLblTypeCount;
        this.lblTypeTotals = lblTypeTotals;

        this.typeSkewIndexes = typeSkewIndexes;
        this.lblTypeSkewIndexes = lblTypeSkewIndexes;


        cachedCoefficients = new double[numTopics];
        cachedLabelCoefficients = new double[numTopics];

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

    //homer
    public int[] getLabelsPerTopic() {
        return labelsPerTopic;
    }

    public int[][] getlblTypeTopicCounts() {
        return lbltypeTopicCounts;
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

        docLblLengthCounts = new int[size];
        topicLblDocCounts = new int[numTopics][size];;
    }

    public void collectAlphaStatistics() {
        shouldSaveState = true;
    }

    public void resetSkewWeight(double skewWeight, double lblSkewWeight) {
        this.skewWeight = skewWeight;
        this.lblSkewWeight = lblSkewWeight;
    }

    public void resetBeta(double beta, double betaSum) {
        this.beta = beta;
        this.betaSum = betaSum;
    }

    public void resetGamma(double gamma, double gammaSum) {
        this.gamma = gamma;
        this.gammaSum = gammaSum;
    }

    /**
     * Once we have sampled the local counts, trash the "global" type topic
     * counts and reuse the space to build a summary of the type topic counts
     * specific to this worker's section of the corpus.
     */
    public void buildLocalTypeTopicCounts() {

        // Clear the topic totals
        Arrays.fill(tokensPerTopic, 0);

        Arrays.fill(labelsPerTopic, 0);

        // Clear the type/topic counts, only 
        //  looking at the entries before the first 0 entry.

        for (int type = 0; type < typeTopicCounts.length; type++) {

            int[] topicCounts = typeTopicCounts[type];

            int position = 0;
            while (position < topicCounts.length
                    && topicCounts[position] > 0) {
                topicCounts[position] = 0;
                position++;
            }
        }

        for (int lbltype = 0; lbltype < lbltypeTopicCounts.length; lbltype++) {

            int[] lbltopicCounts = lbltypeTopicCounts[lbltype];

            int position = 0;
            while (position < lbltopicCounts.length
                    && lbltopicCounts[position] > 0) {
                lbltopicCounts[position] = 0;
                position++;
            }
        }

        for (int doc = startDoc;
                doc < data.size() && doc < startDoc + numDocs;
                doc++) {

            TopicAssignment document = data.get(doc);

            FeatureSequence tokens = (FeatureSequence) document.instance.getData();
            FeatureSequence labels = (FeatureSequence) document.instance.getTarget();
            FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;


            int[] topics = topicSequence.getFeatures();

            for (int position = 0; position < tokens.size(); position++) {

                int topic = topics[position];

                if (topic == ParallelTopicModel.UNASSIGNED_TOPIC) {
                    continue;
                }

                tokensPerTopic[topic]++;

                // The format for these arrays is 
                //  the topic in the rightmost bits
                //  the count in the remaining (left) bits.
                // Since the count is in the high bits, sorting (desc)
                //  by the numeric value of the int guarantees that
                //  higher counts will be before the lower counts.

                int type = tokens.getIndexAtPosition(position);

                int[] currentTypeTopicCounts = typeTopicCounts[ type];

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

            FeatureSequence lblTopicSequence = (FeatureSequence) document.lblTopicSequence;
            int[] lblTopics = lblTopicSequence.getFeatures();

            for (int position = 0; position < labels.size(); position++) {

                int topic = lblTopics[position];

                if (topic == ParallelTopicModel.UNASSIGNED_TOPIC) {
                    continue;
                }

                labelsPerTopic[topic]++;

                // The format for these arrays is 
                //  the topic in the rightmost bits
                //  the count in the remaining (left) bits.
                // Since the count is in the high bits, sorting (desc)
                //  by the numeric value of the int guarantees that
                //  higher counts will be before the lower counts.

                int type = labels.getIndexAtPosition(position);

                int[] currentlblTypeTopicCounts = lbltypeTopicCounts[ type];

                // Start by assuming that the array is either empty
                //  or is in sorted (descending) order.

                // Here we are only adding counts, so if we find 
                //  an existing location with the topic, we only need
                //  to ensure that it is not larger than its left neighbor.

                int index = 0;
                int currentTopic = currentlblTypeTopicCounts[index] & topicMask;
                int currentValue;

                while (currentlblTypeTopicCounts[index] > 0 && currentTopic != topic) {
                    index++;
                    if (index == currentlblTypeTopicCounts.length) {
                        System.out.println("overflow on type " + type);
                    }
                    currentTopic = currentlblTypeTopicCounts[index] & topicMask;
                }
                currentValue = currentlblTypeTopicCounts[index] >> topicBits;

                if (currentValue == 0) {
                    // new value is 1, so we don't have to worry about sorting
                    //  (except by topic suffix, which doesn't matter)

                    currentlblTypeTopicCounts[index] =
                            (1 << topicBits) + topic;
                } else {
                    currentlblTypeTopicCounts[index] =
                            ((currentValue + 1) << topicBits) + topic;

                    // Now ensure that the array is still sorted by 
                    //  bubbling this value up.
                    while (index > 0
                            && currentlblTypeTopicCounts[index] > currentlblTypeTopicCounts[index - 1]) {
                        int temp = currentlblTypeTopicCounts[index];
                        currentlblTypeTopicCounts[index] = currentlblTypeTopicCounts[index - 1];
                        currentlblTypeTopicCounts[index - 1] = temp;

                        index--;
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

            isFinished = false;

            // Initialize the smoothing-only sampling bucket
            smoothingOnlyMass = 0;
            smoothingOnlyLabelMass = 0;



            // Initialize the cached coefficients, using only smoothing.
            //  These values will be selectively replaced in documents with
            //  non-zero counts in particular topics.

            for (int topic = 0; topic < numTopics; topic++) {
                smoothingOnlyMass += alpha[topic] * beta / (tokensPerTopic[topic] + betaSum);
                
                if (ignoreLabels) {
                   // smoothingOnlyMass += alpha[topic] * beta / (tokensPerTopic[topic] + betaSum);
                    cachedCoefficients[topic] = alpha[topic] / (tokensPerTopic[topic] + betaSum);
                } else {
                   // smoothingOnlyMass += (1 + lblWeight) * alpha[topic] * beta / (tokensPerTopic[topic] + betaSum);
                    cachedCoefficients[topic] = (1 + lblWeight) * alpha[topic] / (tokensPerTopic[topic] + betaSum);
                }
                
                smoothingOnlyLabelMass += alpha[topic] * gamma / (labelsPerTopic[topic] + gammaSum);
                
                if (ignoreLabels) {
                   // smoothingOnlyLabelMass += alpha[topic] * gamma / (labelsPerTopic[topic] + gammaSum);
                    cachedLabelCoefficients[topic] = alpha[topic] / (labelsPerTopic[topic] + gammaSum);
                } else {
                   // smoothingOnlyLabelMass += (1 + 1 / lblWeight) * alpha[topic] * gamma / (labelsPerTopic[topic] + gammaSum);
                    cachedLabelCoefficients[topic] = (1 + 1 / lblWeight) * alpha[topic] / (labelsPerTopic[topic] + gammaSum);
                }
            }



            for (int doc = startDoc;
                    doc < data.size() && doc < startDoc + numDocs;
                    doc++) {

                /*
                 if (doc % 10000 == 0) {
                 System.out.println("processing doc " + doc);
                 }
                 */




                /*

                 FeatureSequence tokenSequence =
                 (FeatureSequence) data.get(doc).instance.getData();

                 LabelSequence topicSequence =
                 (LabelSequence) data.get(doc).topicSequence;

                 LabelSequence lblTopicSequence =
                 (LabelSequence) data.get(doc).lblTopicSequence;

                 FeatureSequence labelSequence =
                 (FeatureSequence) data.get(doc).instance.getTarget();*/


                sampleTopicsForOneDoc(data.get(doc));
                //typeTopicCounts);
                //, cachedCoefficients, tokensPerTopic, betaSum, beta, smoothingOnlyMass,
                //lbltypeTopicCounts, cachedLabelCoefficients, labelsPerTopic, gammaSum, gamma, smoothingOnlyLabelMass);

                //  sampleTopicsForOneDoc(tokenSequence, topicSequence,
                //          true, typeTopicCounts, cachedCoefficients, tokensPerTopic, betaSum, beta, smoothingOnlyMass);

                //homer sample labels now


                // sampleTopicsForOneDoc(labelSequence, topicSequence,
                //         true, lbltypeTopicCounts, cachedLabelCoefficients, labelsPerTopic, gammaSum, gamma, smoothingOnlyLabelMass);
                //homer 

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

    //frequency & exclusivity weight
    double calcFEW(
            int[] currentTypeTopicCounts,
            int currentTotalTypeCount,
            int maxTypeCount) {


        int index = 0;
        //int currentTopic, currentValue;
        double skewIndex = 0;
        // distinctivines, exclusivity calculation
        while (index < currentTypeTopicCounts.length
                && currentTypeTopicCounts[index] > 0) {
            //currentTopic = currentTypeTopicCounts[index] & topicMask;
            //currentValue = currentTypeTopicCounts[index] >> topicBits;
            skewIndex += Math.pow(currentTypeTopicCounts[index] >> topicBits, 2);
        }
        skewIndex = skewIndex / Math.pow(currentTotalTypeCount, 2);
        // frequency consideration
        skewIndex = currentTotalTypeCount / maxTypeCount * skewIndex;

        return skewIndex;
    }

    protected int removeOldTopicContribution(
            int position,
            int[] oneDocTopics,
            MassValue mass,
            int[] localTopicCounts,
            int[] localLblTopicCounts,
            int[] localTopicIndex,
            double[] cachedCoefficients,
            int[] tokensPerTopic,
            double betaSum,
            double beta,
            double lblWeight,
            int nonZeroTopics) {

        int oldTopic = oneDocTopics[position];


        int denseIndex = 0;


        if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {
            //	Remove this token from all counts. 


            // Remove this topic's contribution to the 
            //  normalizing constants


            if (ignoreLabels) {
                mass.smoothingOnlyMass -= alpha[oldTopic] * beta
                        / (tokensPerTopic[oldTopic] + betaSum);
                mass.topicBetaMass -= beta * localTopicCounts[oldTopic]
                        / (tokensPerTopic[oldTopic] + betaSum);
            } else {

                mass.smoothingOnlyMass -= (1 + lblWeight) * alpha[oldTopic] * beta
                        / (tokensPerTopic[oldTopic] + betaSum);
                mass.topicBetaMass -= beta * (localTopicCounts[oldTopic] + lblWeight * localLblTopicCounts[oldTopic])
                        / (tokensPerTopic[oldTopic] + betaSum);
            }
            // Decrement the local doc/topic counts

            localTopicCounts[oldTopic]--;

            // Maintain the dense index, if we are deleting
            //  the old topic
            if (localTopicCounts[oldTopic] == 0 && localLblTopicCounts[oldTopic] == 0) {

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
                        localTopicIndex[denseIndex] =
                                localTopicIndex[denseIndex + 1];
                    }
                    denseIndex++;
                }

                nonZeroTopics--;
            }

            // Decrement the global topic count totals
            tokensPerTopic[oldTopic]--;
            assert (tokensPerTopic[oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";


            // Add the old topic's contribution back into the
            //  normalizing constants.
            if (ignoreLabels) {
                mass.smoothingOnlyMass += alpha[oldTopic] * beta
                        / (tokensPerTopic[oldTopic] + betaSum);
                mass.topicBetaMass += beta * localTopicCounts[oldTopic]
                        / (tokensPerTopic[oldTopic] + betaSum);

                // Reset the cached coefficient for this topic
                cachedCoefficients[oldTopic] =
                        (alpha[oldTopic] + localTopicCounts[oldTopic])
                        / (tokensPerTopic[oldTopic] + betaSum);
            } else {
                mass.smoothingOnlyMass += (1 + lblWeight) * alpha[oldTopic] * beta
                        / (tokensPerTopic[oldTopic] + betaSum);
                mass.topicBetaMass += beta * (localTopicCounts[oldTopic] + lblWeight * localLblTopicCounts[oldTopic])
                        / (tokensPerTopic[oldTopic] + betaSum);

                // Reset the cached coefficient for this topic
                cachedCoefficients[oldTopic] = ((1 + lblWeight) * alpha[oldTopic] + localTopicCounts[oldTopic] + lblWeight * localLblTopicCounts[oldTopic])
                        / (tokensPerTopic[oldTopic] + betaSum);
            }
        }

        return nonZeroTopics;
    }

    protected void calcSamplingValuesPerType(
            //FeatureSequence tokenSequence,
            int position,
            int[] oneDocTopics,
            MassValue mass,
            double[] topicTermScores,
            int[] currentTypeTopicCounts,
            int[] localTopicCounts,
            //int[] localTopicIndex,
            double[] cachedCoefficients,
            int[] tokensPerTopic,
            double betaSum,
            double beta,
            int[] typeTotals,
            int typeIndex,
            double[] typeSkewIndexes,
            double skewWeight) {

        int oldTopic = oneDocTopics[position];

        // Now go over the type/topic counts, decrementing
        //  where appropriate, and calculating the score
        //  for each topic at the same time.

        int index = 0;
        int currentTopic, currentValue;

        double prevSkew = typeSkewIndexes[typeIndex];
        //typeSkewIndexes[typeIndex] = 0;
        double score;
        boolean alreadyDecremented = (oldTopic == ParallelTopicModel.UNASSIGNED_TOPIC);


        mass.topicTermMass = 0.0;
        //int totalCounts = 0;

        while (index < currentTypeTopicCounts.length
                && currentTypeTopicCounts[index] > 0) {

            currentTopic = currentTypeTopicCounts[index] & topicMask;
            currentValue = currentTypeTopicCounts[index] >> topicBits;
            //totalCounts += currentValue;

//            if (!ignoreSkewness) {
//                typeSkewIndexes[typeIndex] += Math.pow((double) currentValue, 2);
//            }

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
                double skewInx = 1;
                if (!ignoreSkewness) {
                    skewInx = skewWeight * (1 + prevSkew);
                }


                score = cachedCoefficients[currentTopic] * currentValue * skewInx;

                mass.topicTermMass += score;
                topicTermScores[index] = score;

                index++;
            }
        }
        // skewIndex = skewIndex / Math.pow(typeTotals[typeIndex], 2);
        //skewIndex = (double) typeTotals[typeIndex] / (double) avgTypeCount * skewIndex;

        /* UPDATE as an optimization step during Summing
         if (totalCounts == 0) {
         totalCounts = typeTotals[typeIndex];
         }
        
         if (!ignoreSkewness && totalCounts>0) {
         typeSkewIndexes[typeIndex] = typeSkewIndexes[typeIndex] / Math.pow((double) totalCounts, 2);
         }
         * */
    }

    protected int findNewTopic(
            double sample,
            MassValue mass,
            double[] topicTermScores,
            int[] currentTypeTopicCounts,
            int[] localTopicCounts,
            int[] localLblTopicCounts,
            int[] localTopicIndex,
            int[] tokensPerTopic,
            double betaSum,
            double beta,
            int nonZeroTopics,
            double lblWeight) {
        int i, denseIndex;
        int currentValue;
        int newTopic = -1;



        if (sample < mass.topicTermMass) {
            //topicTermCount++;

            i = -1;
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


        } else {
            sample -= mass.topicTermMass;

            if (sample < mass.topicBetaMass) {
                //betaTopicCount++;

                // sample /= beta;

                for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {

                    int topic = localTopicIndex[denseIndex];

                    if (ignoreLabels) {
                        sample -= beta * localTopicCounts[topic]
                                / (tokensPerTopic[topic] + betaSum);
                    } else {

                        if (lblWeight < 1) {
                            int b = 0;
                        }
                        sample -= beta * (localTopicCounts[topic] + lblWeight * localLblTopicCounts[topic])
                                / (tokensPerTopic[topic] + betaSum);
                    }

                    if (sample <= 0.0) {
                        newTopic = topic;
                        break;
                    }
                }

                if (lblWeight < 1 && newTopic == -1) {
                    int a = 0;
                }


            } else {
                //smoothingOnlyCount++;

                sample -= mass.topicBetaMass;

                sample /= beta;

                newTopic = 0;
                if (ignoreLabels) {
                    sample -= alpha[newTopic]
                            / (tokensPerTopic[newTopic] + betaSum);
                } else {
                    sample -= ((1 + lblWeight) * alpha[newTopic])
                            / (tokensPerTopic[newTopic] + betaSum);
                }

                while (sample > 0.0) {
                    newTopic++;
                    if (ignoreLabels) {
                        sample -= alpha[newTopic]
                                / (tokensPerTopic[newTopic] + betaSum);
                    } else {


                        sample -= ((1 + lblWeight) * alpha[newTopic])
                                / (tokensPerTopic[newTopic] + betaSum);
                    }
                }

                //if (sample<0.00001)

            }


            if (newTopic == -1) {
                int b = 0;
            }

            // Move to the position for the new topic,
            //  which may be the first empty position if this
            //  is a new topic for this word.

            int index = 0;
            while (currentTypeTopicCounts[index] > 0
                    && (currentTypeTopicCounts[index] & topicMask) != newTopic) {
                index++;
                if (index == currentTypeTopicCounts.length) {
                    System.err.println("type: " + " new topic: " + newTopic);
                    //System.err.println("type: " + type + " new topic: " + newTopic);
                    for (int k = 0; k < currentTypeTopicCounts.length; k++) {
                        System.err.print((currentTypeTopicCounts[k] & topicMask) + ":"
                                + (currentTypeTopicCounts[k] >> topicBits) + " ");
                    }
                    System.err.println();

                }
            }


            // index should now be set to the position of the new topic,
            //  which may be an empty cell at the end of the list.

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
        return newTopic;
    }

    protected void sampleTopicsForOneDoc(TopicAssignment doc // int[][] typeTopicCounts
            //,
            //double[] cachedCoefficients,
            //int[] tokensPerTopic,
            //double betaSum,
            //double beta,
            //double smoothingOnlyMass,
            //int[][] lblTypeTopicCounts,
            //double[] cachedLabelCoefficients,
            //int[] labelsPerTopic,
            //double gammaSum,
            //double gamma,
            //double smoothingOnlyLblMass
            ) {

        FeatureSequence tokenSequence =
                (FeatureSequence) doc.instance.getData();

        LabelSequence topicSequence =
                (LabelSequence) doc.topicSequence;

        MassValue massValue = new MassValue();
        massValue.topicBetaMass = 0.0;
        massValue.topicTermMass = 0.0;
        massValue.smoothingOnlyMass = smoothingOnlyMass;

        int nonZeroTopics = 0;


        int[] oneDocTopics = topicSequence.getFeatures();
        int[] localTopicCounts = new int[numTopics];
        int[] localTopicIndex = new int[numTopics];

        //Label Init
        LabelSequence lblTopicSequence =
                (LabelSequence) doc.lblTopicSequence;
        FeatureSequence labelSequence =
                (FeatureSequence) doc.instance.getTarget();

        MassValue massLblValue = new MassValue();
        massLblValue.topicBetaMass = 0.0;
        massLblValue.topicTermMass = 0.0;
        massLblValue.smoothingOnlyMass = smoothingOnlyLabelMass;



        int[] oneDocLblTopics = lblTopicSequence.getFeatures();
        int[] localLblTopicCounts = new int[numTopics];


        //initSampling

        int docLength = tokenSequence.getLength();
        //		populate topic counts
        for (int position = 0; position < docLength; position++) {
            if (oneDocTopics[position] == ParallelTopicModel.UNASSIGNED_TOPIC) {
                continue;
            }
            localTopicCounts[oneDocTopics[position]]++;
        }

        docLength = labelSequence.getLength();
        //		populate topic counts
        for (int position = 0; position < docLength; position++) {
            if (oneDocLblTopics[position] == ParallelTopicModel.UNASSIGNED_TOPIC) {
                continue;
            }
            localLblTopicCounts[oneDocLblTopics[position]]++;
        }

        // Build an array that densely lists the topics that
        //  have non-zero counts.
        int denseIndex = 0;
        for (int topic = 0; topic < numTopics; topic++) {
            if (localTopicCounts[topic] != 0 || localLblTopicCounts[topic] != 0) {
                localTopicIndex[denseIndex] = topic;
                denseIndex++;
            }
        }

        // Record the total number of non-zero topics
        nonZeroTopics = denseIndex;
        if (nonZeroTopics < 20) {
            int a = 1;
        }

        //Initialize the topic count/beta sampling bucket
        // Initialize cached coefficients and the topic/beta 
        //  normalizing constant.
        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
            int topic = localTopicIndex[denseIndex];
            int n = localTopicCounts[topic];
            int nl = localLblTopicCounts[topic];

            if (ignoreLabels) {
                //	initialize the normalization constant for the (B * n_{t|d}) term
                massValue.topicBetaMass += beta * n / (tokensPerTopic[topic] + betaSum);
                //massLblValue.topicBetaMass += gamma * nl / (labelsPerTopic[topic] + gammaSum);
                //	update the coefficients for the non-zero topics
                cachedCoefficients[topic] = (alpha[topic] + n) / (tokensPerTopic[topic] + betaSum);
                //cachedLabelCoefficients[topic] = (alpha[topic] + nl) / (labelsPerTopic[topic] + gammaSum);
            } else {
                massValue.topicBetaMass += beta * (n + lblWeight * nl) / (tokensPerTopic[topic] + betaSum);
                //massLblValue.topicBetaMass += gamma * (nl + (1 / lblWeight) * n) / (labelsPerTopic[topic] + gammaSum);
                cachedCoefficients[topic] = ((1 + lblWeight) * alpha[topic] + n + lblWeight * nl) / (tokensPerTopic[topic] + betaSum);
                //cachedLabelCoefficients[topic] = ((1 + (1 / lblWeight)) * alpha[topic] + nl + (1 / lblWeight) * n) / (labelsPerTopic[topic] + gammaSum);
            }


        }

        //end of Init Sampling 


        double[] topicTermScores = new double[numTopics];
        int[] currentTypeTopicCounts;
        //	Iterate over the positions (words) in the document 
        docLength = tokenSequence.getLength();
        for (int position = 0; position < docLength; position++) {


            int type = tokenSequence.getIndexAtPosition(position);
            currentTypeTopicCounts = typeTopicCounts[type];

            nonZeroTopics = removeOldTopicContribution(position, oneDocTopics, massValue, localTopicCounts, localLblTopicCounts,
                    localTopicIndex, cachedCoefficients, tokensPerTopic, betaSum, beta, lblWeight, nonZeroTopics);

            //calcSamplingValuesPerType
            calcSamplingValuesPerType(
                    //tokenSequence,
                    position,
                    oneDocTopics,
                    massValue,
                    topicTermScores,
                    currentTypeTopicCounts,
                    localTopicCounts,
                    //localTopicIndex,
                    cachedCoefficients,
                    tokensPerTopic,
                    betaSum,
                    beta,
                    typeTotals,
                    type,
                    typeSkewIndexes,
                    skewWeight);


            double sample = 0;


            sample = random.nextUniform() * (massValue.smoothingOnlyMass + massValue.topicBetaMass + massValue.topicTermMass);


            double origSample = sample;

            //	Make sure it actually gets set
            int newTopic = -1;


            newTopic = findNewTopic(
                    sample,
                    massValue,
                    topicTermScores,
                    currentTypeTopicCounts,
                    localTopicCounts,
                    localLblTopicCounts,
                    localTopicIndex,
                    tokensPerTopic,
                    betaSum,
                    beta,
                    nonZeroTopics,
                    lblWeight);



            if (newTopic == -1) {
                System.err.println("WorkerRunnable sampling error: " + origSample + " " + sample + " " + massValue.smoothingOnlyMass + " "
                        + massValue.topicBetaMass + " " + massValue.topicTermMass);
                newTopic = numTopics - 1; // TODO is this appropriate
                //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
            }
            //assert(newTopic != -1);

            //			Put that new topic into the counts
            oneDocTopics[position] = newTopic;

            if (ignoreLabels) {
                massValue.smoothingOnlyMass -= alpha[newTopic] * beta
                        / (tokensPerTopic[newTopic] + betaSum);
                massValue.topicBetaMass -= beta * localTopicCounts[newTopic]
                        / (tokensPerTopic[newTopic] + betaSum);
            } else {
                massValue.smoothingOnlyMass -= (1 + lblWeight) * alpha[newTopic] * beta
                        / (tokensPerTopic[newTopic] + betaSum);
                massValue.topicBetaMass -= beta * (localTopicCounts[newTopic] + lblWeight * localLblTopicCounts[newTopic])
                        / (tokensPerTopic[newTopic] + betaSum);
            }
            localTopicCounts[newTopic]++;

            // If this is a new topic for this document,
            //  add the topic to the dense index.
            if (localTopicCounts[newTopic] == 1 && localLblTopicCounts[newTopic] == 0) {

                // First find the point where we 
                //  should insert the new topic by going to
                //  the end (which is the only reason we're keeping
                //  track of the number of non-zero
                //  topics) and working backwards

                denseIndex = nonZeroTopics;

                while (denseIndex > 0 && localTopicIndex[denseIndex - 1] > newTopic) {

                    localTopicIndex[denseIndex] =
                            localTopicIndex[denseIndex - 1];
                    denseIndex--;
                }

                localTopicIndex[denseIndex] = newTopic;
                nonZeroTopics++;
            }

            tokensPerTopic[newTopic]++;

            if (ignoreLabels) {
                //	update the coefficients for the non-zero topics
                cachedCoefficients[newTopic] =
                        (alpha[newTopic] + localTopicCounts[newTopic])
                        / (tokensPerTopic[newTopic] + betaSum);

                massValue.smoothingOnlyMass += alpha[newTopic] * beta
                        / (tokensPerTopic[newTopic] + betaSum);
                massValue.topicBetaMass += beta * localTopicCounts[newTopic]
                        / (tokensPerTopic[newTopic] + betaSum);
            } else {
                massValue.smoothingOnlyMass += (1 + lblWeight) * alpha[newTopic] * beta
                        / (tokensPerTopic[newTopic] + betaSum);
                massValue.topicBetaMass += beta * (localTopicCounts[newTopic] + lblWeight * localLblTopicCounts[newTopic])
                        / (tokensPerTopic[newTopic] + betaSum);

                cachedCoefficients[newTopic] = ((1 + lblWeight) * alpha[newTopic] + localTopicCounts[newTopic] + lblWeight * localLblTopicCounts[newTopic])
                        / (tokensPerTopic[newTopic] + betaSum);
            }
        }


        // sample labels
        // init labels 
        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
            int topic = localTopicIndex[denseIndex];
            int n = localTopicCounts[topic];
            int nl = localLblTopicCounts[topic];

            if (ignoreLabels) {
                //	initialize the normalization constant for the (B * n_{t|d}) term
                //massValue.topicBetaMass += beta * n / (tokensPerTopic[topic] + betaSum);
                massLblValue.topicBetaMass += gamma * nl / (labelsPerTopic[topic] + gammaSum);
                //	update the coefficients for the non-zero topics
                //cachedCoefficients[topic] = (alpha[topic] + n) / (tokensPerTopic[topic] + betaSum);
                cachedLabelCoefficients[topic] = (alpha[topic] + nl) / (labelsPerTopic[topic] + gammaSum);
            } else {
                //massValue.topicBetaMass += beta * (n + lblWeight * nl) / (tokensPerTopic[topic] + betaSum);
                massLblValue.topicBetaMass += gamma * (nl + (1 / lblWeight) * n) / (labelsPerTopic[topic] + gammaSum);
                // cachedCoefficients[topic] = ((1 + lblWeight) * alpha[topic] + n + lblWeight * nl) / (tokensPerTopic[topic] + betaSum);
                cachedLabelCoefficients[topic] = ((1 + (1 / lblWeight)) * alpha[topic] + nl + (1 / lblWeight) * n) / (labelsPerTopic[topic] + gammaSum);
            }


        }

        double[] topicLblTermScores = new double[numTopics];
        int[] currentLblTypeTopicCounts;
        int docLblLength = labelSequence.getLength();

        //	Iterate over the positions (words) in the document 
        for (int position = 0; position < docLblLength; position++) {

            int type = labelSequence.getIndexAtPosition(position);
            currentLblTypeTopicCounts = lbltypeTopicCounts[type];

            nonZeroTopics = removeOldTopicContribution(position, oneDocLblTopics, massLblValue, localLblTopicCounts, localTopicCounts,
                    localTopicIndex, cachedLabelCoefficients, labelsPerTopic, gammaSum, gamma, 1 / lblWeight, nonZeroTopics);

            //calcSamplingValuesPerType
            calcSamplingValuesPerType(
                    //labelSequence,
                    position,
                    oneDocLblTopics,
                    massLblValue,
                    topicLblTermScores,
                    currentLblTypeTopicCounts,
                    localLblTopicCounts,
                    //localTopicIndex,
                    cachedLabelCoefficients,
                    labelsPerTopic,
                    gammaSum,
                    gamma,
                    lblTypeTotals,
                    type,
                    lblTypeSkewIndexes,
                    lblSkewWeight);
            //massLblValue.smoothingOnlyMass = 0; //ignore smoothing mass 



            double sample = random.nextUniform() * (massLblValue.smoothingOnlyMass + massLblValue.topicBetaMass + massLblValue.topicTermMass);

            //double sample = random.nextUniform() * (massValue.smoothingOnlyMass + massValue.topicBetaMass + massLblValue.smoothingOnlyMass + massLblValue.topicBetaMass + massLblValue.topicTermMass);

            double origSample = sample;

            //	Make sure it actually gets set
            int newTopic = -1;

            newTopic = findNewTopic(
                    sample,
                    massLblValue,
                    topicLblTermScores,
                    currentLblTypeTopicCounts,
                    localLblTopicCounts,
                    localTopicCounts,
                    localTopicIndex,
                    labelsPerTopic,
                    gammaSum,
                    gamma,
                    nonZeroTopics,
                    1 / lblWeight);


            if (newTopic == -1) {
                System.err.println("WorkerRunnable sampling labels error: " + origSample + " " + sample + " " + massLblValue.smoothingOnlyMass + " "
                        + massLblValue.topicBetaMass + " " + massLblValue.topicTermMass);
                //newTopic = numTopics - 1; // TODO is this appropriate
                //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
            }
            assert (newTopic != -1);

            //			Put that new topic into the counts
            oneDocLblTopics[position] = newTopic;

            if (ignoreLabels) {
                massLblValue.smoothingOnlyMass -= alpha[newTopic] * gamma
                        / (labelsPerTopic[newTopic] + gammaSum);
                massLblValue.topicBetaMass -= gamma * localLblTopicCounts[newTopic]
                        / (labelsPerTopic[newTopic] + gammaSum);
            } else {

                massLblValue.smoothingOnlyMass -= (1 + 1 / lblWeight) * alpha[newTopic] * gamma
                        / (labelsPerTopic[newTopic] + gammaSum);
                massLblValue.topicBetaMass -= gamma * (localLblTopicCounts[newTopic] + (1 / lblWeight) * localTopicCounts[newTopic])
                        / (labelsPerTopic[newTopic] + gammaSum);
            }

            localLblTopicCounts[newTopic]++;

            // If this is a new topic for this document,
            //  add the topic to the dense index.
            if (localLblTopicCounts[newTopic] == 1 && localTopicCounts[newTopic] == 0) {

                // First find the point where we 
                //  should insert the new topic by going to
                //  the end (which is the only reason we're keeping
                //  track of the number of non-zero
                //  topics) and working backwards

                denseIndex = nonZeroTopics;

                while (denseIndex > 0
                        && localTopicIndex[denseIndex - 1] > newTopic) {

                    localTopicIndex[denseIndex] =
                            localTopicIndex[denseIndex - 1];
                    denseIndex--;
                }

                localTopicIndex[denseIndex] = newTopic;
                nonZeroTopics++;
            }

            labelsPerTopic[newTopic]++;

            //	update the coefficients for the non-zero topics
            if (ignoreLabels) {
                cachedLabelCoefficients[newTopic] =
                        (alpha[newTopic] + localLblTopicCounts[newTopic])
                        / (labelsPerTopic[newTopic] + gammaSum);

                massLblValue.smoothingOnlyMass += alpha[newTopic] * gamma
                        / (labelsPerTopic[newTopic] + gammaSum);
                massLblValue.topicBetaMass += gamma * localLblTopicCounts[newTopic]
                        / (labelsPerTopic[newTopic] + gammaSum);

            } else {

                cachedLabelCoefficients[newTopic] = ((1 + 1 / lblWeight) * alpha[newTopic] + localLblTopicCounts[newTopic] + 1 / lblWeight * localTopicCounts[newTopic])
                        / (labelsPerTopic[newTopic] + gammaSum);

                massLblValue.smoothingOnlyMass += (1 + 1 / lblWeight) * alpha[newTopic] * gamma
                        / (labelsPerTopic[newTopic] + gammaSum);
                massLblValue.topicBetaMass += gamma * (localLblTopicCounts[newTopic] + (1 / lblWeight) * localTopicCounts[newTopic])
                        / (labelsPerTopic[newTopic] + gammaSum);

            }



        }
        if (shouldSaveState) {
            // Update the document-topic count histogram,
            //  for dirichlet estimation
            docLengthCounts[ docLength]++;


            for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
                int topic = localTopicIndex[denseIndex];

                topicDocCounts[topic][ localTopicCounts[topic]]++;
            }

            docLblLengthCounts[ docLblLength]++;

            for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
                int topic = localTopicIndex[denseIndex];

                topicLblDocCounts[topic][ localLblTopicCounts[topic]]++;
            }
        }

        //	Clean up our mess: reset the coefficients to values with only
        //	smoothing. The next doc will update its own non-zero topics...
        for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {

            int topic = localTopicIndex[denseIndex];

            if (ignoreLabels) {
                cachedCoefficients[topic] = alpha[topic] / (tokensPerTopic[topic] + betaSum);
                cachedLabelCoefficients[topic] = alpha[topic] / (labelsPerTopic[topic] + gammaSum);
            } else {
                cachedCoefficients[topic] = (1 + lblWeight) * alpha[topic] / (tokensPerTopic[topic] + betaSum);
                cachedLabelCoefficients[topic] = (1 + 1 / lblWeight) * alpha[topic] / (labelsPerTopic[topic] + gammaSum);
            }

        }

        smoothingOnlyMass = massValue.smoothingOnlyMass;
        smoothingOnlyLabelMass = massLblValue.smoothingOnlyMass;
    }
}
