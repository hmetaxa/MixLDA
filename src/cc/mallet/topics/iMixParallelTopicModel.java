/* Copyright (C) 2005 Univ. of Massachusetts Amherst, Computer Science Dept.
 This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
 http://www.cs.umass.edu/~mccallum/mallet
 This software is provided under the terms of the Common Public License,
 version 1.0, as published by http://www.opensource.org.	For further
 information, see the file `LICENSE' included with this distribution. */
package cc.mallet.topics;

import cc.mallet.topics.TopicAssignment;
import cc.mallet.types.*;
import cc.mallet.util.MalletLogger;
import cc.mallet.util.Randoms;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import java.io.*;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Formatter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.TreeSet;

import java.util.concurrent.*;
import java.util.logging.*;
import java.util.zip.*;

/**
 * Mix Parallel semi supervised, multi modal topic modal based on MALLET
 * Parallel topic model, which is a simple parallel threaded implementation of
 * LDA, following Newman, Asuncion, Smyth and Welling, Distributed Algorithms
 * for Topic Models JMLR (2009), with SparseLDA sampling scheme and data
 * structure from Yao, Mimno and McCallum, Efficient Methods for Topic Model
 * Inference on Streaming Document Collections, KDD (2009).
 *
 * @author Omiros Metaxas
 */
public class iMixParallelTopicModel implements Serializable {

    public class TopicMapping {

        public int from;
        public int to;
        //public int nonZeroTopics;
        //public double few;//frequency exclusivity weight we have an array for that
    }

    public enum SkewType {

        None,
        LabelsOnly,
        TextAndLabels
    }
    public static final int UNASSIGNED_TOPIC = -1;
    public static Logger logger = MalletLogger.getLogger(iMixParallelTopicModel.class.getName());
    public ArrayList<MixTopicModelTopicAssignment> data;  // the training instances and their topic assignments <entityId, Instances & TopicAssignments Per Modality>
    public Alphabet[] alphabet; // the alphabet for the input data per modality
    public LabelAlphabet topicAlphabet;  // the alphabet for the topics
    public int numTopics; // Number of topics to be fit
    public byte numModalities; // Number of modalities
    // These values are used to encode type/topic counts as
    //  count/topic pairs in a single int.
    public int topicMask;
    public int topicBits;
    public int[] numTypes; // per modality
    public int[] totalTokens; //per modality
    public int[] totalDocsPerModality; //number of docs containing this modality 
    //public int totalLabels;
    public TDoubleArrayList[] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
    public double[] alphaSum;
    public double[] beta;   // Prior on per-topic multinomial distribution over token types (per modality) 
    public double[] betaSum; // per modality
    public boolean usingSymmetricAlpha = false;
    public static final double DEFAULT_BETA = 0.01;
    //protected double gamma;   // Prior on per-topic multinomial distribution over labels
    //protected double gammaSum;
    //public static final double DEFAULT_GAMMA = 0.1;
    public TIntArrayList[][] typeTopicCounts; //per modality // indexed by <modalityIndex, feature index, topic index>
    public TIntArrayList[] tokensPerTopic; //per modality// indexed by <modalityIndex,topic index>
    //public int[][] typeTopicCounts; // indexed by <feature index, topic index>
    //public int[] tokensPerTopic; // indexed by <topic index>
    //protected int[][] lbltypeTopicCounts; // indexed by <label index, topic index>
    //protected int[] labelsPerTopic; // indexed by <topic index>
    // for dirichlet estimation
    //public int[] docLengthCounts; // histogram of document sizes taking into consideration (summing up) all modalities
    //public TIntObjectHashMap<int[]> topicDocCounts; // histogram of document/topic counts, indexed by <topic index, sequence position index> considering all modalities
    public int numIterations = 990;
    public int burninPeriod = 200;
    public int independentIterations = 50;
    public int saveSampleInterval = 10;
    public int optimizeInterval = 50;
    public int temperingInterval = 0;
    public int showTopicsInterval = 50;
    public int wordsPerTopic = 10;
    public int lblsPerTopic = 5;
    //public int numlblsPerTopic = 10;
    public int saveStateInterval = 0;
    public String stateFilename = null;
    public int saveModelInterval = 0;
    public String modelFilename = null;
    public int randomSeed = -1;
    public NumberFormat formatter;
    public boolean printLogLikelihood = true;
    public boolean ignoreLabels = false;
    public SkewType skewOn = SkewType.None;
    // The number of times each type appears in the corpus
    public int[][] typeTotals; //per modality
    // The skew index of eachType
    public double[][] typeSkewIndexes; //<modality, type>
    // The skew index of each Lbl Type
    //public double[] lblTypeSkewIndexes;
    // The max over typeTotals, used for beta optimization
    //public int[] lblTypeTotals;
    // The max over typeTotals, used for gamma optimization
    int[] maxTypeCount; //per modality
    double[] avgTypeCount; //per modality
    //int maxLblTypeCount;
    //double avgLblTypeCount;
    int numThreads = 1;
    double[] skewWeight; //per modality
    double[][] p_a; // a for beta prior for modalities correlation
    double[][] p_b; // b for beta prir for modalities correlation
    double[][][] pDistr_Mean; // modalities correlation distribution accross documents (used in a, b beta params optimization)
    double[][][] pDistr_Var; // modalities correlation distribution accross documents (used in a, b beta params optimization)
    public double[][] convergenceRates;//= new TObjectIntHashMap<Double>(); 
    public double[][] perplexities;//= new TObjectIntHashMap<Double>(); 
    public int numIndependentTopics; //= 5;
    private int numCommonTopics;
    private int histogramSize = 0;

    //double lblSkewWeight = 1;
    public iMixParallelTopicModel(int numberOfTopics, byte numModalities) {
        this(numberOfTopics, 5, numModalities, defaultAlphaSum(numModalities), defaultBeta(numModalities), false, SkewType.LabelsOnly);
    }

    public iMixParallelTopicModel(int numberOfTopics, int numberOfIndependentTopics, byte numModalities, double[] alphaSum, double[] beta, boolean ignoreLabels, SkewType skewnOn) {
        this(newLabelAlphabet(numberOfTopics + numberOfIndependentTopics * numModalities), numberOfIndependentTopics, numModalities, alphaSum, defaultBeta(numModalities), ignoreLabels, skewnOn);
    }

    private static LabelAlphabet newLabelAlphabet(int numTopics) {
        LabelAlphabet ret = new LabelAlphabet();
        for (int i = 0; i < numTopics; i++) {
            ret.lookupIndex("topic" + i);
        }
        return ret;
    }

    private static double[] defaultBeta(int numModalities) {
        double[] ret = new double[numModalities];
        Arrays.fill(ret, DEFAULT_BETA);
        return ret;
    }

    private static double[] defaultAlphaSum(int numModalities) {
        double[] ret = new double[numModalities];
        Arrays.fill(ret, 1);
        return ret;
    }

    public iMixParallelTopicModel(LabelAlphabet topicAlphabet, int numberOfIndependentTopics, byte numModalities, double[] alphaSum, double[] beta, boolean ignoreLabels, SkewType skewnOn) {

        this.numModalities = numModalities;
        this.numIndependentTopics = numberOfIndependentTopics;

        this.data = new ArrayList<MixTopicModelTopicAssignment>();
        this.numTypes = new int[numModalities];
        this.totalTokens = new int[numModalities];
        this.betaSum = new double[numModalities];
        this.totalDocsPerModality = new int[numModalities];

        this.typeTopicCounts = new TIntArrayList[numModalities][];
        this.tokensPerTopic = new TIntArrayList[numModalities];
        //this.docLengthCounts = new int[numModalities][];
        //this.topicDocCounts = new int[numModalities][][];
        this.typeTotals = new int[numModalities][];
        this.typeSkewIndexes = new double[numModalities][];

        this.maxTypeCount = new int[numModalities];
        this.avgTypeCount = new double[numModalities];
        this.skewWeight = new double[numModalities];
        Arrays.fill(this.skewWeight, 1);

        this.topicAlphabet = topicAlphabet;
        this.numTopics = topicAlphabet.size();
        this.numCommonTopics = this.numTopics - numModalities * numberOfIndependentTopics;

        this.alphabet = new Alphabet[numModalities];

        if (Integer.bitCount(numTopics) == 1) {
            // exact power of 2
            topicMask = numTopics - 1;
            topicBits = Integer.bitCount(topicMask);
        } else {
            // otherwise add an extra bit
            topicMask = Integer.highestOneBit(numTopics) * 2 - 1;
            topicBits = Integer.bitCount(topicMask);
        }

        this.ignoreLabels = ignoreLabels;
        this.skewOn = skewnOn;
        this.alphaSum = alphaSum;
        this.alpha = new TDoubleArrayList[numModalities];

        this.beta = beta;

        //this.tokensPerTopic = new  TIntArrayList[numModalities];
        convergenceRates = new double[numModalities][120];
        perplexities = new double[numModalities][120];

        formatter = NumberFormat.getInstance();
        formatter.setMaximumFractionDigits(5);

        logger.info("Coded LDA: " + numTopics + " topics, " + topicBits + " topic bits, "
                + Integer.toBinaryString(topicMask) + " topic mask");

        p_a = new double[numModalities][numModalities];
        p_b = new double[numModalities][numModalities];

    }

    public Alphabet[] getAlphabet() {
        return alphabet;
    }

    public LabelAlphabet getTopicAlphabet() {
        return topicAlphabet;
    }

    public int getNumTopics() {
        return numTopics;
    }

    public ArrayList<MixTopicModelTopicAssignment> getData() {
        return data;
    }

    public void setNumIterations(int numIterations) {
        this.numIterations = numIterations;
    }

    public void setIndependentIterations(int numIterations) {
        this.independentIterations = numIterations;
    }

    public void setBurninPeriod(int burninPeriod) {
        this.burninPeriod = burninPeriod;
    }

    public void setTopicDisplay(int interval, int n) {
        this.showTopicsInterval = interval;
        this.wordsPerTopic = n;
    }

    public void setRandomSeed(int seed) {
        randomSeed = seed;
    }

    /**
     * Interval for optimizing Dirichlet hyperparameters
     */
    public void setOptimizeInterval(int interval) {
        this.optimizeInterval = interval;

        // Make sure we always have at least one sample
        //  before optimizing hyperparameters
        if (saveSampleInterval > optimizeInterval) {
            saveSampleInterval = optimizeInterval;
        }
    }

    public void setSymmetricAlpha(boolean b) {
        usingSymmetricAlpha = b;
    }

    public void setTemperingInterval(int interval) {
        temperingInterval = interval;
    }

    public void setNumThreads(int threads) {
        this.numThreads = threads;
    }

    /**
     * Define how often and where to save a text representation of the current
     * state. Files are GZipped.
     *
     * @param interval Save a copy of the state every <code>interval</code>
     * iterations.
     * @param filename Save the state to this file, with the iteration number as
     * a suffix
     */
    public void setSaveState(int interval, String filename) {
        this.saveStateInterval = interval;
        this.stateFilename = filename;
    }

    /**
     * Define how often and where to save a serialized model.
     *
     * @param interval Save a serialized model every <code>interval</code>
     * iterations.
     * @param filename Save to this file, with the iteration number as a suffix
     */
    public void setSaveSerializedModel(int interval, String filename) {
        this.saveModelInterval = interval;
        this.modelFilename = filename;
    }

    public void addInstances(InstanceList[] training) {

        //    Iterator<Integer> keySetIterator = map.keySet().iterator();
//while(keySetIterator.hasNext()){
        TObjectIntHashMap<String> entityPosition = new TObjectIntHashMap<String>();

        for (Byte i = 0; i < numModalities; i++) {

            Alphabet tmpAlphabet = training[i].getDataAlphabet();
            Integer tmpNumTypes = tmpAlphabet.size();
            alphabet[i] = tmpAlphabet;
            logger.info("Modality: " + i + " Alphabet count: " + tmpAlphabet.size());

            numTypes[i] = tmpNumTypes;
            betaSum[i] = beta[i] * tmpNumTypes;

            alpha[i] = new TDoubleArrayList(numTopics);
            this.alpha[i].fill(0, numTopics, alphaSum[i] / numTopics);

            typeTopicCounts[i] = new TIntArrayList[tmpNumTypes];
            tokensPerTopic[i] = new TIntArrayList(numTopics);

            typeTotals[i] = new int[tmpNumTypes];
            typeSkewIndexes[i] = new double[tmpNumTypes];

            Randoms random = null;
            if (randomSeed == -1) {
                random = new Randoms();
            } else {
                random = new Randoms(randomSeed);
            }

            int doc = 0;

            for (Instance instance : training[i]) {
                doc++;
                long iterationStart = System.currentTimeMillis();

                FeatureSequence tokens = (FeatureSequence) instance.getData();
                int size = tokens.size();

                int[] topics = new int[size]; //topicSequence.getFeatures();
                for (int position = 0; position < topics.length; position++) {

                    //int topic = random.nextInt(numTopics);
                    int type = tokens.getIndexAtPosition(position);
                    typeTotals[i][ type]++;

                    int topic = ThreadLocalRandom.current().nextInt(numCommonTopics + numIndependentTopics);//random.nextInt(numTopics);
                    if (topic >= numCommonTopics) {
                        topic = topic - numCommonTopics + 1;
                        topic = numCommonTopics - 1 + numIndependentTopics * i + topic;
                    }
                    topics[position] = topic;
                }

                //lblSequence.
                TopicAssignment t = new TopicAssignment(instance, new LabelSequence(topicAlphabet, topics), new long[size]);
                MixTopicModelTopicAssignment mt;
                String entityId = (String) instance.getName();

                //int index = i == 0 ? -1 : data.indexOf(mt);
                int index = -1;
                //if (i != 0 && (entityPosition.containsKey(entityId))) {

                if (i != 0 && entityPosition.containsKey(entityId)) {

                    index = entityPosition.get(entityId);
                    mt = data.get(index);
                    mt.Assignments[i] = t;

                } else {
                    mt = new MixTopicModelTopicAssignment(entityId, new TopicAssignment[numModalities]);
                    mt.Assignments[i] = t;
                    data.add(mt);
                    index = data.size() - 1;
                    entityPosition.put(entityId, index);
                }

                long elapsedMillis = System.currentTimeMillis() - iterationStart;
                if (doc % 100 == 0) {
                    logger.fine(elapsedMillis + "ms " + "  docNum:" + doc);

                }

            }

            //omiros here
            maxTypeCount[i] = 0;
            avgTypeCount[i] = 0d;

            // Allocate enough space so that we never have to worry about
            //  overflows: either the number of topics or the number of times
            //  the type occurs.
            for (int type = 0; type < numTypes[i]; type++) {
                avgTypeCount[i] += (double) typeTotals[i][type] / (double) numTypes[i];
                if (typeTotals[i][type] > maxTypeCount[i]) {
                    maxTypeCount[i] = typeTotals[i][type];
                }
                typeTopicCounts[i][type] = new TIntArrayList(Math.min(numTopics, typeTotals[i][type]));
            }

        }

        //LabelSequence lblSequence =  new LabelSequence(topicAlphabet);
        buildInitialTypeTopicCounts();

        initializeHistograms();
    }

    public void initializeFromState(File stateFile) throws IOException {
        String line;
        String[] fields;

        BufferedReader reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(stateFile))));
        line = reader.readLine();

        // Skip some lines starting with "#" that describe the format and specify hyperparameters
        while (line.startsWith("#")) {
            line = reader.readLine();
        }

        fields = line.split(" ");

        for (MixTopicModelTopicAssignment entity : data) {
            for (Byte i = 0; i < numModalities; i++) {
                TopicAssignment document = entity.Assignments[i];
                if (document != null) {
                    FeatureSequence tokens = (FeatureSequence) document.instance.getData();
                    FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;
                    int[] topics = topicSequence.getFeatures();
                    for (int position = 0; position < tokens.size(); position++) {
                        int type = tokens.getIndexAtPosition(position);

                        if (type == Integer.parseInt(fields[3])) {
                            topics[position] = Integer.parseInt(fields[7]);
                        } else {
                            System.err.println("instance list and state do not match: " + line);
                            throw new IllegalStateException();
                        }

                        line = reader.readLine();
                        if (line != null) {
                            fields = line.split(" ");
                        }
                    }
                }

            }

        }

        buildInitialTypeTopicCounts();

        initializeHistograms();
    }

    public void buildInitialTypeTopicCounts() {

        for (Byte i = 0; i < numModalities; i++) {
            // Clear the topic totals
            //tokensPerTopic[i].clear();
            tokensPerTopic[i].fill(0, numTopics, 0);
            //tokensPerTopic[i].reset();
            //Arrays.fill(tokensPerTopic[i], 0);

            // Clear the type/topic counts, only 
            //  looking at the entries before the first 0 entry.
            for (int type = 0; type < numTypes[i]; type++) {

                typeSkewIndexes[i][type] = 0; //TODO: Initialize based on documents
                //tokensPerTopic[i].fill(0, numTopics, 0);
                typeTopicCounts[i][type].fill(0, numTopics, 0);

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

        Arrays.fill(totalDocsPerModality, 0);

        for (MixTopicModelTopicAssignment entity : data) {
            for (Byte i = 0; i < numModalities; i++) {
                TopicAssignment document = entity.Assignments[i];

                if (document != null) {
                    totalDocsPerModality[i]++;
                    FeatureSequence tokens = (FeatureSequence) document.instance.getData();
                    FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;
                    int[] topics = topicSequence.getFeatures();

                    for (int position = 0; position < tokens.size(); position++) {

                        int topic = topics[position];

                        if (topic == UNASSIGNED_TOPIC) {
                            continue;
                        }

                        TIntArrayList tmplist = tokensPerTopic[i];

                        int tmp = tmplist.get(topic);
                        tokensPerTopic[i].set(topic, tmp + 1);
                        //tokensPerTopic[i][topic]++;

                        // The format for these arrays is 
                        //  the topic in the rightmost bits
                        //  the count in the remaining (left) bits.
                        // Since the count is in the high bits, sorting (desc)
                        //  by the numeric value of the int guarantees that
                        //  higher counts will be before the lower counts.
                        int type = tokens.getIndexAtPosition(position);
                        TIntArrayList currentTypeTopicCounts = typeTopicCounts[i][ type];

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
                                logger.info("overflow on type " + type);
                            }
                            currentTopic = currentTypeTopicCounts.get(index) & topicMask;
                        }
                        currentValue = currentTypeTopicCounts.get(index) >> topicBits;

                        if (currentValue == 0) {
                            // new value is 1, so we don't have to worry about sorting
                            //  (except by topic suffix, which doesn't matter)

                            currentTypeTopicCounts.set(index, (1 << topicBits) + topic);
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

    public void mergeSimilarTopics(int numWords, Set modalities, double mergeSimilarity) {

        // consider similarity on top numWords
        HashMap<String, SparseVector> labelVectors = new HashMap<String, SparseVector>();
        String labelId = "";
        NormalizedDotProductMetric cosineSimilarity = new NormalizedDotProductMetric();

        int[] topicMapping = new int[numTopics];
        int previousVocabularySize = 0;
        for (int topic = 0; topic < numTopics; topic++) {
            labelId = Integer.toString(topic);
            int[] wordTypes = new int[numWords * modalities.size()];
            double[] weights = new double[numWords * modalities.size()];

            for (Byte m = 0; m < numModalities && modalities.contains(m); m++) {

                ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords(m);
                TreeSet<IDSorter> sortedWords = topicSortedWords.get(topic);
                Iterator<IDSorter> iterator = sortedWords.iterator();

                int wordCnt = 0;
                while (iterator.hasNext() && wordCnt < numWords) {

                    IDSorter info = iterator.next();
                    wordTypes[wordCnt] = previousVocabularySize+info.getID();//((String) entity).hashCode();
                    weights[wordCnt] = info.getWeight();
                    wordCnt++;
                }

                previousVocabularySize += maxTypeCount[m];
            }

            labelVectors.put(labelId, new SparseVector(wordTypes, weights, numWords, numWords, true, false, true));
        }

        double similarity = 0;
        //double maxSimilarity = 0;
        String labelTextId;

        //TObjectDoubleHashMap<String> topicSimilarities = new TObjectDoubleHashMap<String>();
        double[][] topicSimilarities = new double[numTopics][numTopics];

        for (int i = 0; i < numTopics; i++) {
            Arrays.fill(topicSimilarities[i], 0);
        }
        TIntArrayList usedTopics = new TIntArrayList();
        TIntArrayList emptyTopics = new TIntArrayList();
        for (int t = 0; t < numTopics; t++) {

            for (int t_text = 0; t_text < numTopics; t_text++) {
                labelId = Integer.toString(t);
                labelTextId = "0_" + t_text;
                SparseVector source = labelVectors.get(labelId);
                SparseVector target = labelVectors.get(labelTextId);
                similarity = 0;
                if (source != null & target != null) {
                    similarity = 1 - Math.abs(cosineSimilarity.distance(source, target)); // the function returns distance not similarity
                }
                topicSimilarities[t][t_text] = similarity;
            }

        }

        Arrays.fill(topicMapping, -1);
        boolean found = true;
        TopicMapping topicMap = new TopicMapping();
        while (found) {
            found = findNextMapping(topicSimilarities, topicMap, mergeSimilarity);
            if (found) {
                if (!usedTopics.contains(topicMap.from)) {
                    emptyTopics.add(topicMap.from);
                }
                topicMapping[topicMap.from] = topicMap.to;
                usedTopics.add(topicMap.to);
                if (emptyTopics.contains(topicMap.to)) {
                    emptyTopics.remove(emptyTopics.indexOf(topicMap.to));
                }
            }
        }

        for (int i = 0; i < numTopics; i++) {
            if (topicMapping[i] == 0 && usedTopics.contains(i)) {
                topicMapping[i] = emptyTopics.get(0);
                emptyTopics.remove(0);
            }
        }

        for (MixTopicModelTopicAssignment entity : data) {
            for (Byte m = 0; m < numModalities; m++) {
                TopicAssignment document = entity.Assignments[m];

                if (document != null) {

                    //FeatureSequence tokens = (FeatureSequence) document.instance.getData();
                    FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;
                    int[] topics = topicSequence.getFeatures();

                    for (int position = 0; position < topics.length; position++) {
                        int oldTopic = topics[position];
                        if (topicMapping[oldTopic] != -1) {
                            topics[position] = topicMapping[oldTopic];
                        }

                    }
                }
            }
        }
        buildInitialTypeTopicCounts();

    }

    public void alignTopicsInModalities() {
//        int[] totalModalityTokens = new int[numModalities];
//        int[] totalConvergedTokens = new int[numModalities];
//        Arrays.fill(totalModalityTokens, 0);
//        Arrays.fill(totalConvergedTokens, 0);

        //int[][][] entityIdPerTopicPerModality = new int[numModalities][numTopics][data.size()];
        //int[][][] entityIdPerTopicPerModality = new int[numModalities][numTopics][data.size()];
        HashMap<String, SparseVector> labelVectors = new HashMap<String, SparseVector>();
        String labelId = "";
        NormalizedDotProductMetric cosineSimilarity = new NormalizedDotProductMetric();
        //TObjectIntHashMap<String> topicEntitiesCnt = new TObjectIntHashMap<String>();

        HashMap<String, TIntIntHashMap> topicEntitiesPerModality = new HashMap<String, TIntIntHashMap>();

        for (Byte m = 0; m < numModalities; m++) {

            for (int t = 0; t < numTopics; t++) {
                labelId = m + "_" + t;
                topicEntitiesPerModality.put(labelId, new TIntIntHashMap());
            }
        }
        //TObjectIntHashMap<String> topicEntitiesCnt = new TObjectIntHashMap<String>();
        // topicEntitiesCnt.clear();
        for (MixTopicModelTopicAssignment entity : data) {

            int entityId = entity.EntityId.hashCode();
            try {
                entityId = Integer.parseInt(entity.EntityId);

            } catch (NumberFormatException e) {
                // entityId = entity.EntityId.hashCode();
            }

            for (Byte m = 0; m < numModalities; m++) {

                //for (int t = 0; t < numCommonTopics; t++) {
                TopicAssignment document = entity.Assignments[m];

                if (document != null) {

                    //FeatureSequence tokens = (FeatureSequence) document.instance.getData();
                    FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;
                    int[] topics = topicSequence.getFeatures();

                    for (int position = 0; position < topics.length; position++) {
                        //if (topics[position] < numCommonTopics) {
                        labelId = m + "_" + topics[position];
                        topicEntitiesPerModality.get(labelId).adjustOrPutValue(entityId, 1, 1);
                        // }
                    }
                    // }
                }
            }
        }

        for (Byte m = 0; m < numModalities; m++) {

            for (int t = 0; t < numTopics; t++) {
                if (t < numCommonTopics || (m > 0 && t >= numCommonTopics + m * numIndependentTopics && t < numCommonTopics + (m + 1) * numIndependentTopics)) {
                    labelId = m + "_" + t;
                    int[] entities = new int[data.size()];
                    double[] weights = new double[data.size()];
                    int cnt = 0;
                    int[] entityIds = topicEntitiesPerModality.get(labelId).keys();
                    Arrays.sort(entityIds);
                    for (int entity : entityIds) {
                        entities[cnt] = entity;//((String) entity).hashCode();
                        weights[cnt] = topicEntitiesPerModality.get(labelId).get(entity);
                        cnt++;
                    }

                    labelVectors.put(labelId, new SparseVector(entities, weights, entities.length, entities.length, true, false, true));
                }
            }
        }

        double similarity = 0;
        //double maxSimilarity = 0;
        String labelTextId;
        int[][] topicMapping = new int[numModalities][numTopics];
        //TObjectDoubleHashMap<String> topicSimilarities = new TObjectDoubleHashMap<String>();
        double[][] topicSimilarities = new double[numTopics][numTopics];

        for (Byte m = 1; m < numModalities; m++) {
            for (int i = 0; i < numTopics; i++) {
                Arrays.fill(topicSimilarities[i], 0);
            }
            TIntArrayList usedTopics = new TIntArrayList();
            TIntArrayList emptyTopics = new TIntArrayList();
            for (int t = 0; t < numTopics; t++) {

                for (int t_text = 0; t_text < numTopics; t_text++) {
                    labelId = m + "_" + t;
                    labelTextId = "0_" + t_text;
                    SparseVector source = labelVectors.get(labelId);
                    SparseVector target = labelVectors.get(labelTextId);
                    similarity = 0;
                    if (source != null & target != null) {
                        similarity = 1 - Math.abs(cosineSimilarity.distance(source, target)); // the function returns distance not similarity
                    }
                    topicSimilarities[t][t_text] = similarity;
                }

            }

            Arrays.fill(topicMapping[m], -1);
            boolean found = true;
            TopicMapping topicMap = new TopicMapping();
            while (found) {
                found = findNextMapping(topicSimilarities, topicMap,0);
                if (found) {
                    if (!usedTopics.contains(topicMap.from)) {
                        emptyTopics.add(topicMap.from);
                    }
                    topicMapping[m][topicMap.from] = topicMap.to;
                    usedTopics.add(topicMap.to);
                    if (emptyTopics.contains(topicMap.to)) {
                        emptyTopics.remove(emptyTopics.indexOf(topicMap.to));
                    }
                }
            }

            for (int i = 0; i < numTopics; i++) {
                if (topicMapping[m][i] == 0 && usedTopics.contains(i)) {
                    topicMapping[m][i] = emptyTopics.get(0);
                    emptyTopics.remove(0);
                }
            }

        }

        for (MixTopicModelTopicAssignment entity : data) {
            for (Byte m = 1; m < numModalities; m++) {
                TopicAssignment document = entity.Assignments[m];

                if (document != null) {

                    //FeatureSequence tokens = (FeatureSequence) document.instance.getData();
                    FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;
                    int[] topics = topicSequence.getFeatures();

                    for (int position = 0; position < topics.length; position++) {
                        int oldTopic = topics[position];
                        if (topicMapping[m][oldTopic] != -1) {
                            topics[position] = topicMapping[m][oldTopic];
                        }

                    }
                }

            }
        }

        buildInitialTypeTopicCounts();

    }

    private boolean findNextMapping(double[][] topicSimilarities, TopicMapping topicMap, double maxSimilarity) {
        boolean found = false;
        topicMap.from = 0;
        topicMap.to = 0;
        //double maxSimilarity = 0;
        for (int i = 0; i < numTopics; i++) {
            for (int j = 0; j < numTopics; j++) {
                if (topicSimilarities[i][j] != 0 && topicSimilarities[i][j] > maxSimilarity) {
                    found = true;
                    maxSimilarity = topicSimilarities[i][j];
                    topicMap.from = i;
                    topicMap.to = j;
                    //Arrays.fill(topicSimilarities[i], 0);
                    for (int k = 0; k < numCommonTopics; k++) {
                        topicSimilarities[i][k] = 0;
                        topicSimilarities[k][j] = 0;
                    }
                }
            }
        }
        return found;
    }

    //fast sampling initialization: first p=0 --> sample every modality independently, then map topics based on cosine or symmetricKL simalrity
    // on entities distribution 
    //Needs much work to align topics 
     /*
     public void alignTopicsInModalities() {
     //        int[] totalModalityTokens = new int[numModalities];
     //        int[] totalConvergedTokens = new int[numModalities];
     //        Arrays.fill(totalModalityTokens, 0);
     //        Arrays.fill(totalConvergedTokens, 0);

     //int[][][] entityIdPerTopicPerModality = new int[numModalities][numTopics][data.size()];
     //int[][][] entityIdPerTopicPerModality = new int[numModalities][numTopics][data.size()];



     HashMap<String, SparseVector> labelVectors = new HashMap<String, SparseVector>();
     String labelId = "";
     NormalizedDotProductMetric cosineSimilarity = new NormalizedDotProductMetric();
     //TObjectIntHashMap<String> topicEntitiesCnt = new TObjectIntHashMap<String>();

     HashMap<String, TIntIntHashMap> topicEntitiesPerModality = new HashMap<String, TIntIntHashMap>();

     for (Byte m = 0; m < numModalities; m++) {

     for (int t = 0; t < numTopics; t++) {
     if (t < numCommonTopics || (t >= numCommonTopics + m * numIndependentTopics && t < numCommonTopics + (m + 1) * numIndependentTopics)) {
     labelId = m + "_" + t;
     topicEntitiesPerModality.put(labelId, new TIntIntHashMap());
     }
     }
     }
     //TObjectIntHashMap<String> topicEntitiesCnt = new TObjectIntHashMap<String>();
     // topicEntitiesCnt.clear();
     for (MixTopicModelTopicAssignment entity : data) {

     int entityId = entity.EntityId.hashCode();
     try {
     entityId = Integer.parseInt(entity.EntityId);

     } catch (NumberFormatException e) {
     // entityId = entity.EntityId.hashCode();
     }

     for (Byte m = 0; m < numModalities; m++) {

     //for (int t = 0; t < numCommonTopics; t++) {
     TopicAssignment document = entity.Assignments[m];

     if (document != null) {

     //FeatureSequence tokens = (FeatureSequence) document.instance.getData();
     FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;
     int[] topics = topicSequence.getFeatures();

     for (int position = 0; position < topics.length; position++) {
     //if (topics[position] < numCommonTopics) {
     labelId = m + "_" + topics[position];
     topicEntitiesPerModality.get(labelId).adjustOrPutValue(entityId, 1, 1);
     // }
     }
     // }
     }
     }


     for (Byte m = 0; m < numModalities; m++) {


     for (int t = 0; t < numTopics; t++) {
     if (t < numCommonTopics || (t >= numCommonTopics + m * numIndependentTopics && t < numCommonTopics + (m + 1) * numIndependentTopics)) {
     labelId = m + "_" + t;
     int[] entities = new int[data.size()];
     double[] weights = new double[data.size()];
     int cnt = 0;
     int[] entityIds = topicEntitiesPerModality.get(labelId).keys();
     Arrays.sort(entityIds);
     for (int entId : entityIds) {
     entities[cnt] = entId;//((String) entity).hashCode();
     weights[cnt] = topicEntitiesPerModality.get(labelId).get(entId);
     cnt++;
     }

     labelVectors.put(labelId, new SparseVector(entities, weights, entities.length, entities.length, true, false, true));
     }
     }
     }
     }
     double similarity = 0;
     //double maxSimilarity = 0;
     String labelTextId;
     int[][] topicMapping = new int[numModalities][numTopics];
     //TObjectDoubleHashMap<String> topicSimilarities = new TObjectDoubleHashMap<String>();
     double[][] topicSimilarities = new double[numTopics][numTopics];
     for (Byte m = 1; m < numModalities; m++) {
     for (int i = 0; i < numTopics; i++) {
     Arrays.fill(topicSimilarities[i], 0);
     }
     TIntArrayList usedTopics = new TIntArrayList();
     TIntArrayList emptyTopics = new TIntArrayList();
     for (int t = 0; t < numTopics; t++) {

     for (int t_text = 0; t_text < numTopics; t_text++) {
     labelId = m + "_" + t;
     labelTextId = "0_" + t_text;
     SparseVector source = labelVectors.get(labelId);
     SparseVector target = labelVectors.get(labelTextId);
     similarity = 0;
     if (source != null & target != null) {
     similarity = 1 - Math.abs(cosineSimilarity.distance(source, target)); // the function returns distance not similarity
     }
     topicSimilarities[t][t_text] = similarity;
     }

     }

     Arrays.fill(topicMapping[m], -1);
     boolean found = true;
     TopicMapping topicMap = new TopicMapping();
     while (found) {
     found = findNextMapping(topicSimilarities, topicMap);
     if (found) {
     if (!usedTopics.contains(topicMap.from)) {
     emptyTopics.add(topicMap.from);
     }
     topicMapping[m][topicMap.from] = topicMap.to;
     usedTopics.add(topicMap.to);
     if (emptyTopics.contains(topicMap.to)) {
     emptyTopics.remove(emptyTopics.indexOf(topicMap.to));
     }
     }
     }

     for (int i = 0; i < numTopics; i++) {
     if (topicMapping[m][i] == 0 && usedTopics.contains(i)) {
     topicMapping[m][i] = emptyTopics.get(0);
     emptyTopics.remove(0);
     }
     }

     }
     for (MixTopicModelTopicAssignment entity : data) {
     for (Byte m = 1; m < numModalities; m++) {
     TopicAssignment document = entity.Assignments[m];

     if (document != null) {

     //FeatureSequence tokens = (FeatureSequence) document.instance.getData();
     FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;
     int[] topics = topicSequence.getFeatures();

     for (int position = 0; position < topics.length; position++) {
     int oldTopic = topics[position];
     if (oldTopic < numTopics && topicMapping[m][oldTopic] != -1) {
     topics[position] = topicMapping[m][oldTopic];
     }

     }
     }


     }
     }

     buildInitialTypeTopicCounts();
     }
     */
    public void sumTypeTopicCounts(iMixWorkerRunnable[] runnables, boolean calcSkew) {

        // Clear the type/topic counts, only 
        //  looking at the entries before the first 0 entry.
        for (Byte i = 0; i < numModalities; i++) {

            for (Byte m = 0; m < numModalities; m++) {

                Arrays.fill(pDistr_Mean[i][m], 0);

            }
            // Clear the topic totals
            //tokensPerTopic[i].reset();
            tokensPerTopic[i].fill(0, numTopics, 0);
            //Arrays.fill(tokensPerTopic[i], 0);

            for (int type = 0; type < numTypes[i]; type++) {

                typeTopicCounts[i][type].fill(0, numTopics, 0);//.reset();
//                int[] targetCounts = typeTopicCounts[i][type];
//
//                int position = 0;
//                while (position < targetCounts.length
//                        && targetCounts[position] > 0) {
//                    targetCounts[position] = 0;
//                    position++;
//                }

            }
        }

        for (int thread = 0; thread < numThreads; thread++) {

            // Handle the total-tokens-per-topic array
            TIntArrayList[] sourceTotals = runnables[thread].getTokensPerTopic();
            TIntArrayList[][] sourceTypeTopicCounts = runnables[thread].getTypeTopicCounts();

            double[][][] distrP_Mean = runnables[thread].getPDistr_Mean();

            for (Byte i = 0; i < numModalities; i++) {

                for (Byte m = 0; m < numModalities; m++) {
                    for (int doc = 0; doc < pDistr_Mean[i][m].length; doc++) {
                        pDistr_Mean[i][m][doc] += distrP_Mean[i][m][doc]; //TODO: Omiros check if each thread has its own range... 

                    }
                }

                for (int topic = 0; topic < numTopics; topic++) {
                    tokensPerTopic[i].set(topic, tokensPerTopic[i].get(topic) + sourceTotals[i].get(topic));
                }

                // Now handle the individual type topic counts
                for (int type = 0; type < numTypes[i]; type++) {

                    // Here the source is the individual thread counts,
                    //  and the target is the global counts.
                    int[] sourceCounts = sourceTypeTopicCounts[i][type].toArray();
                    TIntArrayList targetCounts = typeTopicCounts[i][type];

                    int sourceIndex = 0;

                    while (sourceIndex < sourceCounts.length
                            && sourceCounts[sourceIndex] > 0) {

                        int topic = sourceCounts[sourceIndex] & topicMask;
                        int count = sourceCounts[sourceIndex] >> topicBits;

                        int targetIndex = 0;
                        int currentTopic = targetCounts.get(targetIndex) & topicMask;
                        int currentCount;

                        while (targetCounts.get(targetIndex) > 0 && currentTopic != topic) {
                            targetIndex++;
                            currentTopic = targetCounts.get(targetIndex) & topicMask;

                        }

                        //if (targetCounts.get(targetIndex) == 0 && currentTopic != topic) {
                        //    logger.info("overflow in merging on type " + type + " moodality: " + i + " thread: " + thread);
                        // }
                        currentCount = targetCounts.get(targetIndex) >> topicBits;

                        targetCounts.set(targetIndex,
                                ((currentCount + count) << topicBits) + topic);

                        // Now ensure that the array is still sorted by 
                        //  bubbling this value up.
                        while (targetIndex > 0
                                && targetCounts.get(targetIndex) > targetCounts.get(targetIndex - 1)) {
                            int temp = targetCounts.get(targetIndex);
                            targetCounts.set(targetIndex, targetCounts.get(targetIndex - 1));
                            targetCounts.set(targetIndex - 1, temp);

                            targetIndex--;
                        }

                        sourceIndex++;
                    }
                }

            }
        }

    }

    //TODO save weights in DB
    private void calcSkew() {
        // Calc Skew weight
        //skewOn == SkewType.LabelsOnly
        double skewSum = 0;
        int nonZeroSkewCnt = 1;
        byte initModality = 0;
        if (skewOn == SkewType.LabelsOnly) {
            initModality = 1;
        }

        for (Byte i = 0; i < numModalities; i++) {
            if (skewOn != SkewType.None && i >= initModality) {

                for (int type = 0; type < numTypes[i]; type++) {

                    int totalTypeCounts = 0;
                    typeSkewIndexes[i][type] = 0;

                    int[] targetCounts = typeTopicCounts[i][type].toArray();

                    int index = 0;
                    int count = 0;
                    while (index < targetCounts.length
                            && targetCounts[index] > 0) {
                        count = targetCounts[index] >> topicBits;
                        typeSkewIndexes[i][type] += Math.pow((double) count, 2);
                        totalTypeCounts += count;
                        //currentTopic = currentTypeTopicCounts[index] & topicMask;
                        index++;
                    }

                    if (totalTypeCounts > 0) {
                        typeSkewIndexes[i][type] = typeSkewIndexes[i][type] / Math.pow((double) (totalTypeCounts), 2);
                    }
                    if (typeSkewIndexes[i][type] > 0) {
                        nonZeroSkewCnt++;
                        skewSum += typeSkewIndexes[i][type];
                    }

                }

                skewWeight[i] = (double) 1 / (1 + skewSum / (double) nonZeroSkewCnt);

            }
        }
    }

    /**
     * Gather statistics on the size of documents and create histograms for use
     * in Dirichlet hyperparameter optimization.
     */
    private void initializeHistograms() {

        int maxTotalAllModalities = 0;
        int[] maxTokens = new int[numModalities];
        int[] maxTotal = new int[numModalities];

        Arrays.fill(totalTokens, 0);
        Arrays.fill(maxTokens, 0);
        Arrays.fill(maxTotal, 0);

        for (MixTopicModelTopicAssignment entity : data) {
            for (Byte i = 0; i < numModalities; i++) {
                int seqLen;
                TopicAssignment document = entity.Assignments[i];
                if (document != null) {
                    FeatureSequence fs = (FeatureSequence) document.instance.getData();
                    seqLen = fs.getLength();
                    // if (seqLen > maxTokens) {
                    //     maxTokens = seqLen;
                    // }

                    totalTokens[i] += seqLen;
                    if (seqLen > maxTotal[i]) {
                        maxTotal[i] = seqLen;
                    }

                }

            }

            //int maxSize = Math.max(maxLabels, maxTokens);
        }
        for (Byte i = 0; i < numModalities; i++) {
            logger.info(" modality: " + i + " max tokens: " + maxTotal[i]);
            logger.info(" modality: " + i + " total tokens: " + totalTokens[i]);
            maxTotalAllModalities += maxTotal[i];
        }
        logger.info("max tokens all modalities: " + maxTotalAllModalities);
        histogramSize = maxTotalAllModalities + 1;
        //not needed
//        docLengthCounts = new int[maxTotalAllModalities + 1];
//        topicDocCounts = new TIntObjectHashMap<int[]>(numTopics); //[maxTotalAllModalities + 1];
//        for (int topic = 0; topic < topicDocCounts.size(); topic++) {
//            topicDocCounts.put(topic, new int[docLengthCounts.length]);
//        }
    }

    public void optimizeAlpha(iMixWorkerRunnable[] runnables) {

        int[][] docLengthCounts = new int[numModalities][histogramSize]; // histogram of document sizes taking into consideration (summing up) all modalities
        int[][][] topicDocCounts = new int[numModalities][numTopics][histogramSize]; // histogram of document/topic counts, indexed by <topic index, sequence position index> considering all modalities

        for (Byte m = 0; m < numModalities; m++) {
            for (int thread = 0; thread < numThreads; thread++) {
                int[][] sourceLengthCounts = runnables[thread].getDocLengthCounts();
                TIntObjectHashMap<int[]>[] sourceTopicCounts = runnables[thread].getTopicDocCounts();

                for (int count = 0; count < sourceLengthCounts[m].length; count++) {
                    // for (Byte i = 0; i < numModalities; i++) {
                    if (sourceLengthCounts[m][count] > 0) {
                        docLengthCounts[m][count] += sourceLengthCounts[m][count];
                        sourceLengthCounts[m][count] = 0;
                    }
                    //}
                }

                for (int topic = 0; topic < numTopics; topic++) {

                    if (!usingSymmetricAlpha) {
                        for (int count = 0; count < sourceTopicCounts[m].get(topic).length; count++) {
                            //for (Byte i = 0; i < numModalities; i++) {
                            if (sourceTopicCounts[m].get(topic)[count] > 0) {
                                topicDocCounts[m][topic][count] += sourceTopicCounts[m].get(topic)[count];
                                sourceTopicCounts[m].get(topic)[count] = 0;
                            }
                            // }
                        }
                    } else {
                        // For the symmetric version, we only need one 
                        //  count array, which I'm putting in the same 
                        //  data structure, but for topic 0. All other
                        //  topic histograms will be empty.
                        // I'm duplicating this for loop, which 
                        //  isn't the best thing, but it means only checking
                        //  whether we are symmetric or not numTopics times, 
                        //  instead of numTopics * longest document length.
                        for (int count = 0; count < sourceTopicCounts[m].get(topic).length; count++) {
                            //for (Byte i = 0; i < numModalities; i++) {
                            if (sourceTopicCounts[m].get(topic)[count] > 0) {
                                topicDocCounts[m][0][count] += sourceTopicCounts[m].get(topic)[count];
                                //			 ^ the only change
                                sourceTopicCounts[m].get(topic)[count] = 0;
                            }
                            //}
                        }
                    }
                }
            }

            if (usingSymmetricAlpha) {
                alphaSum[m] = Dirichlet.learnSymmetricConcentration(topicDocCounts[m][0],
                        docLengthCounts[m],
                        numTopics,
                        alphaSum[m]);
                for (int topic = 0; topic < numTopics; topic++) {
                    alpha[m].set(topic, alphaSum[m] / numTopics);
                }
            } else {
                double[] alphaTmp = alpha[m].toArray();

                alphaSum[m] = Dirichlet.learnParameters(alphaTmp, topicDocCounts[m], docLengthCounts[m], 1.001, 1.0, 1);
                alpha[m].reset();
                alpha[m].add(alphaTmp);
            }

            logger.info("[alpha: " + formatter.format(alpha[m].get(0)) + "] ");
            logger.info("[alphaSum: " + formatter.format(alphaSum[m]) + "] ");
        }
    }

//    public void temperAlpha(MixWorkerRunnable[] runnables) {
//
//        // First clear the sufficient statistic histograms
//
//        Arrays.fill(docLengthCounts, 0);
//        for (int topic = 0; topic < topicDocCounts.length; topic++) {
//            Arrays.fill(topicDocCounts[topic], 0);
//        }
//
//        for (int thread = 0; thread < numThreads; thread++) {
//            int[] sourceLengthCounts = runnables[thread].getDocLengthCounts();
//            int[][] sourceTopicCounts = runnables[thread].getTopicDocCounts();
//
//            for (int count = 0; count < sourceLengthCounts.length; count++) {
//                if (sourceLengthCounts[count] > 0) {
//                    sourceLengthCounts[count] = 0;
//                }
//            }
//
//            for (int topic = 0; topic < numTopics; topic++) {
//
//                for (int count = 0; count < sourceTopicCounts[topic].length; count++) {
//                    if (sourceTopicCounts[topic][count] > 0) {
//                        sourceTopicCounts[topic][count] = 0;
//                    }
//                }
//            }
//        }
//
//        for (int topic = 0; topic < numTopics; topic++) {
//            alpha.get(topic) = 1.0;
//        }
//        alphaSum = numTopics;
//    }
    public boolean checkConvergence(double convergenceLimit, int prevTopicsNum, int iteration) {

        int[] totalModalityTokens = new int[numModalities];
        int[] totalConvergedTokens = new int[numModalities];
        Arrays.fill(totalModalityTokens, 0);
        Arrays.fill(totalConvergedTokens, 0);
        boolean converged = true;

        for (MixTopicModelTopicAssignment entity : data) {
            for (Byte m = 0; m < numModalities; m++) {
                TopicAssignment document = entity.Assignments[m];

                if (document != null) {
                    FeatureSequence tokens = (FeatureSequence) document.instance.getData();
                    //FeatureSequence topicSequence = (FeatureSequence) document.topicSequence;
                    //int[] topics = topicSequence.getFeatures();

                    for (int position = 0; position < tokens.size(); position++) {

                        totalModalityTokens[m]++;
                        long tmpPreviousTopics = document.prevTopicsSequence[position];

                        long currentMask = (long) topicMask << 63 - topicBits;
                        long topTopic = (tmpPreviousTopics & currentMask);
                        topTopic = topTopic >> 63 - topicBits;
                        int index = 1;
                        boolean isSameTopic = true;
                        while (index < prevTopicsNum && isSameTopic) {
                            index++;
                            currentMask = (long) topicMask << 63 - topicBits * index;
                            long curTopic = tmpPreviousTopics & currentMask;
                            curTopic = curTopic >> 63 - topicBits * index;
                            isSameTopic = curTopic == topTopic;
                            //currentTopic = currentTypeTopicCounts[index] & topicMask;

                        }
                        if (isSameTopic) {
                            totalConvergedTokens[m]++;
                        }

                    }
                }

            }
        }
        for (Byte m = 0; m < numModalities; m++) {
            double rate = (double) totalConvergedTokens[m] / (double) totalModalityTokens[m];
            converged = converged && (rate < convergenceLimit);
            convergenceRates[m][iteration] = rate;
            logger.info("Convergence Rate for modality: " + m + "  Converged/Total: " + totalConvergedTokens[m] + "/" + totalModalityTokens[m] + "  (%):" + rate);
        }
        return converged;
    }

    public void optimizeP(iMixWorkerRunnable[] runnables) {

//          for (int thread = 0; thread < numThreads; thread++) {
//              runnables[thread].getPDistr_Mean();
//          }
//we consider beta known = 1 --> a=(inverse digamma) [lnGx-lnG(1-x)+y(b)]
        // --> a = - 1 / (1/N (Sum(lnXi))), i=1..N , where Xi = mean (pDistr_Mean)
        for (Byte m = 0; m < numModalities; m++) {
            for (Byte i = (byte) (m + 1); i < numModalities; i++) {
                //optimize based on mean & variance
                double sum = 0;
                for (int j = 0; j < pDistr_Mean[m][i].length; j++) {
                    sum += pDistr_Mean[m][i][j];
                }
                double mean = sum / totalDocsPerModality[m];

                //double var = 2 * (1 - mean);
                double a = -1.0 / Math.log(mean);
                double b = 1;

                logger.info("[p:" + m + "_" + i + " mean:" + mean + " a:" + a + " b:" + b + "] ");
                p_a[m][i] = Math.min(a, 3);//a;
                p_a[i][m] = Math.min(a, 3);;
                p_b[m][i] = b;
                p_b[i][m] = b;

            }
        }

        // Now publish the new value
//        for (int thread = 0; thread < numThreads; thread++) {
//            runnables[thread].resetP_a(p_a);
//            runnables[thread].resetP_a(p_b);
//        }
    }

    public void optimizeBeta(iMixWorkerRunnable[] runnables) {

        for (Byte i = 0; i < numModalities; i++) {

            double prevBetaSum = betaSum[i];
            // The histogram starts at count 0, so if all of the
            //  tokens of the most frequent type were assigned to one topic,
            //  we would need to store a maxTypeCount + 1 count.
            int[] countHistogram = new int[maxTypeCount[i] + 1];

            // Now count the number of type/topic pairs that have
            //  each number of tokens.
            int index;
            for (int type = 0; type < numTypes[i]; type++) {
                TIntArrayList counts = typeTopicCounts[i][type];
                index = 0;
                while (index < counts.size() && counts.get(index) > 0) {
                    int count = counts.get(index) >> topicBits;
                    countHistogram[count]++;
                    index++;
                }
            }

            // Figure out how large we need to make the "observation lengths"
            //  histogram.
            int maxTopicSize = 0;
            for (int topic = 0; topic < numTopics; topic++) {
                if (tokensPerTopic[i].get(topic) > maxTopicSize) {
                    maxTopicSize = tokensPerTopic[i].get(topic);
                }
            }

            // Now allocate it and populate it.
            int[] topicSizeHistogram = new int[maxTopicSize + 1];
            for (int topic = 0; topic < numTopics; topic++) {
                topicSizeHistogram[ tokensPerTopic[i].get(topic)]++;
            }

            betaSum[i] = Dirichlet.learnSymmetricConcentration(countHistogram,
                    topicSizeHistogram,
                    numTypes[i],
                    betaSum[i]);
            if (Double.isNaN(betaSum[i])) {
                betaSum[i] = prevBetaSum;
            }
            beta[i] = betaSum[i] / numTypes[i];

            logger.info("[beta: " + formatter.format(beta[i]) + "] ");
        }
        // Now publish the new value
        for (int thread = 0; thread < numThreads; thread++) {
            runnables[thread].resetBeta(beta, betaSum);
        }

    }

    public void estimate() throws IOException {

        long startTime = System.currentTimeMillis();

        iMixWorkerRunnable[] runnables = new iMixWorkerRunnable[numThreads];

        int docsPerThread = data.size() / numThreads;
        int offset = 0;
        pDistr_Mean = new double[numModalities][numModalities][data.size()];
        pDistr_Var = new double[numModalities][numModalities][data.size()];
        for (byte i = 0; i < numModalities; i++) {

            Arrays.fill(this.p_a[i], independentIterations == 0 ? 5d : 0d);
            Arrays.fill(this.p_b[i], independentIterations == 0 ? 1d : 0d);
            //Arrays.fill(this.p_a[i], 5d);
            //Arrays.fill(this.p_b[i], 1d);
        }

        if (numThreads > 1) {

            for (int thread = 0; thread < numThreads; thread++) {
                TIntArrayList[] runnableTotals = new TIntArrayList[numModalities];
                TIntArrayList[][] runnableCounts = new TIntArrayList[numModalities][];

                for (Byte i = 0; i < numModalities; i++) {
                    runnableTotals[i] = new TIntArrayList(tokensPerTopic[i].toArray());
                    //System.arraycopy(tokensPerTopic[i], 0, runnableTotals[i], 0, numTopics);
                    runnableCounts[i] = new TIntArrayList[numTypes[i]];
                    for (int type = 0; type < numTypes[i]; type++) {
                        //int[] counts = new int[typeTopicCounts[i][type].clo.length];
                        //System.arraycopy(typeTopicCounts[i][type], 0, counts, 0, counts.length);
                        runnableCounts[i][type] = new TIntArrayList(typeTopicCounts[i][type].toArray());
                    }
                }
                // some docs may be missing at the end due to integer division
                if (thread == numThreads - 1) {
                    docsPerThread = data.size() - offset;
                }

                Randoms random = null;
                if (randomSeed == -1) {
                    random = new Randoms();
                } else {
                    random = new Randoms(randomSeed);
                }

                runnables[thread] = new iMixWorkerRunnable(numTopics, numIndependentTopics,
                        alpha, alphaSum, beta,
                        random, data,
                        runnableCounts, runnableTotals,
                        offset, docsPerThread,
                        ignoreLabels, numModalities,
                        typeSkewIndexes, skewOn, skewWeight, p_a, p_b);

                runnables[thread].initializeAlphaStatistics(histogramSize);

                offset += docsPerThread;

            }
        } else {

            // If there is only one thread, copy the typeTopicCounts
            //  arrays directly, rather than allocating new memory.
            Randoms random = null;
            if (randomSeed == -1) {
                random = new Randoms();
            } else {
                random = new Randoms(randomSeed);
            }

            runnables[0] = new iMixWorkerRunnable(numTopics, numIndependentTopics,
                    alpha, alphaSum, beta,
                    random, data,
                    typeTopicCounts, tokensPerTopic,
                    offset, docsPerThread,
                    ignoreLabels, numModalities,
                    typeSkewIndexes, skewOn, skewWeight, p_a, p_b);

            runnables[0].initializeAlphaStatistics(histogramSize);

            // If there is only one thread, we 
            //  can avoid communications overhead.
            // This switch informs the thread not to 
            //  gather statistics for its portion of the data.
            runnables[0].makeOnlyThread();
        }

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        for (int iteration = 1; iteration <= numIterations; iteration++) {

            long iterationStart = System.currentTimeMillis();

            if (showTopicsInterval != 0 && iteration != 0 && iteration % showTopicsInterval == 0) {
                logger.info("\n" + displayTopWords(wordsPerTopic, lblsPerTopic, false));
            }

            if (saveStateInterval != 0 && iteration % saveStateInterval == 0) {
                this.printState(new File(stateFilename + '.' + iteration));
            }

            if (saveModelInterval != 0 && iteration % saveModelInterval == 0) {
                this.write(new File(modelFilename + '.' + iteration));
            }

            if (numThreads > 1) {

                // Submit runnables to thread pool
                for (int thread = 0; thread < numThreads; thread++) {

                    if (iteration > burninPeriod && optimizeInterval != 0
                            && iteration % saveSampleInterval == 0) {
                        runnables[thread].collectAlphaStatistics();
                    }

                    logger.fine("submitting thread " + thread);
                    executor.submit(runnables[thread]);
                    //runnables[thread].run();
                }

                // I'm getting some problems that look like 
                //  a thread hasn't started yet when it is first
                //  polled, so it appears to be finished. 
                // This only occurs in very short corpora.
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                }

                boolean finished = false;
                while (!finished) {

                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                    }

                    finished = true;

                    // Are all the threads done?
                    for (int thread = 0; thread < numThreads; thread++) {
                        //logger.info("thread " + thread + " done? " + runnables[thread].isFinished);
                        finished = finished && runnables[thread].isFinished;
                    }

                }

                //System.out.print("[" + (System.currentTimeMillis() - iterationStart) + "] ");
                //synchronize counts
                if (iteration == independentIterations) {
                    logger.info("Topics alignment started");
                    alignTopicsInModalities();
                    for (byte i = 0; i < numModalities; i++) {
                        Arrays.fill(this.p_a[i], 5d);
                        Arrays.fill(this.p_b[i], 1d);
                    }
                    logger.info("Topics alignment finished");
                } else {
                    sumTypeTopicCounts(runnables, iteration > burninPeriod);
                }

                //sumTypeTopicCounts(runnables, iteration > burninPeriod);
                //System.out.print("[" + (System.currentTimeMillis() - iterationStart) + "] ");
                //place synchronized values back to threads
                for (int thread = 0; thread < numThreads; thread++) {

                    for (Byte i = 0; i < numModalities; i++) {
                        runnables[thread].getTokensPerTopic()[i].reset();
                        runnables[thread].getTokensPerTopic()[i].add(tokensPerTopic[i].toArray());

                        //System.arraycopy(tokensPerTopic[i], 0, runnableTotals[i], 0, numTopics);
                    }

                    runnables[thread].resetSkewWeight(skewWeight);

                    //int[][][] runnableCounts = runnables[thread].getTypeTopicCounts();
                    for (Byte i = 0; i < numModalities; i++) {
                        for (int type = 0; type < numTypes[i]; type++) {
                            runnables[thread].getTypeTopicCounts()[i][type].reset();
                            runnables[thread].getTypeTopicCounts()[i][type].add(typeTopicCounts[i][type].toArray());
                            //runnables[thread].getTypeTopicCounts()[i][type] = (TIntArrayList) typeTopicCounts[i][type].clone();
//                            int[] targetCounts = runnables[thread].getTypeTopicCounts()[i][type];
//                            int[] sourceCounts = typeTopicCounts[i][type].toArray();
//
//                            int index = 0;
//                            while (index < sourceCounts.length) {
//
//                                if (sourceCounts[index] != 0) {
//                                    targetCounts[index] = sourceCounts[index];
//                                } else if (targetCounts[index] != 0) {
//                                    targetCounts[index] = 0;
//                                } else {
//                                    break;
//                                }
//
//                                index++;
//                            }
                            //System.arraycopy(typeTopicCounts[type], 0, counts, 0, counts.length);
                        }
                    }

                }
            } else {
                if (iteration > burninPeriod && optimizeInterval != 0
                        && iteration % saveSampleInterval == 0) {
                    runnables[0].collectAlphaStatistics();
                }
                runnables[0].run();
            }

            long elapsedMillis = System.currentTimeMillis() - iterationStart;
            if (elapsedMillis < 1000) {
                logger.info(elapsedMillis + "ms ");
            } else {
                logger.info((elapsedMillis / 1000) + "s ");
            }

            if (iteration > burninPeriod && optimizeInterval != 0
                    && iteration % optimizeInterval == 0) {

                optimizeAlpha(runnables);
                optimizeBeta(runnables);
                optimizeP(runnables);

                logger.info("[O " + (System.currentTimeMillis() - iterationStart) + "] ");
            }

            if (iteration % 10 == 0) {
                if (printLogLikelihood) {
                    checkConvergence(0.8, 3, iteration / 10);
                    for (Byte i = 0; i < numModalities; i++) {
                        double ll = modelLogLikelihood()[i] / totalTokens[i];
                        perplexities[i][iteration / 10] = ll;
                        logger.info("<" + iteration + "> modality<" + i + "> LL/token: " + formatter.format(ll)); //LL for eachmodality
                    }
                } else {
                    logger.info("<" + iteration + ">");
                }
            }
        }

        executor.shutdownNow();

        long seconds = Math.round((System.currentTimeMillis() - startTime) / 1000.0);
        long minutes = seconds / 60;
        seconds %= 60;
        long hours = minutes / 60;
        minutes %= 60;
        long days = hours / 24;
        hours %= 24;

        StringBuilder timeReport = new StringBuilder();
        timeReport.append("\nTotal time: ");
        if (days != 0) {
            timeReport.append(days);
            timeReport.append(" days ");
        }
        if (hours != 0) {
            timeReport.append(hours);
            timeReport.append(" hours ");
        }
        if (minutes != 0) {
            timeReport.append(minutes);
            timeReport.append(" minutes ");
        }
        timeReport.append(seconds);
        timeReport.append(" seconds");

        logger.info(timeReport.toString());
    }

    public void printTopWords(File file, int numWords, int numLabels, boolean useNewLines) throws IOException {
        PrintStream out = new PrintStream(file);
        printTopWords(out, numWords, numLabels, useNewLines);

        out.close();
    }

    public void saveTopics(String SQLLiteDB, String experimentId) {

        Connection connection = null;
        Statement statement = null;
        try {
            // create a database connection
            if (!SQLLiteDB.isEmpty()) {
                connection = DriverManager.getConnection(SQLLiteDB);
                statement = connection.createStatement();
                statement.setQueryTimeout(30);  // set timeout to 30 sec.
                //statement.executeUpdate("drop table if exists TopicAnalysis");
                statement.executeUpdate("create table if not exists TopicAnalysis (TopicId integer, ItemType integer, Item nvarchar(100), Counts double, ExperimentId nvarchar(50), DiscrWeight) ");
                String deleteSQL = String.format("Delete from TopicAnalysis where  ExperimentId = '%s'", experimentId);
                statement.executeUpdate(deleteSQL);

                PreparedStatement bulkInsert = null;
                String sql = "insert into TopicAnalysis values(?,?,?,?,?,? );";

                try {
                    connection.setAutoCommit(false);
                    bulkInsert = connection.prepareStatement(sql);

                    ArrayList<ArrayList<TreeSet<IDSorter>>> topicSortedWords = new ArrayList<ArrayList<TreeSet<IDSorter>>>(numModalities);

                    for (Byte m = 0; m < numModalities; m++) {
                        topicSortedWords.add(getSortedWords(m));
                    }

                    for (int topic = 0; topic < numTopics; topic++) {
                        for (Byte m = 0; m < numModalities; m++) {
                            TreeSet<IDSorter> sortedWords = topicSortedWords.get(m).get(topic);

                            int word = 1;
                            Iterator<IDSorter> iterator = sortedWords.iterator();

                            while (iterator.hasNext() && word < 20) {
                                IDSorter info = iterator.next();
                                bulkInsert.setInt(1, topic);
                                bulkInsert.setInt(2, m);
                                bulkInsert.setString(3, alphabet[m].lookupObject(info.getID()).toString());
                                bulkInsert.setDouble(4, info.getWeight());
                                bulkInsert.setString(5, experimentId);
                                bulkInsert.setDouble(6, 1);
                                bulkInsert.executeUpdate();

                                word++;
                            }

                        }

                    }

                    // also find and write phrases 
                    TObjectIntHashMap<String>[] phrases = findTopicPhrases();

                    for (int ti = 0; ti < numTopics; ti++) {

                        // Print phrases
                        Object[] keys = phrases[ti].keys();
                        int[] values = phrases[ti].values();
                        double counts[] = new double[keys.length];
                        for (int i = 0; i < counts.length; i++) {
                            counts[i] = values[i];
                        }
                        // double countssum = MatrixOps.sum(counts);
                        Alphabet alph = new Alphabet(keys);
                        RankedFeatureVector rfv = new RankedFeatureVector(alph, counts);
                        int max = rfv.numLocations() < 20 ? rfv.numLocations() : 20;
                        for (int ri = 0; ri < max; ri++) {
                            int fi = rfv.getIndexAtRank(ri);

                            double count = counts[fi];// / countssum;
                            String phraseStr = alph.lookupObject(fi).toString();

                            bulkInsert.setInt(1, ti);
                            bulkInsert.setInt(2, -1);
                            bulkInsert.setString(3, phraseStr);
                            bulkInsert.setDouble(4, count);
                            bulkInsert.setString(5, experimentId);
                            bulkInsert.setDouble(6, 1);
                            bulkInsert.executeUpdate();
                        }

                    }
                    connection.commit();
//                if (!sql.equals("")) {
//                    statement.executeUpdate(sql);
//                }

//                    if (!sql.equals("")) {
//                        statement.executeUpdate(sql);
//                    }
                } catch (SQLException e) {

                    if (connection != null) {
                        try {
                            System.err.print("Transaction is being rolled back");
                            connection.rollback();
                        } catch (SQLException excep) {
                            System.err.print("Error in insert topicAnalysis");
                        }
                    }
                } finally {

                    if (bulkInsert != null) {
                        bulkInsert.close();
                    }
                    connection.setAutoCommit(true);
                }
            }
        } catch (SQLException e) {
            // if the error message is "out of memory", 
            // it probably means no database file is found
            System.err.println(e.getMessage());
        } finally {
            try {
                if (connection != null) {
                    connection.close();
                }
            } catch (SQLException e) {
                // connection close failed.
                System.err.println(e);
            }

        }
    }

    /**
     * Return an array of sorted sets (one set per topic). Each set contains
     * IDSorter objects with integer keys into the alphabet. To get direct
     * access to the Strings, use getTopWords().
     */
    public ArrayList<TreeSet<IDSorter>> getSortedWords(int modality) {

        ArrayList<TreeSet<IDSorter>> topicSortedWords = new ArrayList<TreeSet<IDSorter>>(numTopics);

        // Initialize the tree sets
        for (int topic = 0; topic < numTopics; topic++) {
            topicSortedWords.add(new TreeSet<IDSorter>());
        }

        int nonZeroCnt = 1;

//        double skewSum = 0;
//        double skewWeight = 1;
//
//        if (skewOn == SkewType.TextAndLabels) {
//
//            for (int i = 0; i < typeSkewIndexes.length; i++) {
//                if (typeSkewIndexes[i] > 0) {
//                    nonZeroCnt++;
//                }
//                skewSum += typeSkewIndexes[i];
//            }
//
//            skewWeight = (double) 1 / (1 + skewSum / (double) nonZeroCnt);
//        } NOT NEEDED 
        // Collect counts
        for (int type = 0; type < numTypes[modality]; type++) {

            int[] topicCounts = typeTopicCounts[modality][type].toArray();

            int index = 0;
            while (index < topicCounts.length
                    && topicCounts[index] > 0) {

                int topic = topicCounts[index] & topicMask;
                int count = topicCounts[index] >> topicBits;

                if (skewOn == SkewType.TextAndLabels && modality == 0 || (skewOn == SkewType.LabelsOnly && modality > 0)) {
                    topicSortedWords.get(topic).add(new IDSorter(type, skewWeight[modality] * count * (1 + typeSkewIndexes[modality][type]))); //TODO Check it
                } else {
                    topicSortedWords.get(topic).add(new IDSorter(type, count));
                }

                index++;
            }
        }

        return topicSortedWords;
    }

    /**
     * Return an array (one element for each topic) of arrays of words, which
     * are the most probable words for that topic in descending order. These are
     * returned as Objects, but will probably be Strings.
     *
     * @param numWords The maximum length of each topic's array of words (may be
     * less).
     */
    public Object[][] getTopWords(int numWords, int modality) {

        ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords(modality);
        Object[][] result = new Object[numTopics][];

        for (int topic = 0; topic < numTopics; topic++) {

            TreeSet<IDSorter> sortedWords = topicSortedWords.get(topic);

            // How many words should we report? Some topics may have fewer than
            //  the default number of words with non-zero weight.
            int limit = numWords;
            if (sortedWords.size() < numWords) {
                limit = sortedWords.size();
            }

            result[topic] = new Object[limit];

            Iterator<IDSorter> iterator = sortedWords.iterator();
            for (int i = 0; i < limit; i++) {
                IDSorter info = iterator.next();
                result[topic][i] = alphabet[modality].lookupObject(info.getID());
            }
        }

        return result;
    }

    public void printTopWords(PrintStream out, int numWords, int numLabels, boolean usingNewLines) {
        out.print(displayTopWords(numWords, numLabels, usingNewLines));
    }

    public String displayTopWords(int numWords, int numLabels, boolean usingNewLines) {

        StringBuilder out = new StringBuilder();
        ArrayList<ArrayList<TreeSet<IDSorter>>> topicSortedWords = new ArrayList<ArrayList<TreeSet<IDSorter>>>(4);

        for (Byte m = 0; m < numModalities; m++) {
            topicSortedWords.add(getSortedWords(m));
        }
        // Print results for each topic

        for (int topic = 0; topic < numTopics; topic++) {
            for (Byte m = 0; m < numModalities; m++) {
                TreeSet<IDSorter> sortedWords = topicSortedWords.get(m).get(topic);

                int word = 1;
                Iterator<IDSorter> iterator = sortedWords.iterator();
                if (usingNewLines) {
                    out.append(topic + "\t" + formatter.format(alpha[m].get(topic)) + "\n");
                    while (iterator.hasNext() && word < numWords) {
                        IDSorter info = iterator.next();
                        out.append(alphabet[m].lookupObject(info.getID()) + "\t" + formatter.format(info.getWeight()) + "\n");
                        word++;
                    }

                } else {
                    out.append(topic + "\t" + formatter.format(alpha[m].get(topic)) + "\t");

                    while (iterator.hasNext() && word < numWords) {
                        IDSorter info = iterator.next();
                        out.append(alphabet[m].lookupObject(info.getID()) + " ");
                        word++;
                    }

                }
            }
            out.append("\n");
        }
        return out.toString();
    }

    public void topicXMLReport(PrintWriter out, int numWords, int numLabels) {

        ArrayList<ArrayList<TreeSet<IDSorter>>> topicSortedWords = new ArrayList<ArrayList<TreeSet<IDSorter>>>(4);

        for (Byte m = 0; m < numModalities; m++) {
            topicSortedWords.add(getSortedWords(m));
        }
        out.println("<?xml version='1.0' ?>");
        out.println("<topicModel>");
        for (int topic = 0; topic < numTopics; topic++) {
            for (Byte m = 0; m < numModalities; m++) {
                out.println("  <topic id='" + topic + "' alpha='" + alpha[m].get(topic) + "' modality='" + m
                        + "' totalTokens='" + tokensPerTopic[m].get(topic)
                        + "'>");
                int word = 1;
                Iterator<IDSorter> iterator = topicSortedWords.get(m).get(topic).iterator();
                while (iterator.hasNext() && word < numWords) {
                    IDSorter info = iterator.next();
                    out.println("	<word rank='" + word + "'>"
                            + alphabet[m].lookupObject(info.getID())
                            + "</word>");
                    word++;
                }

            }
            out.println("  </topic>");
        }
        out.println("</topicModel>");
    }

    public TObjectIntHashMap<String>[] findTopicPhrases() {
        int numTopics = this.getNumTopics();

        TObjectIntHashMap<String>[] phrases = new TObjectIntHashMap[numTopics];
        Alphabet alphabet = this.getAlphabet()[0];

        // Get counts of phrases in topics
        // Search bigrams within corpus to see if they have been assigned to the same topic, adding them to topic phrases
        for (int ti = 0; ti < numTopics; ti++) {
            phrases[ti] = new TObjectIntHashMap<String>();
        }
        for (int di = 0; di < this.getData().size(); di++) {

            TopicAssignment t = this.getData().get(di).Assignments[0];
            if (t != null) {
                Instance instance = t.instance;
                FeatureSequence fvs = (FeatureSequence) instance.getData();
                boolean withBigrams = false;
                if (fvs instanceof FeatureSequenceWithBigrams) {
                    withBigrams = true;
                }
                int prevtopic = -1;
                int prevfeature = -1;
                int topic = -1;
                StringBuffer sb = null;
                int feature = -1;
                int doclen = fvs.size();
                for (int pi = 0; pi < doclen; pi++) {
                    feature = fvs.getIndexAtPosition(pi);
                    topic = t.topicSequence.getIndexAtPosition(pi);
                    if (topic == prevtopic && (!withBigrams || ((FeatureSequenceWithBigrams) fvs).getBiIndexAtPosition(pi) != -1)) {
                        if (sb == null) {
                            sb = new StringBuffer(alphabet.lookupObject(prevfeature).toString() + " " + alphabet.lookupObject(feature));
                        } else {
                            sb.append(" ");
                            sb.append(alphabet.lookupObject(feature));
                        }
                    } else if (sb != null) {
                        String sbs = sb.toString();
                        //logger.info ("phrase:"+sbs);
                        if (phrases[prevtopic].get(sbs) == 0) {
                            phrases[prevtopic].put(sbs, 0);
                        }
                        phrases[prevtopic].increment(sbs);
                        prevtopic = prevfeature = -1;
                        sb = null;
                    } else {
                        prevtopic = topic;
                        prevfeature = feature;
                    }
                }
            }
        }

        return phrases;
    }

    public void topicPhraseXMLReport(PrintWriter out, int numWords) {

        //Phrases only for modality 0 --> text
        int numTopics = this.getNumTopics();
        Alphabet alphabet = this.getAlphabet()[0];

        TObjectIntHashMap<String>[] phrases = findTopicPhrases();
        // phrases[] now filled with counts

        // Now start printing the XML
        out.println("<?xml version='1.0' ?>");
        out.println("<topics>");

        ArrayList<TreeSet<IDSorter>> topicSortedWords = getSortedWords(0);
        double[] probs = new double[alphabet.size()];
        for (int ti = 0; ti < numTopics; ti++) {
            out.print("  <topic id=\"" + ti + "\" alpha=\"" + alpha[0].get(ti)
                    + "\" totalTokens=\"" + tokensPerTopic[0].get(ti) + "\" ");

            // For gathering <term> and <phrase> output temporarily 
            // so that we can get topic-title information before printing it to "out".
            ByteArrayOutputStream bout = new ByteArrayOutputStream();
            PrintStream pout = new PrintStream(bout);
            // For holding candidate topic titles
            AugmentableFeatureVector titles = new AugmentableFeatureVector(new Alphabet());

            // Print words
            int word = 1;
            Iterator<IDSorter> iterator = topicSortedWords.get(ti).iterator();
            while (iterator.hasNext() && word < numWords) {
                IDSorter info = iterator.next();
                pout.println("	<word weight=\"" + (info.getWeight() / tokensPerTopic[0].get(ti)) + "\" count=\"" + Math.round(info.getWeight()) + "\">"
                        + alphabet.lookupObject(info.getID())
                        + "</word>");
                word++;
                if (word < 20) // consider top 20 individual words as candidate titles
                {
                    titles.add(alphabet.lookupObject(info.getID()), info.getWeight());
                }
            }

            /*
             for (int type = 0; type < alphabet.size(); type++)
             probs[type] = this.getCountFeatureTopic(type, ti) / (double)this.getCountTokensPerTopic(ti);
             RankedFeatureVector rfv = new RankedFeatureVector (alphabet, probs);
             for (int ri = 0; ri < numWords; ri++) {
             int fi = rfv.getIndexAtRank(ri);
             pout.println ("	  <term weight=\""+probs[fi]+"\" count=\""+this.getCountFeatureTopic(fi,ti)+"\">"+alphabet.lookupObject(fi)+	"</term>");
             if (ri < 20) // consider top 20 individual words as candidate titles
             titles.add(alphabet.lookupObject(fi), this.getCountFeatureTopic(fi,ti));
             }
             */
            // Print phrases
            Object[] keys = phrases[ti].keys();
            int[] values = phrases[ti].values();
            double counts[] = new double[keys.length];
            for (int i = 0; i < counts.length; i++) {
                counts[i] = values[i];
            }
            double countssum = MatrixOps.sum(counts);
            Alphabet alph = new Alphabet(keys);
            RankedFeatureVector rfv = new RankedFeatureVector(alph, counts);
            int max = rfv.numLocations() < numWords ? rfv.numLocations() : numWords;
            for (int ri = 0; ri < max; ri++) {
                int fi = rfv.getIndexAtRank(ri);
                pout.println("	<phrase weight=\"" + counts[fi] / countssum + "\" count=\"" + values[fi] + "\">" + alph.lookupObject(fi) + "</phrase>");
                // Any phrase count less than 20 is simply unreliable
                if (ri < 20 && values[fi] > 20) {
                    titles.add(alph.lookupObject(fi), 100 * values[fi]); // prefer phrases with a factor of 100 
                }
            }

            // Select candidate titles
            StringBuffer titlesStringBuffer = new StringBuffer();
            rfv = new RankedFeatureVector(titles.getAlphabet(), titles);
            int numTitles = 10;
            for (int ri = 0; ri < numTitles && ri < rfv.numLocations(); ri++) {
                // Don't add redundant titles
                if (titlesStringBuffer.indexOf(rfv.getObjectAtRank(ri).toString()) == -1) {
                    titlesStringBuffer.append(rfv.getObjectAtRank(ri));
                    if (ri < numTitles - 1) {
                        titlesStringBuffer.append(", ");
                    }
                } else {
                    numTitles++;
                }
            }
            out.println("titles=\"" + titlesStringBuffer.toString() + "\">");
            out.print(bout.toString());
            out.println("  </topic>");
        }
        out.println("</topics>");
    }

    /**
     * Write the internal representation of type-topic counts (count/topic pairs
     * in descending order by count) to a file.
     */
    public void printTypeTopicCounts(File file) throws IOException {
        PrintWriter out = new PrintWriter(new FileWriter(file));
        for (Byte m = 0; m < numModalities; m++) {
            for (int type = 0; type < numTypes[m]; type++) {

                StringBuilder buffer = new StringBuilder();

                buffer.append(type + " " + alphabet[m].lookupObject(type));

                int[] topicCounts = typeTopicCounts[m][type].toArray();

                int index = 0;
                while (index < topicCounts.length
                        && topicCounts[index] > 0) {

                    int topic = topicCounts[index] & topicMask;
                    int count = topicCounts[index] >> topicBits;

                    buffer.append(" " + topic + ":" + count);

                    index++;
                }

                out.println(buffer);
            }
        }
        out.close();
    }

    public void printTopicWordWeights(File file) throws IOException {
        PrintWriter out = new PrintWriter(new FileWriter(file));
        printTopicWordWeights(out);
        out.close();
    }

    /**
     * Print an unnormalized weight for every word in every topic. Most of these
     * will be equal to the smoothing parameter beta.
     */
    public void printTopicWordWeights(PrintWriter out) throws IOException {
        // Probably not the most efficient way to do this...

        for (int topic = 0; topic < numTopics; topic++) {
            for (Byte m = 0; m < numModalities; m++) {
                for (int type = 0; type < numTypes[m]; type++) {

                    int[] topicCounts = typeTopicCounts[m][type].toArray();

                    double weight = beta[m];

                    int index = 0;
                    while (index < topicCounts.length
                            && topicCounts[index] > 0) {

                        int currentTopic = topicCounts[index] & topicMask;

                        if (currentTopic == topic) {
                            weight += topicCounts[index] >> topicBits;
                            break;
                        }

                        index++;
                    }

                    out.println(topic + "\t" + alphabet[m].lookupObject(type) + "\t" + weight);

                }
            }
        }
    }

    /**
     * Get the smoothed distribution over topics for a training instance.
     */
    public double[] getTopicProbabilities(int instanceID, byte modality) {
        LabelSequence topics = data.get(instanceID).Assignments[modality].topicSequence;
        return getTopicProbabilities(topics, modality);
    }

    /**
     * Get the smoothed distribution over topics for a topic sequence, which may
     * be from the training set or from a new instance with topics assigned by
     * an inferencer.
     */
    public double[] getTopicProbabilities(LabelSequence topics, byte modality) {
        double[] topicDistribution = new double[numTopics];

        // Loop over the tokens in the document, counting the current topic
        //  assignments.
        for (int position = 0; position < topics.getLength(); position++) {
            topicDistribution[ topics.getIndexAtPosition(position)]++;
        }

        // Add the smoothing parameters and normalize
        double sum = 0.0;
        for (int topic = 0; topic < numTopics; topic++) {
            topicDistribution[topic] += alpha[modality].get(topic);
            sum += topicDistribution[topic];
        }

        // And normalize
        for (int topic = 0; topic < numTopics; topic++) {
            topicDistribution[topic] /= sum;
        }

        return topicDistribution;
    }

    public void printDocumentTopics(File file) throws IOException {
        PrintWriter out = new PrintWriter(new FileWriter(file));
        printDocumentTopics(out);
        out.close();
    }

    public void printDocumentTopics(PrintWriter out) {
        printDocumentTopics(out, 0.0, -1, "", "", 0.33);
    }

    /**
     * @param out	A print writer
     * @param threshold Only print topics with proportion greater than this
     * number
     * @param max	Print no more than this many topics
     */
    public void printDocumentTopics(PrintWriter out, double threshold, int max, String SQLLiteDB, String experimentId, double lblWeight) {
        out.print("#doc name topic proportion ...\n");
        int[] docLen = new int[numModalities];

        int[][] topicCounts = new int[numModalities][numTopics];

        IDSorter[] sortedTopics = new IDSorter[numTopics];
        for (int topic = 0; topic < numTopics; topic++) {
            // Initialize the sorters with dummy values
            sortedTopics[topic] = new IDSorter(topic, topic);
        }

        if (max < 0 || max > numTopics) {
            max = numTopics;
        }

        Connection connection = null;
        Statement statement = null;
        try {
            // create a database connection
            if (!SQLLiteDB.isEmpty()) {
                connection = DriverManager.getConnection(SQLLiteDB);
                statement = connection.createStatement();
                statement.setQueryTimeout(30);  // set timeout to 30 sec.
                // statement.executeUpdate("drop table if exists TopicsPerDoc");
                statement.executeUpdate("create table if not exists TopicsPerDoc (DocId nvarchar(50), TopicId Integer, Weight Double , ExperimentId nvarchar(50)) ");
                statement.executeUpdate(String.format("Delete from TopicsPerDoc where  ExperimentId = '%s'", experimentId));
            }
            PreparedStatement bulkInsert = null;
            String sql = "insert into TopicsPerDoc values(?,?,?,? );";

            try {
                connection.setAutoCommit(false);
                bulkInsert = connection.prepareStatement(sql);

                for (int doc = 0; doc < data.size(); doc++) {
                    int cntEnd = ignoreLabels ? 1 : numModalities;
                    StringBuilder builder = new StringBuilder();
                    builder.append(doc);
                    builder.append("\t");

                    String docId = "no-name";

                    docId = data.get(doc).EntityId.toString();

                    builder.append(docId);
                    builder.append("\t");

                    for (Byte m = 0; m < cntEnd; m++) {
                        if (data.get(doc).Assignments[m] != null) {
                            Arrays.fill(topicCounts[m], 0);
                            LabelSequence topicSequence = (LabelSequence) data.get(doc).Assignments[m].topicSequence;
                            int[] currentDocTopics = topicSequence.getFeatures();
                            docLen[m] = currentDocTopics.length;

                            // Count up the tokens
                            for (int token = 0; token < docLen[m]; token++) {
                                topicCounts[m][ currentDocTopics[token]]++;
                            }
                        }
                    }

                    // And normalize
                    for (int topic = 0; topic < numTopics; topic++) {
                        double topicProportion = 0;
                        for (Byte m = 0; m < cntEnd; m++) {
                            topicProportion += (double) alpha[m].get(topic) / alphaSum[m] + (double) topicCounts[m][topic] / docLen[m];
                        }
                        sortedTopics[topic].set(topic, (topicProportion / (cntEnd + 1)));

                    }

                    Arrays.sort(sortedTopics);

//      statement.executeUpdate("insert into person values(1, 'leo')");
//      statement.executeUpdate("insert into person values(2, 'yui')");
//      ResultSet rs = statement.executeQuery("select * from person");
                    for (int i = 0; i < max; i++) {
                        if (sortedTopics[i].getWeight() < threshold) {
                            break;
                        }

                        builder.append(sortedTopics[i].getID() + "\t"
                                + sortedTopics[i].getWeight() + "\t");
                        out.println(builder);

                        if (!SQLLiteDB.isEmpty()) {
                            //  sql += String.format(Locale.ENGLISH, "insert into TopicsPerDoc values('%s',%d,%.4f,'%s' );", docId, sortedTopics[i].getID(), sortedTopics[i].getWeight(), experimentId);
                            bulkInsert.setString(1, docId);
                            bulkInsert.setInt(2, sortedTopics[i].getID());
                            bulkInsert.setDouble(3, (double) Math.round(sortedTopics[i].getWeight() * 10000) / 10000);
                            bulkInsert.setString(4, experimentId);
                            bulkInsert.executeUpdate();

                        }

                    }

//                    if ((doc / 10) * 10 == doc && !sql.isEmpty()) {
//                        statement.executeUpdate(sql);
//                        sql = "";
//                    }
                }
                if (!SQLLiteDB.isEmpty()) {
                    connection.commit();
                }
//                if (!sql.isEmpty()) {
//                    statement.executeUpdate(sql);
//                }
//

            } catch (SQLException e) {

                if (connection != null) {
                    try {
                        System.err.print("Transaction is being rolled back");
                        connection.rollback();
                    } catch (SQLException excep) {
                        System.err.print("Error in insert topicAnalysis");
                    }
                }
            } finally {

                if (bulkInsert != null) {
                    bulkInsert.close();
                }
                connection.setAutoCommit(true);
            }

        } catch (SQLException e) {
            // if the error message is "out of memory", 
            // it probably means no database file is found
            System.err.println(e.getMessage());
        } finally {
            try {
                if (connection != null) {
                    connection.close();
                }
            } catch (SQLException e) {
                // connection close failed.
                System.err.println(e);
            }

        }
    }

    public void printState(File f) throws IOException {
        PrintStream out = new PrintStream(new GZIPOutputStream(new BufferedOutputStream(new FileOutputStream(f))));
        printState(out);
        out.close();
    }

    public void printState(PrintStream out) {

        out.println("#doc source pos typeindex type topic");
        out.print("#alpha : ");
        for (Byte m = 0; m < numModalities; m++) {
            out.println("modality:" + m);
            for (int topic = 0; topic < numTopics; topic++) {
                out.print(alpha[m].get(topic) + " ");
            }
        }
        out.println();
        out.println("#beta : " + beta);

        for (int doc = 0; doc < data.size(); doc++) {
            for (Byte m = 0; m < numModalities; m++) {
                if (data.get(doc).Assignments[m] != null) {
                    FeatureSequence tokenSequence = (FeatureSequence) data.get(doc).Assignments[m].instance.getData();
                    LabelSequence topicSequence = (LabelSequence) data.get(doc).Assignments[m].topicSequence;
                    String source = "NA";
                    if (data.get(doc).Assignments[m].instance.getSource() != null) {
                        source = data.get(doc).Assignments[m].instance.getSource().toString();
                    }

                    Formatter output = new Formatter(new StringBuilder(), Locale.US);

                    for (int pi = 0; pi < topicSequence.getLength(); pi++) {
                        int type = tokenSequence.getIndexAtPosition(pi);
                        int topic = topicSequence.getIndexAtPosition(pi);
                        output.format("%d %s %d %d %s %d\n", doc, source, pi, type, alphabet[m].lookupObject(type), topic);

                    }

                    out.print(output);
                }
            }
        }
    }

    public double[] modelLogLikelihood() {

        double[] logLikelihood = new double[numModalities];
        Arrays.fill(logLikelihood, 0);

        // The likelihood of the model is a combination of a 
        // Dirichlet-multinomial for the words in each topic
        // and a Dirichlet-multinomial for the topics in each
        // document.
        // The likelihood function of a dirichlet multinomial is
        //	 Gamma( sum_i alpha_i )	 prod_i Gamma( alpha_i + N_i )
        //	prod_i Gamma( alpha_i )	  Gamma( sum_i (alpha_i + N_i) )
        // So the log likelihood is 
        //	logGamma ( sum_i alpha_i ) - logGamma ( sum_i (alpha_i + N_i) ) + 
        //	 sum_i [ logGamma( alpha_i + N_i) - logGamma( alpha_i ) ]
        // Do the documents first
        double[] topicLogGammas = new double[numTopics];
        int[] docTopics;

        for (Byte m = 0; m < numModalities; m++) {
            for (int topic = 0; topic < numTopics; topic++) {
                topicLogGammas[ topic] = Dirichlet.logGammaStirling(alpha[m].get(topic));
            }
            int[] topicCounts = new int[numTopics];

            int modalityCnt = 0;
            for (int doc = 0; doc < data.size(); doc++) {
                if (data.get(doc).Assignments[m] != null) {
                    LabelSequence topicSequence = (LabelSequence) data.get(doc).Assignments[m].topicSequence;

                    docTopics = topicSequence.getFeatures();
                    if (docTopics.length > 0) {
                        for (int token = 0; token < docTopics.length; token++) {
                            topicCounts[ docTopics[token]]++;
                        }

                        for (int topic = 0; topic < numTopics; topic++) {
                            if (topicCounts[topic] > 0) {
                                logLikelihood[m] += (Dirichlet.logGammaStirling(alpha[m].get(topic) + topicCounts[topic])
                                        - topicLogGammas[ topic]);
                            }
                        }

                        // subtract the (count + parameter) sum term
                        logLikelihood[m] -= Dirichlet.logGammaStirling(alphaSum[m] + docTopics.length);
                        modalityCnt++;

                    }
                    Arrays.fill(topicCounts, 0);
                }
            }

            // add the parameter sum term
            logLikelihood[m] += modalityCnt * Dirichlet.logGammaStirling(alphaSum[m]);

            // And the topics
            // Count the number of type-topic pairs that are not just (logGamma(beta) - logGamma(beta))
            int nonZeroTypeTopics = 0;

            for (int type = 0; type < numTypes[m]; type++) {
                // reuse this array as a pointer

                topicCounts = typeTopicCounts[m][type].toArray();

                int index = 0;
                while (index < topicCounts.length
                        && topicCounts[index] > 0) {
                    int topic = topicCounts[index] & topicMask;
                    int count = topicCounts[index] >> topicBits;

                    nonZeroTypeTopics++;
                    logLikelihood[m] += Dirichlet.logGammaStirling(beta[m] + count);

                    if (Double.isNaN(logLikelihood[m])) {
                        logger.warning("NaN in log likelihood calculation" + " for modality: " + m);
                        logLikelihood[m] = 0;
                        break;
                    } else if (Double.isInfinite(logLikelihood[m])) {
                        logger.warning("infinite log likelihood" + " for modality: " + m);
                        logLikelihood[m] = 0;
                        break;
                    }

                    index++;
                }
            }

            for (int topic = 0; topic < numTopics; topic++) {
                logLikelihood[m] -= Dirichlet.logGammaStirling((beta[m] * numTypes[m])
                        + tokensPerTopic[m].get(topic));

                if (Double.isNaN(logLikelihood[m])) {
                    logger.info("NaN after topic " + topic + " " + tokensPerTopic[m].get(topic) + " for modality: " + m);
                    logLikelihood[m] = 0;
                    break;
                } else if (Double.isInfinite(logLikelihood[m])) {
                    logger.info("Infinite value after topic " + topic + " " + tokensPerTopic[m].get(topic) + " for modality: " + m);
                    logLikelihood[m] = 0;
                    break;
                }

            }

            // logGamma(|V|*beta) for every topic
            logLikelihood[m] += Dirichlet.logGammaStirling(beta[m] * numTypes[m]) * numTopics;

            // logGamma(beta) for all type/topic pairs with non-zero count
            logLikelihood[m] -= Dirichlet.logGammaStirling(beta[m]) * nonZeroTypeTopics;

            if (Double.isNaN(logLikelihood[m])) {
                logger.info("at the end");
                logLikelihood[m] = 0;

            } else if (Double.isInfinite(logLikelihood[m])) {
                logger.info("Infinite value beta " + beta[m] + " * " + numTypes[m]);
                logLikelihood[m] = 0;

            }
        }

        return logLikelihood;
    }
    /**
     * Return a tool for estimating topic distributions for new documents
     * //TODO: build a MixTopicInferencer
     */
//    public TopicInferencer getInferencer() {
////        return new TopicInferencer(typeTopicCounts[0], tokensPerTopic[0].toArray(),
////                data.get(0).Assignments[0].instance.getDataAlphabet(),
////                alpha.toArray(), beta[0], betaSum[0]);
//    }
    /**
     * Return a tool for evaluating the marginal probability of new documents
     * under this model //TODO: build a MixMarginalProbEstimator
     */
//    public MarginalProbEstimator getProbEstimator() {
////        return new MarginalProbEstimator(numTopics, alpha.toArray(), alphaSum, beta[0],
////                typeTopicCounts[0], tokensPerTopic[0].toArray());
//    }
    // Serialization
    private static final long serialVersionUID = 1;
    private static final int CURRENT_SERIAL_VERSION = 0;
    private static final int NULL_INTEGER = -1;

    private void writeObject(ObjectOutputStream out) throws IOException {
        out.writeInt(CURRENT_SERIAL_VERSION);

        out.writeObject(data);
        out.writeObject(alphabet);
        out.writeObject(topicAlphabet);

        out.writeInt(numTopics);

        out.writeInt(topicMask);
        out.writeInt(topicBits);

        out.writeObject(numTypes);

        out.writeObject(alpha);
        out.writeObject(alphaSum);
        out.writeObject(beta);
        out.writeObject(betaSum);

        out.writeObject(typeTopicCounts);
        out.writeObject(tokensPerTopic);

        //out.writeObject(docLengthCounts);
        //out.writeObject(topicDocCounts);
        out.writeInt(numIterations);
        out.writeInt(burninPeriod);
        out.writeInt(saveSampleInterval);
        out.writeInt(optimizeInterval);
        out.writeInt(showTopicsInterval);
        out.writeInt(wordsPerTopic);

        out.writeInt(saveStateInterval);
        out.writeObject(stateFilename);

        out.writeInt(saveModelInterval);
        out.writeObject(modelFilename);

        out.writeInt(randomSeed);
        out.writeObject(formatter);
        out.writeBoolean(printLogLikelihood);

        out.writeInt(numThreads);
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {

        int version = in.readInt();

        data = (ArrayList<MixTopicModelTopicAssignment>) in.readObject();
        alphabet = (Alphabet[]) in.readObject();
        topicAlphabet = (LabelAlphabet) in.readObject();

        numTopics = in.readInt();

        topicMask = in.readInt();
        topicBits = in.readInt();

        numTypes = (int[]) in.readObject();

        alpha = (TDoubleArrayList[]) in.readObject();
        alphaSum = (double[]) in.readObject();
        beta = (double[]) in.readObject();
        betaSum = (double[]) in.readObject();

        typeTopicCounts = (TIntArrayList[][]) in.readObject();
        tokensPerTopic = (TIntArrayList[]) in.readObject();

        //docLengthCounts = (int[]) in.readObject();
        //topicDocCounts = (int[][]) in.readObject();
        numIterations = in.readInt();
        burninPeriod = in.readInt();
        saveSampleInterval = in.readInt();
        optimizeInterval = in.readInt();
        showTopicsInterval = in.readInt();
        wordsPerTopic = in.readInt();

        saveStateInterval = in.readInt();
        stateFilename = (String) in.readObject();

        saveModelInterval = in.readInt();
        modelFilename = (String) in.readObject();

        randomSeed = in.readInt();
        formatter = (NumberFormat) in.readObject();
        printLogLikelihood = in.readBoolean();

        numThreads = in.readInt();
    }

    public void write(File serializedModelFile) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(serializedModelFile));
            oos.writeObject(this);
            oos.close();
        } catch (IOException e) {
            System.err.println("Problem serializing ParallelTopicModel to file "
                    + serializedModelFile + ": " + e);
        }
    }

    public static iMixParallelTopicModel read(File f) throws Exception {

        iMixParallelTopicModel topicModel = null;

        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f));
        topicModel = (iMixParallelTopicModel) ois.readObject();
        ois.close();

        topicModel.initializeHistograms();

        return topicModel;
    }

    public static void main(String[] args) {

        try {

            InstanceList[] training = new InstanceList[1];
            training[0] = InstanceList.load(new File(args[0]));

            int numTopics = args.length > 1 ? Integer.parseInt(args[1]) : 200;

            iMixParallelTopicModel lda = new iMixParallelTopicModel(numTopics, (byte) 1);
            lda.printLogLikelihood = true;
            lda.setTopicDisplay(50, 7);
            lda.addInstances(training);

            lda.setNumThreads(Integer.parseInt(args[2]));
            lda.estimate();
            logger.info("printing state");
            lda.printState(new File("state.gz"));
            logger.info("finished printing");

        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
