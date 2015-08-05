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

/**
 * A parallel topic model runnable task.
 *
 * @author David Mimno, Andrew McCallum
 */

/*

 int FTreeLDA::sampling(int i)
 {
 std::mt19937 urng(i);
 std::uniform_real_distribution<double> d_unif01(0.0, 1.0);

 double * p = new double[K]; // temp variable for sampling
 int *nd_m = new int[K]; //DocTopic counts: number of words per topics in document 
 int *rev_mapper = new int[K]; // Reverse Map of nonzerotopic 
 for (int k = 0; k < K; ++k)
 {
 nd_m[k] = 0;
 rev_mapper[k] = -1;
 }
 std::chrono::high_resolution_clock::time_point ts, tn;
	
 for (int iter = 1; iter <= n_iters; ++iter)
 {
 ts = std::chrono::high_resolution_clock::now();
 // for each document of worker i
 for (int m = i; m < M; m+=nst)
 {
 int kc = 0;
 for (const auto& k : n_mks[m])
 {
 nd_m[k.first] = k.second; //number of words (K.second) to topic K (K.first)
 rev_mapper[k.first] = kc++; //Reverse Map of topic to active topic
 }
 for (int n = 0; n < trngdata->docs[m]->length; ++n)
 {
 int w = trngdata->docs[m]->words[n];
				
 // remove z_ij from the count variables
 int topic = z[m][n]; int old_topic = topic;
 nd_m[topic] -= 1;
 n_mks[m][rev_mapper[topic]].second -= 1;

 // Multi core approximation: do not update fTree[w] apriori
 // trees[w].update(topic, (nw[w][topic] + beta) / (nwsum[topic] + Vbeta));

 //Compute pdw
 double psum = 0;
 int ii = 0;
 /* Travese all non-zero document-topic distribution */
/*	for (const auto& k : n_mks[m])
 {
 psum += k.second * trees[w].getComponent(k.first);
 p[ii++] = psum; //cumulative array for binary search
 }

 double u = d_unif01(urng) * (psum + alpha*trees[w].w[1]);

 if (u < psum) //binary search in non zero topics
 {
 int temp = std::lower_bound(p,p+ii,u) - p;  // position of related non zero topic
 topic = n_mks[m][temp].first; //actual topic
 }
 else //sample in F tree
 {
 topic = trees[w].sample(d_unif01(urng));
 }

 // add newly estimated z_i to count variables
 if (topic!=old_topic)
 {
 if(nd_m[topic] == 0)
 {
 rev_mapper[topic] = n_mks[m].size();
 n_mks[m].push_back(std::pair<int, int>(topic, 1));
 }
 else
 {
 n_mks[m][rev_mapper[topic]].second += 1;
 }
 nd_m[topic] += 1;
 if (nd_m[old_topic] == 0)
 {
 n_mks[m][rev_mapper[old_topic]].first = n_mks[m].back().first;
 n_mks[m][rev_mapper[old_topic]].second = n_mks[m].back().second;
 rev_mapper[n_mks[m].back().first] = rev_mapper[old_topic];
 n_mks[m].pop_back();
 rev_mapper[old_topic] = -1;
 }
				
 cbuff[nst*(w%ntt)+i].push(delta(w,old_topic,topic));
 }
 else
 {
 n_mks[m][rev_mapper[topic]].second += 1;
 nd_m[topic] += 1;
 }
 z[m][n] = topic;
 }
 for (const auto& k : n_mks[m])
 {
 nd_m[k.first] = 0;
 rev_mapper[k.first] = -1;
 }
 }
 tn = std::chrono::high_resolution_clock::now();
 std::cout << "In thread " << i << " at iteration " << iter << " ..." 
 << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count() << std::endl;
 }
 std::cout<<"Returning from "<<i<<std::endl;
	
 delete[] p;
 delete[] nd_m;
 delete[] rev_mapper;
	
 return 0;	
 }
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

    public FastWorkerRunnable(int numTopics,
            double[] alpha, double alphaSum,
            double beta, Randoms random,
            ArrayList<TopicAssignment> data,
            int[][] typeTopicCounts,
            int[] tokensPerTopic,
            int startDoc, int numDocs) {

        this.data = data;

        this.numTopics = numTopics;
        this.numTypes = typeTopicCounts.length;
        trees = new FTree[this.numTypes];

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
        this.betaSum = beta * numTypes;
        this.random = random;

        this.startDoc = startDoc;
        this.numDocs = numDocs;

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

            double[] temp = new double[numTopics];
//            //smooth for all topics
//            for (int topic = 0; topic < numTopics; topic++) {
//                temp[topic] = beta / (tokensPerTopic[topic] + betaSum);
//            }

            for (int w = 0; w < numTypes; ++w) {

                int[] currentTypeTopicCounts = typeTopicCounts[w];
                for (int currentTopic = 0; currentTopic < numTopics; currentTopic++) {

                    temp[currentTopic] = (currentTypeTopicCounts[currentTopic] + beta) / (tokensPerTopic[currentTopic] + betaSum);
                }

                //trees[w].init(numTopics);
                trees[w] = new FTree(temp);

                //reset temp
                Arrays.fill(temp, 0);
//                while (index < currentTypeTopicCounts.length) {
//                    int currentTopic = currentTypeTopicCounts[index] & topicMask;
//                    temp[currentTopic] = beta / (tokensPerTopic[currentTopic] + betaSum);
//                    index++;
//                }

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

                sampleTopicsForOneDoc(tokenSequence, topicSequence,
                        true);

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

        //	Iterate over the positions (words) in the document 
        for (int position = 0; position < docLength; position++) {
            type = tokenSequence.getIndexAtPosition(position);
            oldTopic = oneDocTopics[position];

            currentTypeTopicCounts = typeTopicCounts[type];

            if (oldTopic != ParallelTopicModel.UNASSIGNED_TOPIC) {

                // Decrement the local doc/topic counts
                localTopicCounts[oldTopic]--;

                // Decrement the global type topic count totals
                currentTypeTopicCounts[oldTopic]--;
                // Decrement the global topic count totals
                tokensPerTopic[oldTopic]--;
                assert (tokensPerTopic[oldTopic] >= 0) : "old Topic " + oldTopic + " below 0";
                // Multi core approximation: do not update fTree[w] apriori
                trees[type].update(oldTopic, ((currentTypeTopicCounts[oldTopic] + beta) / (tokensPerTopic[oldTopic] + betaSum)));

            }

            currentTopic = oldTopic;
            for (int MHstep = 0; MHstep < MHsteps; MHstep++) {

                //Sample Doc topic mass
                double sample = random.nextUniform() * (smoothingOnlyMass + docLength);
                double origSample = sample;

                //	Make sure it actually gets set
                newTopic = -1;

                if (sample < smoothingOnlyMass) {

                    newTopic = lower_bound(smoothingOnlyCumValues, sample);

                } else {
                    sample -= smoothingOnlyMass;
                    newTopic = oneDocTopics[(int) sample];

                }

                if (newTopic == -1) {
                    System.err.println("WorkerRunnable sampling error on doc topic mass: " + origSample + " " + sample + " " + smoothingOnlyMass);
                    newTopic = numTopics - 1; // TODO is this appropriate
                    //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
                }

                if (oldTopic != newTopic) {
                    //2. Find acceptance probability
                    double temp_old = (localTopicCounts[oldTopic] + alpha[oldTopic]) * (currentTypeTopicCounts[oldTopic] + beta) / (tokensPerTopic[oldTopic] + betaSum);
                    double temp_new = (localTopicCounts[newTopic] + alpha[newTopic]) * (currentTypeTopicCounts[newTopic] + beta) / (tokensPerTopic[newTopic] + betaSum);
                    double prop_old = (oldTopic == currentTopic) ? (localTopicCounts[oldTopic] + 1 + alpha[oldTopic]) : (localTopicCounts[oldTopic] + alpha[oldTopic]);
                    double prop_new = (newTopic == currentTopic) ? (localTopicCounts[newTopic] + 1 + alpha[newTopic]) : (localTopicCounts[newTopic] + alpha[newTopic]);
                    double acceptance = (temp_new * prop_old) / (temp_old * prop_new);

                    //3. Compare against uniform[0,1]
                    if (random.nextUniform() < acceptance) {
                        currentTopic = newTopic;
                    }
                }
                //cycling proposals

                //Sample Word topic mass
                sample = random.nextUniform() * (trees[type].w[1]);

                //	Make sure it actually gets set
                newTopic = -1;

                newTopic = trees[type].sample(sample);

                if (newTopic == -1) {
                    System.err.println("WorkerRunnable sampling error on word topic mass: " + sample + " " + trees[type].w[1]);
                    newTopic = numTopics - 1; // TODO is this appropriate
                    //throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
                }

                if (oldTopic != newTopic) {
                    //2. Find acceptance probability
                    double temp_old = (localTopicCounts[oldTopic] + alpha[oldTopic]) * (currentTypeTopicCounts[oldTopic] + beta) / (tokensPerTopic[oldTopic] + betaSum);
                    double temp_new = (localTopicCounts[newTopic] + alpha[newTopic]) * (currentTypeTopicCounts[newTopic] + beta) / (tokensPerTopic[newTopic] + betaSum);
                    double acceptance = (temp_new * trees[type].getComponent(oldTopic)) / (temp_old * trees[type].getComponent(newTopic));

                    //3. Compare against uniform[0,1]
                    if (random.nextUniform() < acceptance) {
                        currentTopic = newTopic;
                    }
                }

            }
            //assert(newTopic != -1);

            //			Put that new topic into the counts
            oneDocTopics[position] = currentTopic;

            localTopicCounts[currentTopic]++;

            currentTypeTopicCounts[currentTopic]++;

            tokensPerTopic[currentTopic]++;
            trees[type].update(currentTopic, ((currentTypeTopicCounts[currentTopic] + beta) / (tokensPerTopic[currentTopic] + betaSum)));

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
