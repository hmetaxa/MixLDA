package cc.mallet.topics;

import java.io.Serializable;
import cc.mallet.types.*;
import java.util.ArrayList;

/** This class combines a sequence of observed features
 *   with a sequence of hidden "labels".
 */

public class TopicAssignment implements Serializable {
	public Instance instance;
	public LabelSequence topicSequence;
        public long[] prevTopicsSequence; // array containing points to the entry of the topicSecuenceList which is a LongInteger array. 
        // Max num of topics depends on topicBits (MAX 10-->1024 topics). 
        // We could only use this array and have the first topic bits represent the current topic. Then, on new topic shift all topics right and place new topic
        // In each iteration we should keep a sorted ArrayList<Long> (or <long, int> hashmap) containing new topic assignment for each previous sequence. Then during
        // this iteration we either use this assignment or keep previous one (thus rejecting new) based on a metropolis hasting step
        
        public LabelSequence lblTopicSequence;
        //public ArrayList<LabelSequence> lblsTopicSequence;
	public Labeling topicDistribution;
        
                
	public TopicAssignment (Instance instance, LabelSequence topicSequence) {
		this.instance = instance;
		this.topicSequence = topicSequence;
	}
        
        public TopicAssignment (Instance instance, LabelSequence topicSequence,long[] prevTopicsSequence) {
		this.instance = instance;
		this.topicSequence = topicSequence;
                this.prevTopicsSequence = prevTopicsSequence;
	}
        
        public TopicAssignment (Instance instance, LabelSequence topicSequence,LabelSequence lblTopicSequence) {
		this.instance = instance;
		this.topicSequence = topicSequence;
                this.lblTopicSequence = lblTopicSequence;
	}
}
