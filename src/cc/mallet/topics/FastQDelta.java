/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cc.mallet.topics;

/**
 *
 * @author hmetaxa
 */
public class FastQDelta {

    public int NewTopic;
    public int OldTopic;
    public int Type;
    public int Modality;

    public FastQDelta() {

    }

    public FastQDelta(int newT, int OldT, int type, int mod) {
        NewTopic = newT;
        OldTopic = OldT;
        Type = type;
        Modality = mod;

    }

}
