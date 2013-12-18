package cc.mallet.pipe;

import java.io.*;
import java.util.ArrayList;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import java.util.Arrays;

/**
 * Convert a list of strings into a feature sequence
 *
 * @ Target
 */
public class TargetCSV2FeatureSequence extends Pipe {

    public long totalNanos = 0;

    public TargetCSV2FeatureSequence(Alphabet dataDict) {
        super(null, dataDict);
    }

    public TargetCSV2FeatureSequence() {
        super(null, new Alphabet());
    }

    public Instance pipe(Instance carrier) {
        // if (!((String) carrier.getTarget()).isEmpty()) {
        long start = System.nanoTime();

        try {

            if (!((String) carrier.getTarget()).isEmpty()) {
                ArrayList<String> tokens = new ArrayList<String>(Arrays.asList(((String) carrier.getTarget()).split("\\t")));

                //String[] tokens = ((String)carrier.getTarget()).split("[ \\t]");


                FeatureSequence featureSequence =
                        new FeatureSequence((Alphabet) getTargetAlphabet(), tokens.size());
                for (int i = 0; i < tokens.size(); i++) {
                    featureSequence.add(tokens.get(i));
                }

                carrier.setTarget(featureSequence);
            } else {
                FeatureSequence featureSequence =
                        new FeatureSequence((Alphabet) getTargetAlphabet(), 0);
                carrier.setTarget(featureSequence);
            }

            totalNanos += System.nanoTime() - start;
        } catch (ClassCastException cce) {
            System.err.println("Expecting ArrayList<String>, found " + carrier.getData().getClass());
        }
        //}
        return carrier;
    }
    static final long serialVersionUID = 1;
}
