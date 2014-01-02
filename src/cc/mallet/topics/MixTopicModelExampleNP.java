package cc.mallet.topics;

import cc.mallet.examples.*;
import cc.mallet.util.*;

import cc.mallet.types.*;
import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.*;
import cc.mallet.topics.*;
import cc.mallet.util.Maths;

import java.util.*;
import java.util.regex.*;
import java.io.*;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class MixTopicModelExampleNP {

    public enum LabelType {

        Authors,
        Grants
    }

    public MixTopicModelExampleNP() throws IOException {

        int topWords = 10;
        int topLabels = 10;
        byte numModalities = 2;
        double docTopicsThreshold = 0.03;
        int docTopicsMax = -1;
        boolean ignoreLabels = true;
        MixParallelTopicModel.SkewType skewOn = MixParallelTopicModel.SkewType.LabelsOnly;
        //boolean ignoreSkewness = true;
        int numTopics = 50;
        int numIterations = 300;
        LabelType lblType = LabelType.Grants;

        int pruneCnt = 15; //Reduce features to those that occur more than N times
        int pruneLblCnt = 5;
        double pruneMaxPerc = 0.05;//Remove features that occur in more than (X*100)% of documents. 0.05 is equivalent to IDF of 3.0.


        String experimentId = "50T_300I_NIPS_Flat";

        String SQLLitedb = "jdbc:sqlite:C:/projects/OpenAIRE/fundedarxiv.db";

        Connection connection = null;

        // create a database connection


        //Reader fileReader = new InputStreamReader(new FileInputStream(new File(args[0])), "UTF-8");
        //instances.addThruPipe(new CsvIterator (fileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
        //3, 2, 1)); // data, label, name fields
        String inputDir = "C:\\UoA\\OpenAire\\Datasets\\NIPS\\AuthorsNIPS12raw";
        ArrayList<ArrayList<Instance>> instanceBuffer = new ArrayList<ArrayList<Instance>>(numModalities);

        for (Byte m = 0; m < numModalities; m++) {
            instanceBuffer.add(new ArrayList<Instance>());

        }
        try {

            connection = DriverManager.getConnection(SQLLitedb);
            //String sql = "select Doc.DocId, Doc.text, GROUP_CONCAT(GrantPerDoc.GrantId,\"\t\") as GrantIds from Doc inner join GrantPerDoc on Doc.DocId=GrantPerDoc.DocId where Doc.source='pubmed' and grantPerDoc.grantId like 'ec%' Group By Doc.DocId, Doc.text";
            //String docSource = "pubmed";
            String docSource = "arxiv";

            String grantType = "ec%";

            String sql = "";

            if (lblType == LabelType.Grants) {
                sql =
                        " select Doc.DocId, Doc.text, GROUP_CONCAT(GrantPerDoc.GrantId,'\t') as GrantIds  "
                        + " from Doc inner join "
                        + " GrantPerDoc on Doc.DocId=GrantPerDoc.DocId "
                      //  + " where  "
                      //  + " Doc.source='" + docSource + "' and "
                      //  + " grantPerDoc.grantId like '" + grantType + "' "
                        + " Group By Doc.DocId, Doc.text";
            } else if (lblType == LabelType.Authors) {
                sql =
                        " select Doc.DocId,Doc.text, GROUP_CONCAT(AuthorPerDoc.authorID,'\t') as AuthorIds \n"
                        + "from Doc \n"
                        + "inner join AuthorPerDoc on Doc.DocId=AuthorPerDoc.DocId \n"
                        + "Where \n"
                        + "Doc.source='NIPS' \n"
                        + "Group By Doc.DocId, Doc.text";
            }


            // String sql = "select fundedarxiv.file from fundedarxiv inner join funds on file=filename Group By fundedarxiv.file LIMIT 10" ;
            Statement statement = connection.createStatement();
            statement.setQueryTimeout(30);  // set timeout to 30 sec.
            ResultSet rs = statement.executeQuery(sql);
            while (rs.next()) {
                // read the result set
                //String lblStr = "[" + rs.getString("GrantIds") + "]" ;//+ rs.getString("text");
                //String str = "[" + rs.getString("GrantIds") + "]" + rs.getString("text");
                //System.out.println("name = " + rs.getString("file"));
                //System.out.println("name = " + rs.getString("fundings"));
                //int cnt = rs.getInt("grantsCnt");
                switch (lblType) {
                    case Grants:
                        instanceBuffer.get(0).add(new Instance(rs.getString("text"), null, rs.getString("DocId"), "text"));
                        instanceBuffer.get(1).add(new Instance(rs.getString("GrantIds"), null, rs.getString("DocId"), "grant"));
                        break;
                    case Authors:
                        instanceBuffer.get(0).add(new Instance(rs.getString("text"), null, rs.getString("DocId"), "text"));
                        instanceBuffer.get(1).add(new Instance(rs.getString("AuthorIds"), null, rs.getString("DocId"), "author"));
                        break;
                    default:
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

        numModalities = 2;

        // Begin by importing documents from text to feature sequences
        ArrayList<Pipe> pipeListText = new ArrayList<Pipe>();

        // Pipes: lowercase, tokenize, remove stopwords, map to features
        pipeListText.add(new Input2CharSequence(false)); //homer
        pipeListText.add(new CharSequenceLowercase());
        //SimpleTokenizer tokenizer = new SimpleTokenizer(new File("stoplists/en.txt"));
        //GenerateStoplist(tokenizer, instanceBuffer.get(0), pruneCnt, pruneMaxPerc, false);
        // pipeListText.add(tokenizer);
        pipeListText.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));// Original --> replaced by simpletokenizer in order to use prunedStopiList
        pipeListText.add(new TokenSequenceRemoveStopwords(new File("stoplists/en.txt"), "UTF-8", false, false, false)); //Original --> replaced by simpletokenizer

        //Alphabet alphabet = new Alphabet();
        //pipeListText.add(new StringList2FeatureSequence(alphabet));

        pipeListText.add(new TokenSequence2FeatureSequence()); //Original



        // Other Modalities
        ArrayList<Pipe> pipeListCSV = new ArrayList<Pipe>();
        pipeListCSV.add(new CSV2FeatureSequence());

        InstanceList[] instances = new InstanceList[numModalities];


        instances[0] = new InstanceList(new SerialPipes(pipeListText));
        instances[0].addThruPipe(instanceBuffer.get(0).iterator());

        for (Byte m = 1; m < numModalities; m++) {
            instances[1] = new InstanceList(new SerialPipes(pipeListCSV));
            instances[1].addThruPipe(instanceBuffer.get(1).iterator());
        }


// pruning for all other modalities no text
        for (Byte m = 0; m < numModalities; m++) {
            if ((m == 0 && pruneCnt > 0) || (m > 0 && pruneLblCnt > 0)) {

                // Check which type of data element the instances contain
                Instance firstInstance = instances[m].get(0);
                if (firstInstance.getData() instanceof FeatureSequence) {
                    // Version for feature sequences

                    Alphabet oldAlphabet = instances[m].getDataAlphabet();
                    Alphabet newAlphabet = new Alphabet();

                    // It's necessary to create a new instance list in
                    //  order to make sure that the data alphabet is correct.
                    Noop newPipe = new Noop(newAlphabet, instances[m].getTargetAlphabet());
                    InstanceList newInstanceList = new InstanceList(newPipe);

                    // Iterate over the instances in the old list, adding
                    //  up occurrences of features.
                    int numFeatures = oldAlphabet.size();
                    double[] counts = new double[numFeatures];
                    for (int ii = 0; ii < instances[m].size(); ii++) {
                        Instance instance = instances[m].get(ii);
                        FeatureSequence fs = (FeatureSequence) instance.getData();

                        fs.addFeatureWeightsTo(counts);
                    }

                    Instance instance;

                    // Next, iterate over the same list again, adding 
                    //  each instance to the new list after pruning.
                    while (instances[m].size() > 0) {
                        instance = instances[m].get(0);
                        FeatureSequence fs = (FeatureSequence) instance.getData();

                        fs.prune(counts, newAlphabet, m == 0 ? pruneCnt : pruneLblCnt);


                        newInstanceList.add(newPipe.instanceFrom(new Instance(fs, instance.getTarget(),
                                instance.getName(),
                                instance.getSource())));
                        instances[m].remove(0);
                    }

//                logger.info("features: " + oldAlphabet.size()
                    //                       + " -> " + newAlphabet.size());

                    // Make the new list the official list.
                    instances[m] = newInstanceList;


                } else {
                    throw new UnsupportedOperationException("Pruning features from "
                            + firstInstance.getClass().getName()
                            + " is not currently supported");
                }

            }
        }



        //instances.addThruPipe(new FileIterator(inputDir));

        //instances.addThruPipe (new FileIterator ("C:\\UoA\\OpenAire\\Datasets\\YIpapersTXT\\YIpapersTXT"));
        //
        //instances.addThruPipe(new CsvIterator (fileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),

        // Create a model with 100 topics, alpha_t = 0.01, beta_w = 0.01
        //  Note that the first parameter is passed as the sum over topics, while
        //  the second is 
        InstanceList[] testInstances = new InstanceList[numModalities];
        for (Byte m = 0; m < numModalities; m++) {
            InstanceList[] ilists = instances[m].split(new double[]{.9, .1});
            instances[m] = ilists[0];
            testInstances[m] = ilists[1];
        }
        String outputDir = "C:\\projects\\OpenAIRE\\OUT\\" + experimentId;

        File outPath = new File(outputDir);
        outPath.mkdir();

        String stateFile = outputDir + File.separator + "output_state";
        String outputDocTopicsFile = outputDir + File.separator + "output_doc_topics.csv";
        String outputTopicPhraseXMLReport = outputDir + File.separator + "topicPhraseXMLReport.xml";

        String topicKeysFile = outputDir + File.separator + "output_topic_keys.csv";
        String topicWordWeightsFile = outputDir + File.separator + "topicWordWeightsFile.csv";

        String stateFileZip = outputDir + File.separator + "output_state.gz";
        String modelEvaluationFile = outputDir + File.separator + "model_evaluation.txt";
        String modelDiagnosticsFile = outputDir + File.separator + "model_diagnostics.xml";

        boolean runNPModel = false;
        if (runNPModel) {
            NPTopicModel npModel = new NPTopicModel(5.0, 10.0, 0.1);
            npModel.addInstances(instances[0], 50);
            npModel.setTopicDisplay(20, 10);
            npModel.sample(100);
            FileWriter fwrite = new FileWriter(outputDir + File.separator + "output_NP_topics.csv");
            BufferedWriter NP_Topics_out = new BufferedWriter(fwrite);
            NP_Topics_out.write(npModel.topWords(10) + "\n");
            NP_Topics_out.flush();
            npModel.printState(new File(outputDir + File.separator + "NP_output_state.gz"));
        }

        boolean runHDPModel = false;
        if (runHDPModel) {
            //setup HDP parameters(alpha, beta, gamma, initialTopics)
            HDP hdp = new HDP(1.0, 0.1, 1.0, 10);
            hdp.initialize(instances[0]);

            //set number of iterations, and display result or not 
            hdp.estimate(2000);

            //get topic distribution for first instance
            double[] distr = hdp.topicDistribution(0);
            //print out
            for (int j = 0; j < distr.length; j++) {
                System.out.print(distr[j] + " ");
            }

            //for inferencer
            HDPInferencer inferencer = hdp.getInferencer();
            inferencer.setInstance(testInstances[0]);
            inferencer.estimate(100);
            //get topic distribution for first test instance
            distr = inferencer.topicDistribution(0);
            //print out
            for (int j = 0; j < distr.length; j++) {
                System.out.print(distr[j] + " ");
            }
            //get preplexity
            double prep = inferencer.getPreplexity();
            System.out.println("preplexity for the test set=" + prep);

            //10-folds cross validation, with 1000 iteration for each test.
            hdp.runCrossValidation(10, 1000);

        }
        double[] beta = new double[numModalities];
        Arrays.fill(beta, 0.01);

        boolean runOrigParallelModel = false;
        if (runOrigParallelModel) {
            ParallelTopicModel modelOrig = new ParallelTopicModel(numTopics, 1.0, beta[0]);

            modelOrig.addInstances(instances[0]);

            // Use two parallel samplers, which each look at one half the corpus and combine
            //  statistics after every iteration.
            modelOrig.setNumThreads(2);
            // Run the model for 50 iterations and stop (this is for testing only, 
            //  for real applications, use 1000 to 2000 iterations)
            modelOrig.setNumIterations(numIterations);
            modelOrig.optimizeInterval = 50;
            modelOrig.burninPeriod = 150;
            //model.optimizeInterval = 0;
            //model.burninPeriod = 0;
            //model.saveModelInterval=250;
            modelOrig.estimate();
        }
        MixParallelTopicModel model = new MixParallelTopicModel(numTopics, numModalities, 1.0, beta, ignoreLabels, skewOn);

        // ParallelTopicModel model = new ParallelTopicModel(numTopics, 1.0, 0.01);

        model.addInstances(instances);

        // Use two parallel samplers, which each look at one half the corpus and combine
        //  statistics after every iteration.
        model.setNumThreads(4);
        // Run the model for 50 iterations and stop (this is for testing only, 
        //  for real applications, use 1000 to 2000 iterations)
        model.setNumIterations(numIterations);
        model.optimizeInterval = 50;
        model.burninPeriod = 100;
        //model.optimizeInterval = 0;
        //model.burninPeriod = 0;
        //model.saveModelInterval=250;
        model.estimate();
        model.saveTopics(SQLLitedb, experimentId);
        model.printTopWords(new File(topicKeysFile), topWords, topLabels, false);
        //model.printTopWords(new File(topicKeysFile), topWords,  false);
        //model.printTopicWordWeights(new File(topicWordWeightsFile));
        //model.printTopicLabelWeights(new File(topicLabelWeightsFile));
        model.printState(new File(stateFileZip));
        PrintWriter outState = new PrintWriter(new FileWriter((new File(outputDocTopicsFile))));
        model.printDocumentTopics(outState, docTopicsThreshold, docTopicsMax, SQLLitedb, experimentId, 0.1);

        outState.close();

        PrintWriter outXMLPhrase = new PrintWriter(new FileWriter((new File(outputTopicPhraseXMLReport))));
        model.topicPhraseXMLReport(outXMLPhrase, topWords);
        outState.close();

        GunZipper g = new GunZipper(new File(stateFileZip));
        g.unzip(new File(stateFile));

        try {
            outputCsvFiles(outputDir, true, inputDir, numTopics, stateFile, outputDocTopicsFile, topicKeysFile);
        } catch (Exception e) {
            // if the error message is "out of memory", 
            // it probably means no database file is found
            System.err.println(e.getMessage());
        }


        //calc similarities

        try {
            // create a database connection
            //connection = DriverManager.getConnection(SQLLitedb);
            connection = DriverManager.getConnection(SQLLitedb);
            Statement statement = connection.createStatement();
            statement.setQueryTimeout(30);  // set timeout to 30 sec.

            // statement.executeUpdate("drop table if exists person");
//      statement.executeUpdate("create table person (id integer, name string)");
//      statement.executeUpdate("insert into person values(1, 'leo')");
//      statement.executeUpdate("insert into person values(2, 'yui')");
//      ResultSet rs = statement.executeQuery("select * from person");
            String sql = "";
            switch (lblType) {
                case Grants:
                    sql = "select    GrantId, TopicId, AVG(weight) as Weight from topicsPerDoc Inner Join GrantPerDoc on topicsPerDoc.DocId= GrantPerDoc.DocId"
                            + " where weight>0.02 AND ExperimentId='" + experimentId + "' group By GrantId , TopicId order by  GrantId   , TopicId";
                    break;
                case Authors:
                    sql = "select    AuthorId, TopicId, AVG(weight) as Weight from topicsPerDoc Inner Join AuthorPerDoc on topicsPerDoc.DocId= AuthorPerDoc.DocId"
                            + " where weight>0.02 AND ExperimentId='" + experimentId + "' group By AuthorId , TopicId order by  AuthorId   , TopicId";
                    break;
                default:
            }


            // String sql = "select fundedarxiv.file from fundedarxiv inner join funds on file=filename Group By fundedarxiv.file LIMIT 10" ;

            ResultSet rs = statement.executeQuery(sql);

            HashMap<String, SparseVector> labelVectors = new HashMap<String, SparseVector>();

            String labelId = "";
            int[] topics = new int[numTopics];
            double[] weights = new double[numTopics];
            int cnt = 0;
            double a;
            while (rs.next()) {

                String newLabelId = "";

                switch (lblType) {
                    case Grants:
                        newLabelId = rs.getString("GrantId");
                        break;
                    case Authors:
                        newLabelId = rs.getString("AuthorId");
                        break;
                    default:
                }

                if (!newLabelId.equals(labelId) && !labelId.isEmpty()) {
                    labelVectors.put(labelId, new SparseVector(topics, weights, topics.length, topics.length, true, true, true));
                    topics = new int[numTopics];
                    weights = new double[numTopics];
                    cnt = 0;
                }
                labelId = newLabelId;
                topics[cnt] = rs.getInt("TopicId");
                weights[cnt] = rs.getDouble("Weight");
                cnt++;


            }

            cnt = 0;
            double similarity = 0;
            double similarityThreshold = 0.1;
            NormalizedDotProductMetric cosineSimilarity = new NormalizedDotProductMetric();



            statement.executeUpdate("create table if not exists EntitySimilarity (EntityType int, EntityId1 nvarchar(50), EntityId2 nvarchar(50), Similarity double, ExperimentId nvarchar(50)) ");
            String deleteSQL = String.format("Delete from EntitySimilarity where  ExperimentId = '%s'", experimentId);
            statement.executeUpdate(deleteSQL);

            PreparedStatement bulkInsert = null;
            sql = "insert into EntitySimilarity values(?,?,?,?,?);";

            try {

                connection.setAutoCommit(false);
                bulkInsert = connection.prepareStatement(sql);


                for (String fromGrantId : labelVectors.keySet()) {
                    boolean startCalc = false;

                    for (String toGrantId : labelVectors.keySet()) {
                        if (!fromGrantId.equals(toGrantId) && !startCalc) {
                            continue;
                        } else {
                            startCalc = true;
                            similarity = 1 - cosineSimilarity.distance(labelVectors.get(fromGrantId), labelVectors.get(toGrantId)); // the function returns distance not similarity
                            if (similarity > similarityThreshold && !fromGrantId.equals(toGrantId)) {
                                bulkInsert.setInt(1, lblType.hashCode());
                                bulkInsert.setString(2, fromGrantId);
                                bulkInsert.setString(3, toGrantId);
                                bulkInsert.setDouble(4, (double) Math.round(similarity * 1000) / 1000);
                                bulkInsert.setString(5, experimentId);
                                bulkInsert.executeUpdate();
                            }
                        }
                    }
                }
                connection.commit();

            } catch (SQLException e) {

                if (connection != null) {
                    try {
                        System.err.print("Transaction is being rolled back");
                        connection.rollback();
                    } catch (SQLException excep) {
                        System.err.print("Error in insert grantSimilarity");
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





        if (modelEvaluationFile != null) {
            try {

//                ObjectOutputStream oos =
//                        new ObjectOutputStream(new FileOutputStream(modelEvaluationFile));
//                oos.writeObject(model.getProbEstimator());
//                oos.close();
//                
                PrintStream docProbabilityStream = null;
                if (modelEvaluationFile != null) {
                    docProbabilityStream = new PrintStream(modelEvaluationFile);
                }


                double perplexity = model.getProbEstimator().evaluateLeftToRight(testInstances[0], 10, false, docProbabilityStream);


                System.out.println("preplexity for the test set=" + perplexity);



            } catch (Exception e) {
                System.err.println(e.getMessage());
            }

        }

        if (modelDiagnosticsFile != null) {
            PrintWriter out = new PrintWriter(modelDiagnosticsFile);
            MixTopicModelDiagnostics diagnostics = new MixTopicModelDiagnostics(model, topWords);
            out.println(diagnostics.toXML()); //preferable than XML???
            out.close();
        }



        //If any value in <tt>p2</tt> is <tt>0.0</tt> then the KL-divergence
        //double a = Maths.klDivergence();

        //model.printTypeTopicCounts(new File (wordTopicCountsFile.value));



        // Show the words and topics in the first instance

        // The data alphabet maps word IDs to strings
     /*   Alphabet dataAlphabet = instances.getDataAlphabet();

         FeatureSequence tokens = (FeatureSequence) model.getData().get(0).instance.getData();
         LabelSequence topics = model.getData().get(0).topicSequence;

         Formatter out = new Formatter(new StringBuilder(), Locale.US);
         for (int posit= 0; position < tokens.getLength(); position++) {
         out.format("%s-%d ", dataAlphabet.lookupObject(tokens.getIndexAtPosition(position)), topics.getIndexAtPosition(position));
         }
         System.out.println(out);

         // Estimate the topic distribution of the first instance, 
         //  given the current Gibbs state.
         double[] topicDistribution = model.getTopicProbabilities(0);

         // Get an array of sorted sets of word ID/count pairs
         ArrayList<TreeSet<IDSorter>> topicSortedWords = model.getSortedWords();

         // Show top 5 words in topics with proportions for the first document
         for (int topic = 0; topic < numTopics; topic++) {
         Iterator<IDSorter> iterator = topicSortedWords.get(topic).iterator();

         out = new Formatter(new StringBuilder(), Locale.US);
         out.format("%d\t%.3f\t", topic, topicDistribution[topic]);
         int rank = 0;
         while (iterator.hasNext() && rank < 5) {
         IDSorter idCountPair = iterator.next();
         out.format("%s (%.0f) ", dataAlphabet.lookupObject(idCountPair.getID()), idCountPair.getWeight());
         rank++;
         }
         System.out.println(out);
         }

         // Create a new instance with high probability of topic 0
         StringBuilder topicZeroText = new StringBuilder();
         Iterator<IDSorter> iterator = topicSortedWords.get(0).iterator();

         int rank = 0;
         while (iterator.hasNext() && rank < 5) {
         IDSorter idCountPair = iterator.next();
         topicZeroText.append(dataAlphabet.lookupObject(idCountPair.getID()) + " ");
         rank++;
         }

         // Create a new instance named "test instance" with empty target and source fields.
         InstanceList testing = new InstanceList(instances.getPipe());
         testing.addThruPipe(new Instance(topicZeroText.toString(), null, "test instance", null));

         TopicInferencer inferencer = model.getInferencer();
         double[] testProbabilities = inferencer.getSampledDistribution(testing.get(0), 10, 1, 5);
         System.out.println("0\t" + testProbabilities[0]);
         */

    }

    private void GenerateStoplist(SimpleTokenizer prunedTokenizer, ArrayList<Instance> instanceBuffer, int pruneCount, double docProportionCutoff, boolean preserveCase)
            throws IOException {

        ArrayList<Instance> input = new ArrayList<Instance>();
        for (Instance instance : instanceBuffer) {
            input.add((Instance) instance.clone());
        }

        ArrayList<Pipe> pipes = new ArrayList<Pipe>();
        Alphabet alphabet = new Alphabet();

        CharSequenceLowercase csl = new CharSequenceLowercase();
        SimpleTokenizer st = prunedTokenizer.deepClone();
        StringList2FeatureSequence sl2fs = new StringList2FeatureSequence(alphabet);
        FeatureCountPipe featureCounter = new FeatureCountPipe(alphabet, null);
        FeatureDocFreqPipe docCounter = new FeatureDocFreqPipe(alphabet, null);

        pipes.add(new Input2CharSequence(false)); //homer

        if (!preserveCase) {
            pipes.add(csl);
        }
        pipes.add(st);
        pipes.add(sl2fs);
        if (pruneCount > 0) {
            pipes.add(featureCounter);
        }
        if (docProportionCutoff < 1.0) {
            pipes.add(docCounter);
        }

        Pipe serialPipe = new SerialPipes(pipes);

        Iterator<Instance> iterator = serialPipe.newIteratorFrom(input.iterator());

        int count = 0;

        // We aren't really interested in the instance itself,
        //  just the total feature counts.
        while (iterator.hasNext()) {
            count++;
            if (count % 100000 == 0) {
                System.out.println(count);
            }
            iterator.next();
        }

        if (pruneCount > 0) {
            featureCounter.addPrunedWordsToStoplist(prunedTokenizer, pruneCount);
        }
        if (docProportionCutoff < 1.0) {
            docCounter.addPrunedWordsToStoplist(prunedTokenizer, docProportionCutoff);
        }
    }

    private void outputCsvFiles(String outputDir, Boolean htmlOutputFlag, String inputDir, int numTopics, String stateFile, String outputDocTopicsFile, String topicKeysFile) {

        CsvBuilder cb = new CsvBuilder();
        cb.createCsvFiles(numTopics, outputDir, stateFile, outputDocTopicsFile, topicKeysFile);

        if (htmlOutputFlag) {
            HtmlBuilder hb = new HtmlBuilder(cb.getNtd(), new File(inputDir));
            hb.createHtmlFiles(new File(outputDir));
        }
        //clearExtrafiles(outputDir);
    }

    private void clearExtrafiles(String outputDir) {
        String[] fileNames = {"topic-input.mallet", "output_topic_keys.csv", "output_state.gz",
            "output_doc_topics.csv", "output_state"};
        for (String f : fileNames) {
            if (!(new File(outputDir, f).canWrite())) {
                System.out.println(f);
            }
            Boolean b = new File(outputDir, f).delete();

        }


    }

    public static void main(String[] args) throws Exception {
        Class.forName("org.sqlite.JDBC");
        MixTopicModelExampleNP trainer = new MixTopicModelExampleNP();

    }
}