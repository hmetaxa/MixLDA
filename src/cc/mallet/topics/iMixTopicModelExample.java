package cc.mallet.topics;

import cc.mallet.examples.*;
import cc.mallet.util.*;

import cc.mallet.types.*;
import cc.mallet.pipe.*;
//import cc.mallet.pipe.iterator.*;
//import cc.mallet.topics.*;
//import cc.mallet.util.Maths;
//import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TObjectIntMap;
//import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.util.*;
//import java.util.regex.*;
import java.io.*;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.logging.Logger;

public class iMixTopicModelExample {

    public enum ExperimentType {

        Authors,
        Grants,
        DBLP,
        PM_pdb,
        DBLP_ACM,
        ACM,
        FullGrants,
        FETGrants,
        HEALTHTender,
        HEALTHTenderGrantGroup
    }

    public enum SimilarityType {

        cos,
        Jen_Sha_Div,
        symKL
    }

    public iMixTopicModelExample() throws IOException {

        Logger logger = MalletLogger.getLogger(iMixTopicModelExample.class.getName());
        int topWords = 10;
        int topLabels = 10;
        byte numModalities = 1;
        //int numIndependentTopics = 0;
        double docTopicsThreshold = 0.03;
        int docTopicsMax = -1;
        //boolean ignoreLabels = true;
        boolean calcSimilarities = false;
        boolean runTopicModelling = true;
        boolean calcTokensPerEntity = false;
        //iMixParallelTopicModel.SkewType skewOn = iMixParallelTopicModel.SkewType.None;
        //boolean ignoreSkewness = true;
        int numTopics = 200;
        int maxNumTopics = 200;
        int numIterations = 500; //Max 2000
        int independentIterations = 0;
        int burnIn = 50;
        int optimizeInterval = 20;
        ExperimentType experimentType = ExperimentType.Authors;
        int pruneCnt = 20; //Reduce features to those that occur more than N times
        int pruneLblCnt = 7;
        double pruneMaxPerc = 0.5;//Remove features that occur in more than (X*100)% of documents. 0.05 is equivalent to IDF of 3.0.
        SimilarityType similarityType = SimilarityType.cos; //Cosine 1 jensenShannonDivergence 2 symmetric KLP
        boolean ACMAuthorSimilarity = false;
//boolean runParametric = true;

        boolean DBLP_PPR = false;

        try {

            double[] temp = {0.3, 1.5, 0.4, 0.3};
            FTree tree = new FTree(temp);

            int tmp = tree.sample(2.1/2.5);
            logger.info("FTree sample 2.1 (2):" + tmp);

            double tmp2 = tree.getComponent(0);
            logger.info("FTree getComponent(0):" + tmp2);

            tmp2 = tree.getComponent(3);
            logger.info("FTree getComponent(3):" + tmp2);

            tree.update(2, 1.4);

            tmp = tree.sample(3.19/3.5);
            logger.info("FTree sample 3.19 (2):" + tmp);

        } catch (Exception e) {
            e.printStackTrace();
        }

        //String addedExpId = (experimentType == ExperimentType.ACM ? (ACMAuthorSimilarity ? "Author" : "Category") : "");
        String experimentId = experimentType.toString() + "_" + numTopics + "T_" + (maxNumTopics > numTopics + 1 ? maxNumTopics + "maxT_" : "")
                + numIterations + "IT_" + independentIterations + "IIT_" + burnIn + "B_" + numModalities + "M_" + similarityType.toString(); // + "_" + skewOn.toString();
        String experimentDescription = "";

        String SQLLitedb = "jdbc:sqlite:C:/projects/OpenAIRE/fundedarxiv.db";

        if (experimentType == ExperimentType.Grants) {
            SQLLitedb = "jdbc:sqlite:C:/projects/Datasets/OpenAIRE/fundedarxiv.db";
        } else if (experimentType == ExperimentType.DBLP) {
            SQLLitedb = "jdbc:sqlite:C:/projects/Datasets/DBLPManage/citation-network2.db";
        } else if (experimentType == ExperimentType.PM_pdb) {
            SQLLitedb = "jdbc:sqlite:C:/projects/Datasets/OpenAIRE/europepmc.db";
        } else if (experimentType == ExperimentType.DBLP_ACM) {
            SQLLitedb = "jdbc:sqlite:C:/projects/Datasets/DBLPManage/acm_output.db";
        } else if (experimentType == ExperimentType.ACM) {
            SQLLitedb = "jdbc:sqlite:C:/projects/Datasets/ACM/acmdata1.db";
        } else if (experimentType == ExperimentType.FullGrants) {
            SQLLitedb = "jdbc:sqlite:C:/projects/Datasets/OpenAIRE/NewDB/fp7openaire.db";
        } else if (experimentType == ExperimentType.FETGrants) {
            SQLLitedb = "jdbc:sqlite:C:/projects/Datasets/OpenAIRE/NewDB/fp7openaire.db";
        } else if (experimentType == ExperimentType.HEALTHTender) {
            SQLLitedb = "jdbc:sqlite:C:/projects/Datasets/OpenAIRE/NewDB/fp7openaire.db";
        } else if (experimentType == ExperimentType.HEALTHTenderGrantGroup) {
            SQLLitedb = "jdbc:sqlite:C:/projects/Datasets/OpenAIRE/NewDB/fp7openaire.db";
        }
        Connection connection = null;

        //createRefACMTables(SQLLitedb);
        // create a database connection
        //Reader fileReader = new InputStreamReader(new FileInputStream(new File(args[0])), "UTF-8");
        //instances.addThruPipe(new CsvIterator (fileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
        //3, 2, 1)); // data, label, name fields
        String inputDir = "C:\\UoA\\OpenAire\\Datasets\\NIPS\\AuthorsNIPS12raw";
        ArrayList<ArrayList<Instance>> instanceBuffer = new ArrayList<ArrayList<Instance>>(numModalities);
        if (runTopicModelling || calcTokensPerEntity) {
            //createCitationGraphFile("C:\\projects\\Datasets\\DBLPManage\\acm_output_NET.csv", "jdbc:sqlite:C:/projects/Datasets/DBLPManage/acm_output.db");
            for (byte m = 0; m < numModalities; m++) {
                instanceBuffer.add(new ArrayList<Instance>());

            }
            try {

                connection = DriverManager.getConnection(SQLLitedb);
                //String sql = "select Doc.DocId, Doc.text, GROUP_CONCAT(GrantPerDoc.GrantId,\"\t\") as GrantIds from Doc inner join GrantPerDoc on Doc.DocId=GrantPerDoc.DocId where Doc.source='pubmed' and grantPerDoc.grantId like 'ec%' Group By Doc.DocId, Doc.text";
                //String docSource = "pubmed";
                String docSource = "arxiv";

                String grantType = "FP7";

                String sql = "";

                if (experimentType == ExperimentType.Grants) {
                    experimentDescription = "Topic modeling based on:\n1)Full text publications related to " + grantType + "\n2)Research Areas\n3)Venues (e.g., PubMed, Arxiv, ACM)\n4)Grants per Publication Links\n SimilarityType:" + similarityType.toString();

                    sql = "select Doc.DocId, Doc.text, GROUP_CONCAT(GrantPerDoc.GrantId,'\t') as GrantIds,GROUP_CONCAT(Grant.Category2,'\t') as Areas, Doc.Source  \n"
                            + "                         from Doc inner join \n"
                            + "                         GrantPerDoc on Doc.DocId=GrantPerDoc.DocId\n"
                            + "			 INNER JOIN Grant on Grant.GrantId= GrantPerDoc.GrantId\n"
                            + "                         where \n"
                            + "                        Grant.Category0='" + grantType + "'\n"
                            + "                         Group By Doc.DocId, Doc.text"
                            + " LIMIT 10000";

//                        " select Doc.DocId, Doc.text, GROUP_CONCAT(GrantPerDoc.GrantId,'\t') as GrantIds  "
//                        + " from Doc inner join "
//                        + " GrantPerDoc on Doc.DocId=GrantPerDoc.DocId "
//                        //  + " where  "
//                        //  + " Doc.source='" + docSource + "' and "
//                        //  + " grantPerDoc.grantId like '" + grantType + "' "
//                        + " Group By Doc.DocId, Doc.text";
                } else if (experimentType == ExperimentType.FullGrants) {

                    grantType = "FP7";

                    experimentDescription += "Topic modeling analyzing:\n1)Full text of publications and project descriptions related to " + grantType + "\n2)Research Areas\n3)Venues (e.g., PubMed, Arxiv, ACM)\n4)Grants per Publication Links\n SimilarityType:" + similarityType.toString();

                    sql = "select pubs.originalid AS DocId, \n"
                            + "                            CASE WHEN IFNULL(pubs.fulltext,'')='' THEN pubs.abstract ELSE pubs.fulltext END AS TEXT,\n"
                            //+ "                            CASE WHEN IFNULL(pubs.abstract,'')='' THEN pubs.title||' '||pubs.fulltext ELSE pubs.title||' '||pubs.abstract END AS TEXT,\n"
                            + "                            GROUP_CONCAT(links.projectId,'\t') as GrantIds,"
                            + "                     GROUP_CONCAT(Category2,'\t') as Areas, \n"
                            //+ "                        CASE WHEN IFNULL(journal,'')='' THEN  ''  else Journal END as Venue, \n"
                            + "                        CASE WHEN IFNULL(journal,'')='' THEN  Repository  else Journal END as Venue, \n"
                            + "                    'PubAbstract'   AS TEXTType \n"
                            + "                            from pubs \n"
                            + "                            inner join links on links.OriginalId = pubs.originalid \n"
                            + "			    inner join projectView on links.projectId=projectView.ProjectId  \n"
                            + "                            Group By pubs.originalid \n"
                            + "                            \n"
                            + "                            UNION \n"
                            + "                            \n"
                            + "                            select 'FP7_'||projectView.ProjectId AS DocId, projectView.ABSTRACT AS TEXT, projectView.ProjectId AS GrantIds ,"
                            + " projectView.Category2 AS Areas, \n"
                            + " '' AS Venue, \n"
                            + " 'ProjectAbstract' AS TEXTType \n"
                            + "                            from projectView\n"
                            + "                            where IFNULL(abstract,'')<>''  and (projectView.projectId in (SELECT links.projectId from Links ))\n"
                            + " LIMIT 20000";
//                        " select Doc.DocId, Doc.text, GROUP_CONCAT(GrantPerDoc.GrantId,'\t') as GrantIds  "
//                        + " from Doc inner join "
//                        + " GrantPerDoc on Doc.DocId=GrantPerDoc.DocId "
//                        //  + " where  "
//                        //  + " Doc.source='" + docSource + "' and "
//                        //  + " grantPerDoc.grantId like '" + grantType + "' "
//                        + " Group By Doc.DocId, Doc.text";
                } else if (experimentType == ExperimentType.FETGrants) {

                    sql = "select pubs.originalid AS DocId, \n"
                            + "                            CASE WHEN IFNULL(pubs.fulltext,'')='' THEN pubs.abstract ELSE pubs.fulltext END AS TEXT,\n"
                            + "                            GROUP_CONCAT(links.projectId,'\t') as GrantIds,"
                            + "                       GROUP_CONCAT(Category3,'\t') as Areas, \n"
                            + "                        CASE WHEN IFNULL(journal,'')='' THEN   Repository else Journal END as Venue, \n"
                            + "                            CASE WHEN IFNULL(pubs.fulltext,'')='' THEN 'PubAbstract' ELSE 'PubFullText' END AS TEXTType \n"
                            + "                            from pubs \n"
                            + "                            inner join links on links.OriginalId = pubs.originalid \n"
                            + "			    inner join projectView on links.projectId=projectView.ProjectId  \n"
                            + "                   and projectView.Category1<>'NONFET'\n"
                            + "                            Group By pubs.originalid \n"
                            + "                            \n"
                            + "                            UNION \n"
                            + "                            \n"
                            + "                            select 'FP7_'||projectView.ProjectId AS DocId, projectView.ABSTRACT AS TEXT, projectView.ProjectId AS GrantIds ,"
                            + " projectView.Category3 AS Areas, \n"
                            + " '' AS Venue, \n"
                            + " 'ProjectAbstract' AS TEXTType \n"
                            + "                            from projectView\n"
                            + "                            where IFNULL(abstract,'')<>''  and (projectView.ProjectID in (SELECT links.projectId from Links))\n"
                            + " and projectView.Category1<>'NONFET'\n";

                    experimentDescription = (maxNumTopics > numTopics + 1) ? "Non Parametric" : "";

                    String sqlTextType = "select TEXTType, count(*) as Cnt from ( " + sql + " ) group by TEXTType";

                    grantType = "FP7 FET";
                    Statement statement = connection.createStatement();
                    statement.setQueryTimeout(60);  // set timeout to 30 sec.

                    experimentDescription += "Topic modeling analyzing:\n 1)";
                    ResultSet rs = statement.executeQuery(sqlTextType);
                    while (rs.next()) {
                        experimentDescription += rs.getInt("Cnt") + " ";
                        experimentDescription += rs.getString("TEXTType") + ", ";

                    }
                    experimentDescription += " related to " + grantType + "\n2)Related Research Areas\n3)Publication Venues (e.g., PubMed, Arxiv, ACM, Specific Journals)\n4)Grants per Publication \n SimilarityType:" + similarityType.toString();

                    // + " LIMIT 10000";
//                        " select Doc.DocId, Doc.text, GROUP_CONCAT(GrantPerDoc.GrantId,'\t') as GrantIds  "
//                        + " from Doc inner join "
//                        + " GrantPerDoc on Doc.DocId=GrantPerDoc.DocId "
//                        //  + " where  "
//                        //  + " Doc.source='" + docSource + "' and "
//                        //  + " grantPerDoc.grantId like '" + grantType + "' "
//                        + " Group By Doc.DocId, Doc.text";
                } else if (experimentType == ExperimentType.HEALTHTender) {

                    grantType = "FP7 HEALTH";
                    experimentDescription = (maxNumTopics > numTopics + 1) ? "Non Parametric" : "";
                    experimentDescription += "Topic modeling analyzing:\n1)Full Text of publications and project descriptions related to " + grantType + "\n2)Research Areas\n3)Venues (e.g., PubMed, Arxiv, ACM)\n4)Grants per Publication Links\n SimilarityType:" + similarityType.toString();

                    sql = "Select pubs.originalid AS DocId,\n"
                            + "                            CASE WHEN IFNULL(pubs.fulltext,'')='' THEN pubs.abstract ELSE pubs.fulltext END AS TEXT,\n"
                            + "                             --GROUP_CONCAT(CASE WHEN IFNULL(pubs.abstract,'')='' THEN pubs.fulltext ELSE pubs.abstract END,' ')  AS TEXT,\n"
                            + "                             GROUP_CONCAT(GrantId,'\t') as GrantIds,\n"
                            + "                             GROUP_CONCAT(Category3,'\t') as Areas, \n"
                            + "                             GROUP_CONCAT(Category3Descr,'\t') as AreasDescr, \n"
                            + "                        CASE WHEN IFNULL(journal,'')='' THEN   pubs.repository  else Journal END as Venue, \n"
                            + "                             IFNULL(GROUP_CONCAT(descriptorText,'\t'),'') as MESHdescriptors\n"
                            + "                            -- GROUP_CONCAT(qualifier,'\t') as MESHqualifiers\n"
                            + "                from pubs \n"
                            + "                             inner join links on links.OriginalId = pubs.originalid and links.funder='FP7' \n"
                            + "                              inner join projectView on links.project_code=projectView.GrantId and links.funder='FP7'  and Category2='HEALTH'\n"
                            + "                              --LEFT OUTER  join pmcmetadata on pmcmetadata.pmcid=pubs.originalid \n"
                            + "                              LEFT OUTER  join MeshTermsPerDoc on MeshTermsPerDoc.pmcid=pubs.originalid \n"
                            + "                               Group By pubs.originalid, pubs.fulltext, pubs.abstract, repository, journal \n"
                            + "  UNION \n"
                            + "                              select 'FP7_'||projectView.GrantId AS DocId, \n"
                            + "                              projectView.ABSTRACT AS TEXT, \n"
                            + "                             projectView.GrantId AS GrantIds ,                                               \n"
                            + "                             projectView.Category3 AS Areas,\n"
                            + "                             projectView.Category3Descr AS AreasDescr,\n"
                            + "                             '' AS Venue,\n"
                            + "                             '' AS MESHdescriptors\n"
                            + "                             --'' AS MESHqualifiers\n"
                            + "                             from projectView\n"
                            + "                             where IFNULL(abstract,'')<>'' and  Category2='HEALTH'  \n"
                            + "                             --and (projectView.GrantID in (SELECT links.project_code from Links where links.funder='FP7'))\n"
                            + " UNION \n"
                            + "                              select 'FP7RPT_'||ProjectId AS DocId, \n"
                            + "                             Results||' '||ContextObjectives||' '||PotentialImpact AS TEXT, \n"
                            + "                             ProjectId AS GrantIds, \n"
                            + "                             projectView.Category3 AS Areas,\n"
                            + "                             projectView.Category3Descr AS AreasDescr,\n"
                            + "                             '' AS Venue,\n"
                            + "                             '' AS MESHdescriptors\n"
                            + "                            -- '' AS MESHqualifiers\n"
                            + "                             from FP7FinalReports\n"
                            + "                             inner join projectView on ProjectId=projectView.GrantId  and Category2='HEALTH'  ";
                } else if (experimentType == ExperimentType.Authors) {
                    experimentDescription = "Topic modeling based on:\n 1)Full text NIPS publications\n2)Authors per publication links ";

                    sql = " select Doc.DocId,Doc.text, GROUP_CONCAT(AuthorPerDoc.authorID,'\t') as AuthorIds \n"
                            + "from Doc \n"
                            + "inner join AuthorPerDoc on Doc.DocId=AuthorPerDoc.DocId \n"
                            + "Where \n"
                            + "Doc.source='NIPS' \n"
                            + "Group By Doc.DocId, Doc.text";
                } else if (experimentType == ExperimentType.DBLP) {
                    sql = "			\n"
                            + " select id, title||' '||abstract AS text, Authors, \n"
                            + "  GROUP_CONCAT(citation.Target,',') as citations\n"
                            + " --GROUP_CONCAT(prLinks.Target,',') as citations,  GROUP_CONCAT(prLinks.Counts,',') as citationsCnt \n"
                            + " from papers\n"
                            + "--inner join  prLinks on prLinks.Source= papers.id   \n"
                            + "--AND prLinks.Counts>50\n"
                            + "left outer join citation on citation.Source= papers.id   \n"
                            + " WHERE \n"
                            + " (abstract IS NOT NULL) AND (abstract<>'') \n"
                            + "   Group By papers.id, papers.title, papers.abstract, papers.Authors";

//                        "   select id, title||' '||abstract AS text, Authors, GROUP_CONCAT(prLinks.Target,',') as citations,  GROUP_CONCAT(prLinks.Counts,',') as citationsCnt from papers\n"
//                        + "inner join  prLinks on prLinks.Source= papers.id AND prLinks.Counts>50 \n"
//                        + " WHERE (abstract IS NOT NULL) AND (abstract<>'')  \n"
//                        + " Group By papers.id, papers.title, papers.abstract, papers.Authors\n" 
                    // + " LIMIT 200000"
                } else if (experimentType == ExperimentType.PM_pdb) {
                    sql = " select   pubdoc.pmcId, pubdoc.abstract, pubdoc.body, GROUP_CONCAT(pdblink.pdbCode,'\t') as pbdCodes \n"
                            + "                        from pubdoc  inner join \n"
                            + "                        pdblink on pdblink.pmcId=pubdoc.pmcId \n"
                            + "                    Group By pubdoc.pmcId, pubdoc.abstract, pubdoc.body";
                } else if (experimentType == ExperimentType.DBLP_ACM) {
                    sql = " select id, title||' '||abstract AS text, Authors, \n"
                            + "  ref_id as citations\n"
                            + "  from papers\n"
                            + "  WHERE \n"
                            + " (abstract IS NOT NULL) AND (abstract<>'') \n " //+" AND ref_Id IS NOT NULL AND ref_id<>''  \n" 
                            // + " LIMIT 20000"
                            ;
                } else if (experimentType == ExperimentType.ACM) {
                    experimentDescription = "Topic modeling based on:\n1)Abstracts from ACM publications \n2)Authors\n3)Citations\n4)ACMCategories\n SimilarityType:"
                            + similarityType.toString()
                            + "\n Similarity on Authors & Categories";
                    //+ (ACMAuthorSimilarity ? "Authors" : "Categories");

                    sql = "  select    articleid as id, title||' '||abstract AS text, authors_id AS Authors, \n"
                            + "                 ref_objid as citations                 \n"
                            + "                 ,GROUP_CONCAT(CatId,'\t')  AS categories\n"
                            + "                             from ACMData1 \n"
                            + "                             INNER JOIN PubCategoryView on PubCategoryView.PubId = ArticleId\n"
                            + "                             Group by articleid \n";
                    //+ " LIMIT 100000";

                } else if (experimentType == ExperimentType.HEALTHTenderGrantGroup) {
                    experimentDescription = "Topic modeling based on:\n1)FP7 HEALTH related publications& project reports grouped by GrantId:"
                            + similarityType.toString()
                            + "\n Similarity";
                    //+ (ACMAuthorSimilarity ? "Authors" : "Categories");

                    sql = "Select GrantId,\n"
                            + "                                                       GROUP_CONCAT(CASE WHEN IFNULL(pubs.fulltext,'')='' THEN pubs.abstract ELSE pubs.fulltext END, ' ') AS TEXT,\n"
                            + "                                                        IFNULL(GROUP_CONCAT(descriptorText,'\t'),'') as MESHdescriptors\n"
                            + "                                                       \n"
                            + "                                           from pubs \n"
                            + "                                                        inner join links on links.OriginalId = pubs.originalid and links.funder='FP7' \n"
                            + "                                                         inner join projectView on links.project_code=projectView.GrantId and links.funder='FP7'  and Category2='HEALTH'\n"
                            + "                                                         LEFT OUTER  join MeshTermsPerDoc on MeshTermsPerDoc.pmcid=pubs.originalid \n"
                            + "                                                          Group By GrantId \n"
                            + "                                                          \n"
                            + "                             UNION \n"
                            + "                                                         select projectView.GrantId, \n"
                            + "                                                         projectView.ABSTRACT AS TEXT, \n"
                            + "                                                        '' AS MESHdescriptors\n"
                            + "                                                        from projectView\n"
                            + "                                                        where IFNULL(abstract,'')<>'' and  Category2='HEALTH'  \n"
                            + "                                                        \n"
                            + "                            UNION \n"
                            + "                            Select ProjectId AS GrantId, \n"
                            + "                                                                 Results||' '||ContextObjectives||' '||PotentialImpact AS TEXT, \n"
                            + "                                                           '' AS MESHdescriptors\n"
                            + "                           from FP7FinalReports\n"
                            + "                           inner join projectView on ProjectId=projectView.GrantId  and Category2='HEALTH'  ";
                    //+ " LIMIT 100000";

                }

                // String sql = "select fundedarxiv.file from fundedarxiv inner join funds on file=filename Group By fundedarxiv.file LIMIT 10" ;
                Statement statement = connection.createStatement();
                statement.setQueryTimeout(60);  // set timeout to 30 sec.
                ResultSet rs = statement.executeQuery(sql);
                String txt = "";
                while (rs.next()) {
                    // read the result set
                    //String lblStr = "[" + rs.getString("GrantIds") + "]" ;//+ rs.getString("text");
                    //String str = "[" + rs.getString("GrantIds") + "]" + rs.getString("text");
                    //System.out.println("name = " + rs.getString("file"));
                    //System.out.println("name = " + rs.getString("fundings"));
                    //int cnt = rs.getInt("grantsCnt");
                    switch (experimentType) {
                        case Grants:
                            instanceBuffer.get(0).add(new Instance(rs.getString("text"), null, rs.getString("DocId"), "text"));
                            if (numModalities > 1) {
                                instanceBuffer.get(1).add(new Instance(rs.getString("GrantIds"), null, rs.getString("DocId"), "grant"));
                            }
                            if (numModalities > 2) {
                                instanceBuffer.get(2).add(new Instance(rs.getString("Areas"), null, rs.getString("DocId"), "area"));
                            }
                            ;
                            if (numModalities > 3) {
                                instanceBuffer.get(3).add(new Instance(rs.getString("Source"), null, rs.getString("DocId"), "source"));
                            }
                            ;
                            break;
                        case FullGrants:
                            txt = rs.getString("text");

                            instanceBuffer.get(0).add(new Instance(txt.substring(0, Math.min(txt.length() - 1, 150000)), null, rs.getString("DocId"), "text"));
                            if (numModalities > 1) {
                                instanceBuffer.get(1).add(new Instance(rs.getString("GrantIds"), null, rs.getString("DocId"), "grant"));
                            }
                            if (numModalities > 2) {
                                instanceBuffer.get(2).add(new Instance(rs.getString("Areas"), null, rs.getString("DocId"), "area"));
                            }
                            ;
                            if (numModalities > 3) {
                                if (!rs.getString("Venue").equals("")) {
                                    instanceBuffer.get(3).add(new Instance(rs.getString("Venue"), null, rs.getString("DocId"), "Venue"));
                                }
                            }
                            ;
                            break;
                        case FETGrants:
                            txt = rs.getString("text");

                            instanceBuffer.get(0).add(new Instance(txt.substring(0, Math.min(txt.length() - 1, 150000)), null, rs.getString("DocId"), "text"));
                            if (numModalities > 1) {
                                instanceBuffer.get(1).add(new Instance(rs.getString("GrantIds"), null, rs.getString("DocId"), "grant"));
                            }
                            if (numModalities > 2) {
                                instanceBuffer.get(2).add(new Instance(rs.getString("Areas"), null, rs.getString("DocId"), "area"));
                            }
                            ;
                            if (numModalities > 3) {
                                if (!rs.getString("Venue").equals("")) {
                                    instanceBuffer.get(3).add(new Instance(rs.getString("Venue"), null, rs.getString("DocId"), "Venue"));
                                }
                            }
                            ;
                            break;
                        case HEALTHTender:
                            txt = rs.getString("text");
                            instanceBuffer.get(0).add(new Instance(txt.substring(0, Math.min(txt.length() - 1, 150000)), null, rs.getString("DocId"), "Text"));

                            if (numModalities > 1) {
                                instanceBuffer.get(1).add(new Instance(rs.getString("GrantIds"), null, rs.getString("DocId"), "Grant"));
                            }
                            if (numModalities > 2) {
                                instanceBuffer.get(2).add(new Instance(rs.getString("Areas"), null, rs.getString("DocId"), "Area"));
                            }
                            ;
                            if (numModalities > 3) {
                                if (!rs.getString("Venue").equals("")) {
                                    instanceBuffer.get(3).add(new Instance(rs.getString("Venue"), null, rs.getString("DocId"), "Venue"));
                                }
                            }
                            if (numModalities > 4) {

                                if (!rs.getString("MESHdescriptors").equals("")) {
                                    instanceBuffer.get(4).add(new Instance(rs.getString("MESHdescriptors"), null, rs.getString("DocId"), "MESHdescriptor"));
                                }
                            }
//                            if (numModalities > 5) {
//                                if (!rs.getString("MESHqualifiers").equals("")) {
//                                    instanceBuffer.get(5).add(new Instance(rs.getString("MESHqualifiers"), null, rs.getString("DocId"), "MESHqualifier"));
//                                }
                            //}

                            ;
                            break;
                        case Authors:
                            instanceBuffer.get(0).add(new Instance(rs.getString("text"), null, rs.getString("DocId"), "text"));
                            if (numModalities > 1) {
                                instanceBuffer.get(1).add(new Instance(rs.getString("AuthorIds"), null, rs.getString("DocId"), "author"));
                            }
                            break;
                        case DBLP:
                            instanceBuffer.get(0).add(new Instance(rs.getString("Text"), null, rs.getString("Id"), "text"));

                            if (numModalities > 1) {
                                if (DBLP_PPR) {
                                    String tmpStr = rs.getString("Citations");
                                    String[] citationsCnt = null;
                                    String[] citations = null;
                                    if (tmpStr != null) {
                                        citations = tmpStr.split(",");
                                        citationsCnt = rs.getString("CitationsCnt").split(",");

                                        String citationStr = "";
                                        int index = 0;

                                        while (index < citationsCnt.length) {
                                            int cnt = Integer.parseInt(citationsCnt[index]);
                                            for (int i = 1; i <= cnt / 50; i++) {
                                                if (citationStr != "") {
                                                    citationStr += ",";
                                                }
                                                citationStr += citations[index];
                                            }
                                            index++;

                                        }

                                        instanceBuffer.get(1).add(new Instance(citationStr, null, rs.getString("Id"), "citations"));
                                    }
                                } else {
                                    String tmpStr = rs.getString("Citations");
                                    if (tmpStr != null) {
                                        instanceBuffer.get(1).add(new Instance(tmpStr, null, rs.getString("Id"), "author"));
                                    }
                                }
                            }
                            if (numModalities > 2) {
                                instanceBuffer.get(2).add(new Instance(rs.getString("Authors"), null, rs.getString("Id"), "author"));
                            }
                            break;
                        case PM_pdb:
                            //instanceBuffer.get(0).add(new Instance(rs.getString("abstract") + " " + rs.getString("body"), null, rs.getString("pmcId"), "text"));
                            instanceBuffer.get(0).add(new Instance(rs.getString("abstract"), null, rs.getString("pmcId"), "text"));

                            if (numModalities > 1) {
                                instanceBuffer.get(1).add(new Instance(rs.getString("pbdCodes"), null, rs.getString("pmcId"), "pbdCode"));
                            }
                            break;
                        case DBLP_ACM:
                            instanceBuffer.get(0).add(new Instance(rs.getString("Text"), null, rs.getString("Id"), "text"));

                            if (numModalities > 1) {
                                instanceBuffer.get(1).add(new Instance(rs.getString("Authors"), null, rs.getString("Id"), "citation"));
                            }

                            if (numModalities > 2) {
                                String tmpStr = rs.getString("Citations").replace("\t", ",");
                                instanceBuffer.get(2).add(new Instance(tmpStr, null, rs.getString("Id"), "author"));
                            }
                            break;
                        case ACM:
                            instanceBuffer.get(0).add(new Instance(rs.getString("Text"), null, rs.getString("Id"), "text"));

                            if (numModalities > 1) {
                                String tmpStr = rs.getString("Citations");//.replace("\t", ",");
                                instanceBuffer.get(1).add(new Instance(tmpStr, null, rs.getString("Id"), "citation"));
                            }
                            if (numModalities > 2) {
                                String tmpStr = rs.getString("Categories");//.replace("\t", ",");
                                instanceBuffer.get(2).add(new Instance(tmpStr, null, rs.getString("Id"), "category"));
                            }
                            if (numModalities > 3) {
                                String tmpAuthorsStr = rs.getString("Authors");//.replace("\t", ",");
                                instanceBuffer.get(3).add(new Instance(tmpAuthorsStr, null, rs.getString("Id"), "author"));
                            }

                            break;
                        case HEALTHTenderGrantGroup:
                            txt = rs.getString("text");
                            instanceBuffer.get(0).add(new Instance(txt.substring(0, Math.min(txt.length() - 1, 150000)), null, rs.getString("GrantId"), "Text"));

                            if (numModalities > 1) {

                                if (!rs.getString("MESHdescriptors").equals("")) {
                                    instanceBuffer.get(1).add(new Instance(rs.getString("MESHdescriptors"), null, rs.getString("GrantId"), "MESHdescriptor"));
                                }
                            }
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

            logger.info("Read " + instanceBuffer.get(0).size() + " instances modality: " + instanceBuffer.get(0).get(0).getSource().toString());

            //numModalities = 2;
            // Begin by importing documents from text to feature sequences
            ArrayList<Pipe> pipeListText = new ArrayList<Pipe>();

            // Pipes: lowercase, tokenize, remove stopwords, map to features
            pipeListText.add(new Input2CharSequence(false)); //homer
            pipeListText.add(new CharSequenceLowercase());

            SimpleTokenizer tokenizer = new SimpleTokenizer(new File("stoplists/en.txt"));
            GenerateStoplist(tokenizer, instanceBuffer.get(0), pruneCnt, pruneMaxPerc, false);

            tokenizer.stop("cid");
            tokenizer.stop("null");

            //tokenizer.
            pipeListText.add(tokenizer);
            Alphabet alphabet = new Alphabet();
            pipeListText.add(new StringList2FeatureSequence(alphabet));

            /*
             Alphabet alphabet = new Alphabet();
             CharSequenceLowercase csl = new CharSequenceLowercase();
             StringList2FeatureSequence sl2fs = new StringList2FeatureSequence(alphabet);
             if (!preserveCase.value) {
             pipes.add(csl);
             }
             pipes.add(prunedTokenizer);
             pipes.add(sl2fs);

             Pipe serialPipe = new SerialPipes(pipes);

             InstanceList instances = new InstanceList(serialPipe);
             instances.addThruPipe(reader);
        
             */
            /* orig
             pipeListText.add(new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")));// Original --> replaced by simpletokenizer in order to use prunedStopiList
             pipeListText.add(new TokenSequenceRemoveStopwords(new File("stoplists/en.txt"), "UTF-8", false, false, false)); //Original --> replaced by simpletokenizer
             pipeListText.add(new TokenSequence2FeatureSequence()); 
             */
            // Other Modalities
            ArrayList<Pipe> pipeListCSV = new ArrayList<Pipe>();
            if (experimentType == ExperimentType.DBLP || experimentType == ExperimentType.DBLP_ACM) {
                pipeListCSV.add(new CSV2FeatureSequence(","));
            } else {
                pipeListCSV.add(new CSV2FeatureSequence());
            }

            InstanceList[] instances = new InstanceList[numModalities];

            instances[0] = new InstanceList(new SerialPipes(pipeListText));
            instances[0].addThruPipe(instanceBuffer.get(0).iterator());

            for (byte m = 1; m < numModalities; m++) {
                logger.info("Read " + instanceBuffer.get(m).size() + " instances modality: " + (instanceBuffer.get(m).size() > 0 ? instanceBuffer.get(m).get(0).getSource().toString() : m));
                instances[m] = new InstanceList(new SerialPipes(pipeListCSV));
                instances[m].addThruPipe(instanceBuffer.get(m).iterator());
            }

            logger.info(" instances added through pipe");

            if (calcTokensPerEntity) {
                //TfIdfWeighting(instances[1], SQLLitedb, experimentId, 1);
                TfIdfWeighting(instances[0], SQLLitedb, experimentId, 0);
            }

            if (runTopicModelling) {
// pruning for all other modalities no text
                for (byte m = 1; m < numModalities; m++) {
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

                logger.info(" instances pruned");
                boolean splitCorpus = false;
                InstanceList[] testInstances = null;
                InstanceList[] trainingInstances = instances;
                if (splitCorpus) {
                    //instances.addThruPipe(new FileIterator(inputDir));
                    //instances.addThruPipe (new FileIterator ("C:\\UoA\\OpenAire\\Datasets\\YIpapersTXT\\YIpapersTXT"));
                    //
                    //instances.addThruPipe(new CsvIterator (fileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
                    // Create a model with 100 topics, alpha_t = 0.01, beta_w = 0.01
                    //  Note that the first parameter is passed as the sum over topics, while
                    //  the second is 
                    testInstances = new InstanceList[numModalities];

                    trainingInstances = new InstanceList[numModalities];

                    TObjectIntMap<String> entityPosition = new TObjectIntHashMap<String>();
                    int index = 0;
                    for (byte m = 0; m < numModalities; m++) {
                        Noop newPipe = new Noop(instances[m].getDataAlphabet(), instances[m].getTargetAlphabet());
                        InstanceList newInstanceList = new InstanceList(newPipe);
                        testInstances[m] = newInstanceList;
                        InstanceList newInstanceList2 = new InstanceList(newPipe);
                        trainingInstances[m] = newInstanceList2;
                        for (int i = 0; i < instances[m].size(); i++) {
                            Instance instance = instances[m].get(i);
                            String entityId = (String) instance.getName();
                            if (i < instances[m].size() * 0.8 && m == 0) {
                                entityPosition.put(entityId, index);
                                trainingInstances[m].add(instance);
                                index++;
                            } else if (m != 0 && entityPosition.containsKey(entityId)) {
                                trainingInstances[m].add(instance);
                            } else {
                                testInstances[m].add(instance);
                            }
                        }
                    }
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
                    HDP hdp = new HDP(1.0, 0.1, 4.0, numTopics);
                    hdp.initialize(instances[0]);

                    //set number of iterations, and display result or not 
                    hdp.estimate(numIterations);

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

                boolean runFastParallelModel = false;
                if (runFastParallelModel) {
                    FastQParallelTopicModel modelOrig = new FastQParallelTopicModel(numTopics, 1.0, 0.01);

                    modelOrig.addInstances(instances[0]);

                    // Use two parallel samplers, which each look at one half the corpus and combine
                    //  statistics after every iteration.
                    modelOrig.setNumThreads(3);
                    // Run the model for 50 iterations and stop (this is for testing only, 
                    //  for real applications, use 1000 to 2000 iterations)
                    modelOrig.setNumIterations(numIterations);
                    modelOrig.optimizeInterval = optimizeInterval;
                    modelOrig.burninPeriod = burnIn;
                    //model.optimizeInterval = 0;
                    //model.burninPeriod = 0;
                    //model.saveModelInterval=250;
                    modelOrig.estimate();
                }

                boolean runOrigParallelModel = true;
                if (runOrigParallelModel) {
                    ParallelTopicModel modelOrig = new ParallelTopicModel(numTopics, 1.0, 0.01);

                    modelOrig.addInstances(instances[0]);

                    // Use two parallel samplers, which each look at one half the corpus and combine
                    //  statistics after every iteration.
                    modelOrig.setNumThreads(3);
                    // Run the model for 50 iterations and stop (this is for testing only, 
                    //  for real applications, use 1000 to 2000 iterations)
                    modelOrig.setNumIterations(numIterations);
                    modelOrig.optimizeInterval = optimizeInterval;
                    modelOrig.burninPeriod = burnIn;
                    //model.optimizeInterval = 0;
                    //model.burninPeriod = 0;
                    //model.saveModelInterval=250;
                    modelOrig.estimate();
                }

                double[] beta = new double[numModalities];
                Arrays.fill(beta, 0.01);

                double[] alphaSum = new double[numModalities];
                Arrays.fill(alphaSum, 1);

                double[] gamma = new double[numModalities];
                Arrays.fill(gamma, 1);

                double gammaRoot = 4;

                //Non parametric model
                //iMixLDAParallelTopicModel model = new iMixLDAParallelTopicModel(maxNumTopics, numTopics, numModalities, gamma, gammaRoot, beta, numIterations);
                //parametric model
                MixLDAParallelTopicModel model = new MixLDAParallelTopicModel(numTopics, numModalities, alphaSum, beta, numIterations);

                // ParallelTopicModel model = new ParallelTopicModel(numTopics, 1.0, 0.01);
                //model.setNumIterations(numIterations);
                model.setIndependentIterations(independentIterations);
                model.optimizeInterval = optimizeInterval;
                model.burninPeriod = burnIn;

                model.addInstances(instances);//trainingInstances);//instances);

                logger.info(" instances added");

                // Use two parallel samplers, which each look at one half the corpus and combine
                //  statistics after every iteration.
                model.setNumThreads(4);
            // Run the model for 50 iterations and stop (this is for testing only, 
                //  for real applications, use 1000 to 2000 iterations)

                //model.optimizeInterval = 0;
                //model.burninPeriod = 0;
                //model.saveModelInterval=250;
                model.estimate();

                logger.info("Model estimated");
                model.saveTopics(SQLLitedb, experimentId);

                logger.info("Topics Saved");

                model.printTopWords(
                        new File(topicKeysFile), topWords, topLabels, false);
                logger.info("Top words printed");
            //model.printTopWords(new File(topicKeysFile), topWords,  false);
                //model.printTopicWordWeights(new File(topicWordWeightsFile));
                //model.printTopicLabelWeights(new File(topicLabelWeightsFile));

                //No printing state 
                //model.printState(
                //        new File(stateFileZip));
                // logger.info("printState finished");
                PrintWriter outState = null;// new PrintWriter(new FileWriter((new File(outputDocTopicsFile))));

                model.printDocumentTopics(outState, docTopicsThreshold, docTopicsMax, SQLLitedb, experimentId,
                        0.1);

                if (outState != null) {
                    outState.close();
                }

                logger.info("printDocumentTopics finished");

                logger.info("Model Metadata: \n" + model.getExpMetadata());

                model.saveExperiment(SQLLitedb, experimentId, experimentDescription);

                if (modelEvaluationFile != null) {
                    try {

//                ObjectOutputStream oos =
//                        new ObjectOutputStream(new FileOutputStream(modelEvaluationFile));
//                oos.writeObject(model.getProbEstimator());
//                oos.close();
//                
                        PrintStream docProbabilityStream = null;
                        docProbabilityStream = new PrintStream(modelEvaluationFile);
//TODO...
                        double perplexity = 0;
                        if (splitCorpus) {
                            perplexity = model.getProbEstimator().evaluateLeftToRight(testInstances[0], 10, false, docProbabilityStream);
                        }
                        //  System.out.println("perplexity for the test set=" + perplexity);
                        logger.info("perplexity calculation finished");
                        //iMixLDATopicModelDiagnostics diagnostics = new iMixLDATopicModelDiagnostics(model, topWords);
                        MixLDATopicModelDiagnostics diagnostics = new MixLDATopicModelDiagnostics(model, topWords);
                        diagnostics.saveToDB(SQLLitedb, experimentId, perplexity);
                        logger.info("full diagnostics calculation finished");

                    } catch (Exception e) {
                        System.err.println(e.getMessage());
                    }

                }

                //PrintWriter outXMLPhrase = new PrintWriter(new FileWriter((new File(outputTopicPhraseXMLReport))));
                //model.topicPhraseXMLReport(outXMLPhrase, topWords);
                //outState.close();
                //logger.info("topicPhraseXML report finished");
//        GunZipper g = new GunZipper(new File(stateFileZip));
//
//        g.unzip(
//                new File(stateFile));
//
//        try {
//            // outputCsvFiles(outputDir, true, inputDir, numTopics, stateFile, outputDocTopicsFile, topicKeysFile);
//            logger.info("outputCsvFiles finished");
//        } catch (Exception e) {
//            // if the error message is "out of memory", 
//            // it probably means no database file is found
//            System.err.println(e.getMessage());
//        }
            }
        }
        if (calcSimilarities) {

            //calc similarities
            logger.info("similarities calculation Started");
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
                switch (experimentType) {
                    case Grants:
                        sql = "select    GrantId, TopicId, AVG(weight) as Weight from PubTopic Inner Join GrantPerDoc on PubTopic.PubId= GrantPerDoc.DocId"
                                + " where weight>0.02 AND ExperimentId='" + experimentId + "' group By GrantId , TopicId order by  GrantId   , TopicId";
                        break;
                    case FullGrants:
                        sql = "select    links.projectId  as ProjectId, TopicId, AVG(weight) as Weight from PubTopic Inner Join links  on PubTopic.PubId= links.OriginalId "
                                + " where weight>0.02 AND ExperimentId='" + experimentId
                                + "' group By links.projectId , TopicId order by  links.projectId, TopicId";

                        break;
                    case FETGrants:
                        sql = "select    links.projectId as ProjectId, TopicId, AVG(weight) as Weight from PubTopic Inner Join links  on PubTopic.PubId= links.OriginalId "
                                + "   inner join projectView on links.projectId=projectView.projectId  and Category1<>'NONFET'\n"
                                + " where weight>0.02 AND ExperimentId='" + experimentId
                                + "' group By links.projectId , TopicId order by  links.projectId, TopicId";

                        break;
                    case HEALTHTender:
                        sql = "select    links.projectId  as ProjectId, TopicId, AVG(weight) as Weight from PubTopic Inner Join links  on PubTopic.PubId= links.OriginalId "
                                + "   inner join projectView on links.projectId=projectView.projectId and links.funder='FP7'  and Category2='HEALTH'\n"
                                + " where weight>0.02 AND ExperimentId='" + experimentId
                                + "' group By links.projectId , TopicId order by  links.projectId, TopicId";

                        break;
                    case Authors:
                        sql = "select    AuthorId, TopicId, AVG(weight) as Weight from PubTopic Inner Join AuthorPerDoc on PubTopic.PubId= AuthorPerDoc.DocId"
                                + " where weight>0.02 AND ExperimentId='" + experimentId + "' group By AuthorId , TopicId order by  AuthorId   , TopicId";
                        break;
                    case ACM:
                        if (ACMAuthorSimilarity) {
                            sql = "select    PubAuthor.AuthorId, TopicId, AVG(weight) as Weight from PubTopic \n"
                                    + "Inner Join PubAuthor on PubTopic.PubId= PubAuthor.PubId  \n"
                                    + "INNER JOIN (Select AuthorId FROM PubAuthor\n"
                                    + "GROUP BY AuthorId HAVING Count(*)>10) catCnts1 ON catCnts1.AuthorId = PubAuthor.AuthorId "
                                    + " where weight>0.02 AND ExperimentId='" + experimentId + "' group By PubAuthor.AuthorId,  TopicId order by  PubAuthor.AuthorId   ,weight desc, TopicId";
                        } else {
                            sql = "select    PubCategory.CatId as Category, TopicId, AVG(weight) as Weight from PubTopic \n"
                                    + "Inner Join PubCategory on PubTopic.PubId= PubCategory.PubId  \n"
                                    + "INNER JOIN (Select CatId FROM PubCategory \n"
                                    + "GROUP BY CatId HAVING Count(*)>10) catCnts1 ON catCnts1.CatId = PubCategory.catId\n"
                                    + "where weight>0.02 AND ExperimentId='" + experimentId + "' group By PubCategory.CatId , TopicId order by  PubCategory.CatId, Weight desc, TopicId";
                        }

                        break;
                    case PM_pdb:
                        sql = "select    pdbCode, TopicId, AVG(weight) as Weight from PubTopic Inner Join pdblink on PubTopic.PubId= pdblink.pmcId"
                                + " where weight>0.02 AND ExperimentId='" + experimentId + "' group By pdbCode , TopicId order by  pdbCode   , TopicId";

                        break;
                    case DBLP:
                        sql = "select  Source, TopicId, AVG(weight) as Weight from PubTopic Inner Join prlinks on PubTopic.PubId= prlinks.source"
                                + " where weight>0.02 AND ExperimentId='" + experimentId + "' group By Source , TopicId order by  Source   , TopicId";

                        break;
                    default:
                }

                // String sql = "select fundedarxiv.file from fundedarxiv inner join funds on file=filename Group By fundedarxiv.file LIMIT 10" ;
                ResultSet rs = statement.executeQuery(sql);

                HashMap<String, SparseVector> labelVectors = null;
                HashMap<String, double[]> similarityVectors = null;
                if (similarityType == SimilarityType.cos) {
                    labelVectors = new HashMap<String, SparseVector>();
                } else {
                    similarityVectors = new HashMap<String, double[]>();
                }

                String labelId = "";
                int[] topics = new int[maxNumTopics];
                double[] weights = new double[maxNumTopics];
                int cnt = 0;
                double a;
                while (rs.next()) {

                    String newLabelId = "";

                    switch (experimentType) {
                        case Grants:

                            newLabelId = rs.getString("GrantId");
                            break;
                        case FullGrants:
                        case FETGrants:
                        case HEALTHTender:
                            newLabelId = rs.getString("projectId");
                            break;
                        case Authors:
                            newLabelId = rs.getString("AuthorId");
                            break;
                        case ACM:
                            if (ACMAuthorSimilarity) {
                                newLabelId = rs.getString("AuthorId");
                            } else {
                                newLabelId = rs.getString("Category");
                            }
                            break;
                        case PM_pdb:
                            newLabelId = rs.getString("pdbCode");
                            break;
                        case DBLP:
                            newLabelId = rs.getString("Source");
                            break;
                        default:
                    }

                    if (!newLabelId.equals(labelId) && !labelId.isEmpty()) {
                        if (similarityType == SimilarityType.cos) {
                            labelVectors.put(labelId, new SparseVector(topics, weights, topics.length, topics.length, true, true, true));
                        } else {
                            similarityVectors.put(labelId, weights);
                        }
                        topics = new int[maxNumTopics];
                        weights = new double[maxNumTopics];
                        cnt = 0;
                    }
                    labelId = newLabelId;
                    topics[cnt] = rs.getInt("TopicId");
                    weights[cnt] = rs.getDouble("Weight");
                    cnt++;

                }

                cnt = 0;
                double similarity = 0;
                double similarityThreshold = 0.15;
                NormalizedDotProductMetric cosineSimilarity = new NormalizedDotProductMetric();

                int entityType = experimentType.ordinal();
                if (experimentType == ExperimentType.ACM && !ACMAuthorSimilarity) {
                    entityType = 100 + entityType;
                };

                statement.executeUpdate("create table if not exists EntitySimilarity (EntityType int, EntityId1 nvarchar(50), EntityId2 nvarchar(50), Similarity double, ExperimentId nvarchar(50)) ");
                String deleteSQL = String.format("Delete from EntitySimilarity where  ExperimentId = '%s' and entityType=%d", experimentId, entityType);
                statement.executeUpdate(deleteSQL);

                PreparedStatement bulkInsert = null;
                sql = "insert into EntitySimilarity values(?,?,?,?,?);";

                try {

                    connection.setAutoCommit(false);
                    bulkInsert = connection.prepareStatement(sql);

                    if (similarityType == SimilarityType.Jen_Sha_Div) {
                        for (String fromGrantId : similarityVectors.keySet()) {
                            boolean startCalc = false;

                            for (String toGrantId : similarityVectors.keySet()) {
                                if (!fromGrantId.equals(toGrantId) && !startCalc) {
                                    continue;
                                } else {
                                    startCalc = true;
                                    similarity = Maths.jensenShannonDivergence(similarityVectors.get(fromGrantId), similarityVectors.get(toGrantId)); // the function returns distance not similarity
                                    if (similarity > similarityThreshold && !fromGrantId.equals(toGrantId)) {

                                        bulkInsert.setInt(1, entityType);
                                        bulkInsert.setString(2, fromGrantId);
                                        bulkInsert.setString(3, toGrantId);
                                        bulkInsert.setDouble(4, (double) Math.round(similarity * 1000) / 1000);
                                        bulkInsert.setString(5, experimentId);
                                        bulkInsert.executeUpdate();
                                    }
                                }
                            }
                        }
                    } else if (similarityType == SimilarityType.cos) {
                        for (String fromGrantId : labelVectors.keySet()) {
                            boolean startCalc = false;

                            for (String toGrantId : labelVectors.keySet()) {
                                if (!fromGrantId.equals(toGrantId) && !startCalc) {
                                    continue;
                                } else {
                                    startCalc = true;
                                    similarity = 1 - Math.abs(cosineSimilarity.distance(labelVectors.get(fromGrantId), labelVectors.get(toGrantId))); // the function returns distance not similarity
                                    if (similarity > similarityThreshold && !fromGrantId.equals(toGrantId)) {
                                        bulkInsert.setInt(1, entityType);
                                        bulkInsert.setString(2, fromGrantId);
                                        bulkInsert.setString(3, toGrantId);
                                        bulkInsert.setDouble(4, (double) Math.round(similarity * 1000) / 1000);
                                        bulkInsert.setString(5, experimentId);
                                        bulkInsert.executeUpdate();
                                    }
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

            logger.info("similarities calculation finished");
        }

//        if (modelDiagnosticsFile
//                != null) {
//            PrintWriter out = new PrintWriter(modelDiagnosticsFile);
//            MixTopicModelDiagnostics diagnostics = new MixTopicModelDiagnostics(model, topWords, perplexity);
//            diagnostics.saveToDB(SQLLitedb, experimentId);
//            out.println(diagnostics.toXML()); //preferable than XML???
//            out.close();
//        }
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

    private void SaveTopKTokensPerEntity(int K, boolean TfIDFweighting, InstanceList instances) {

    }

    private void TfIdfWeighting(InstanceList instances, String SQLLiteDB, String experimentId, int itemType) {

        int N = instances.size();

        Alphabet alphabet = instances.getDataAlphabet();
        Object[] tokens = alphabet.toArray();
        System.out.println("# Number of dimensions: " + tokens.length);
        // determine document frequency for each term
        int[] df = new int[tokens.length];
        for (Instance instance : instances) {
            FeatureVector fv = new FeatureVector((FeatureSequence) instance.getData());
            int[] indices = fv.getIndices();
            for (int index : indices) {
                df[index]++;
            }
        }

        // determine document length for each document
        int[] lend = new int[N];
        double lenavg = 0;
        for (int i = 0; i < N; i++) {
            Instance instance = instances.get(i);
            FeatureVector fv = new FeatureVector((FeatureSequence) instance.getData());
            int[] indices = fv.getIndices();
            double length = 0.0;
            for (int index : indices) {
                length += fv.value(index);
            }
            lend[i] = (int) length;
            lenavg += length;
        }
        if (N > 1) {
            lenavg /= (double) N;
        }

        Connection connection = null;
        Statement statement = null;
        PreparedStatement bulkInsert = null;

        try {
            // create a database connection
            if (!SQLLiteDB.isEmpty()) {
                connection = DriverManager.getConnection(SQLLiteDB);
                statement = connection.createStatement();
                statement.setQueryTimeout(30);  // set timeout to 30 sec.
                statement.executeUpdate("create table if not exists TokensPerEntity (EntityId nvarchar(100), ItemType int, Token nvarchar(100), Counts double, TFIDFCounts double, ExperimentId nvarchar(50)) ");

                statement.executeUpdate("create Index if not exists IX_TokensPerEntity_Entity_Counts ON TokensPerEntity ( EntityId, ExperimentId, ItemType, Counts DESC, TFIDFCounts DESC, Token)");
                statement.executeUpdate("create Index if not exists IX_TokensPerEntity_Entity_TFIDFCounts ON TokensPerEntity ( EntityId, ExperimentId, ItemType,  TFIDFCounts DESC, Counts DESC, Token)");

                statement.executeUpdate("create View if not exists TokensPerEntityView AS select rv1.EntityId, rv1.ItemType, rv1.Token, rv1.Counts, rv1.TFIDFCounts, rv1.ExperimentId \n"
                        + "FROM TokensPerEntity rv1\n"
                        + "WHERE Token in\n"
                        + "(\n"
                        + "SELECT Token\n"
                        + "FROM TokensPerEntity rv2\n"
                        + "WHERE EntityId = rv1.EntityId AND Counts>2 AND ItemType=rv1.ItemType AND ExperimentId=rv1.ExperimentId \n"
                        + "ORDER BY\n"
                        + "TFIDFCounts DESC\n"
                        + "LIMIT 20\n"
                        + ")");

                String deleteSQL = String.format("Delete from TokensPerEntity where  ExperimentId = '%s' and itemtype= %d", experimentId, itemType);
                statement.executeUpdate(deleteSQL);

                String sql = "insert into TokensPerEntity values(?,?,?,?,?,?);";

                connection.setAutoCommit(false);
                bulkInsert = connection.prepareStatement(sql);

                for (int i = 0; i < N; i++) {
                    Instance instance = instances.get(i);

                    FeatureVector fv = new FeatureVector((FeatureSequence) instance.getData());
                    int[] indices = fv.getIndices();
                    for (int index : indices) {
                        double tf = fv.value(index);
                        double tfcomp = tf / (tf + 0.5 + 1.5 * (double) lend[i] / lenavg);
                        double idfcomp = Math.log((double) N / (double) df[index]) / Math.log(N + 1);
                        double tfIdf = tfcomp * idfcomp;
                        fv.setValue(index, tfIdf);
                        String token = fv.getAlphabet().lookupObject(index).toString();

                        bulkInsert.setString(1, instance.getName().toString());
                        bulkInsert.setInt(2, itemType);
                        bulkInsert.setString(3, token);
                        bulkInsert.setDouble(4, tf);
                        bulkInsert.setDouble(5, tfIdf);
                        bulkInsert.setString(6, experimentId);

                        bulkInsert.executeUpdate();
                    }
                }

                connection.commit();
            }
        } catch (SQLException e) {

            if (connection != null) {
                try {
                    System.err.print("Transaction is being rolled back");
                    connection.rollback();
                } catch (SQLException excep) {
                    System.err.print("Error in insert TokensPerEntity");
                }
            }
        } finally {
            try {
                if (bulkInsert != null) {
                    bulkInsert.close();
                }
                connection.setAutoCommit(true);
            } catch (SQLException excep) {
                System.err.print("Error in insert TokensPerEntity");
            }
        }

        //TODO: Sort Feature Vector Values
        // FeatureVector.toSimpFilefff
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

        Iterator<String> wordIter = alphabet.iterator();
        while (wordIter.hasNext()) {
            String word = (String) wordIter.next();
            if (word.contains("cidcid") || word.contains("nullnull") || word.contains("usepackage") || word.contains("fig")) {
                prunedTokenizer.stop(word);
            }
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

    public void createCitationGraphFile(String outputCsv, String SQLLitedb) {
        //String SQLLitedb = "jdbc:sqlite:C:/projects/OpenAIRE/fundedarxiv.db";

        Connection connection = null;
        try {

            FileWriter fwrite = new FileWriter(outputCsv);
            BufferedWriter out = new BufferedWriter(fwrite);
            String header = "# DBLP citation graph \n"
                    + "# fromNodeId, toNodeId \n";
            out.write(header);

            connection = DriverManager.getConnection(SQLLitedb);

            String sql = "select id, ref_id from papers where ref_num >0 ";
            Statement statement = connection.createStatement();
            statement.setQueryTimeout(30);  // set timeout to 30 sec.
            ResultSet rs = statement.executeQuery(sql);
            while (rs.next()) {
                // read the result set
                int Id = rs.getInt("Id");
                String citationNums = rs.getString("ref_id");

                String csvLine = "";//Id + "\t" + citationNums;

                String[] str = citationNums.split("\t");
                for (int i = 0; i < str.length - 1; i++) {
                    csvLine = Id + "\t" + str[i];
                    out.write(csvLine + "\n");
                }

            }
            out.flush();
        } catch (SQLException e) {
            // if the error message is "out of memory", 
            // it probably means no database file is found
            System.err.println(e.getMessage());
        } catch (Exception e) {
            System.err.println("File input error");
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

    public void createRefACMTables(String SQLLitedb) {
        //String SQLLitedb = "jdbc:sqlite:C:/projects/OpenAIRE/fundedarxiv.db";

        Connection connection = null;
        try {

            connection = DriverManager.getConnection(SQLLitedb);

            Statement statement = connection.createStatement();
            statement.executeUpdate("create table if not exists Author (AuthorId nvarchar(50), FirstName nvarchar(50), LastName nvarchar(50), MiddleName nvarchar(10), Affilication TEXT) ");
            String deleteSQL = String.format("Delete from Author ");
            statement.executeUpdate(deleteSQL);

            statement.executeUpdate("create table if not exists Citation (CitationId nvarchar(50), Reference TEXT) ");
            deleteSQL = String.format("Delete from Citation ");
            statement.executeUpdate(deleteSQL);

            statement.executeUpdate("create table if not exists Category (Category TEXT) ");
            deleteSQL = String.format("Delete from Category ");
            statement.executeUpdate(deleteSQL);

            statement = connection.createStatement();
            statement.executeUpdate("create table if not exists PubAuthor (PubId nvarchar(50), AuthorId nvarchar(50)) ");
            deleteSQL = String.format("Delete from PubAuthor ");
            statement.executeUpdate(deleteSQL);

            statement.executeUpdate("create table if not exists PubCitation (PubId nvarchar(50), CitationId nvarchar(50)) ");
            deleteSQL = String.format("Delete from PubCitation ");
            statement.executeUpdate(deleteSQL);

            statement.executeUpdate("create table if not exists PubCategory (PubId nvarchar(50), Category TEXT) ");
            deleteSQL = String.format("Delete from PubCategory ");
            statement.executeUpdate(deleteSQL);

            PreparedStatement authorBulkInsert = null;
            PreparedStatement citationBulkInsert = null;
            PreparedStatement categoryBulkInsert = null;

            PreparedStatement pubAuthorBulkInsert = null;
            PreparedStatement pubCitationBulkInsert = null;
            PreparedStatement pubCategoryBulkInsert = null;

            String authorInsertsql = "insert into Author values(?,?,?,?,?);";
            String citationInsertsql = "insert into Citation values(?,?);";
            String categoryInsertsql = "insert into Category values(?);";

            String pubAuthorInsertsql = "insert into pubAuthor values(?,?);";
            String pubCitationInsertsql = "insert into pubCitation values(?,?);";
            String pubCategoryInsertsql = "insert into pubCategory values(?,?);";

            TObjectIntHashMap<String> authorsLst = new TObjectIntHashMap<String>();
            TObjectIntHashMap<String> citationsLst = new TObjectIntHashMap<String>();
            TObjectIntHashMap<String> categorysLst = new TObjectIntHashMap<String>();

            try {

                connection.setAutoCommit(false);
                authorBulkInsert = connection.prepareStatement(authorInsertsql);
                citationBulkInsert = connection.prepareStatement(citationInsertsql);
                categoryBulkInsert = connection.prepareStatement(categoryInsertsql);

                pubAuthorBulkInsert = connection.prepareStatement(pubAuthorInsertsql);
                pubCitationBulkInsert = connection.prepareStatement(pubCitationInsertsql);
                pubCategoryBulkInsert = connection.prepareStatement(pubCategoryInsertsql);

                String sql = "	Select articleid,authors_id,authors_firstname,authors_lastname,authors_middlename,authors_affiliation,authors_role, \n"
                        + "			ref_objid,reftext,primarycategory,othercategory \n"
                        + "			 from ACMData1 \n";
                // + "			  LIMIT 10";

                statement.setQueryTimeout(30);  // set timeout to 30 sec.

                ResultSet rs = statement.executeQuery(sql);

                while (rs.next()) {
                    // read the result set
                    String Id = rs.getString("articleid");

                    String authorIdsStr = rs.getString("authors_id");
                    String[] authorIds = authorIdsStr.split("\t");

                    String authors_firstnamesStr = rs.getString("authors_firstname");
                    String[] authors_firstnames = authors_firstnamesStr.split("\t");

                    String authors_lastnamesStr = rs.getString("authors_lastname");
                    String[] authors_lastnames = authors_lastnamesStr.split("\t");

                    String authors_middlenamesStr = rs.getString("authors_middlename");
                    String[] authors_middlenames = authors_middlenamesStr.split("\t");

                    String authors_affiliationsStr = rs.getString("authors_affiliation");
                    String[] authors_affiliations = authors_affiliationsStr.split("\t");

                    for (int i = 0; i < authorIds.length - 1; i++) {
                        String authorId = authorIds[i];
                        if (!authorsLst.containsKey(authorId)) {
                            authorsLst.put(authorId, 1);
                            String lstName = authors_lastnames.length - 1 > i ? authors_lastnames[i] : "";
                            String fstName = authors_firstnames.length - 1 > i ? authors_firstnames[i] : "";
                            String mName = authors_middlenames.length - 1 > i ? authors_middlenames[i] : "";
                            String affiliation = authors_affiliations.length - 1 > i ? authors_affiliations[i] : "";

                            authorBulkInsert.setString(1, authorId);
                            authorBulkInsert.setString(2, lstName);
                            authorBulkInsert.setString(3, fstName);
                            authorBulkInsert.setString(4, mName);
                            authorBulkInsert.setString(5, affiliation);
                            authorBulkInsert.executeUpdate();
                        }
                        pubAuthorBulkInsert.setString(1, Id);
                        pubAuthorBulkInsert.setString(2, authorId);
                        pubAuthorBulkInsert.executeUpdate();

                    }

                    String citationIdsStr = rs.getString("ref_objid");
                    String[] citationIds = citationIdsStr.split("\t");

                    String citationsStr = rs.getString("reftext");
                    String[] citations = citationsStr.split("\t");

                    for (int i = 0; i < citationIds.length - 1; i++) {
                        String citationId = citationIds[i];
                        if (!citationsLst.containsKey(citationId)) {
                            citationsLst.put(citationId, 1);
                            String ref = citations.length - 1 > i ? citations[i] : "";

                            citationBulkInsert.setString(1, citationId);
                            citationBulkInsert.setString(2, ref);
                            citationBulkInsert.executeUpdate();
                        }
                        pubCitationBulkInsert.setString(1, Id);
                        pubCitationBulkInsert.setString(2, citationId);
                        pubCitationBulkInsert.executeUpdate();
                    }

                    String prCategoriesStr = rs.getString("primarycategory");
                    String[] prCategories = prCategoriesStr.split("\t");

                    String categoriesStr = rs.getString("othercategory");
                    String[] categories = categoriesStr.split("\t");

                    for (int i = 0; i < prCategories.length - 1; i++) {
                        String category = prCategories[i];
                        if (!categorysLst.containsKey(category)) {
                            categorysLst.put(category, 1);
                            categoryBulkInsert.setString(1, category);
                            categoryBulkInsert.executeUpdate();
                        }
                        pubCategoryBulkInsert.setString(1, Id);
                        pubCategoryBulkInsert.setString(2, category);
                        pubCategoryBulkInsert.executeUpdate();
                    }

                    for (int i = 0; i < categories.length - 1; i++) {
                        String category = categories[i];
                        if (!categorysLst.containsKey(category)) {
                            categorysLst.put(category, 1);
                            categoryBulkInsert.setString(1, category);
                            categoryBulkInsert.executeUpdate();
                        }

                        pubCategoryBulkInsert.setString(1, Id);
                        pubCategoryBulkInsert.setString(2, category);
                        pubCategoryBulkInsert.executeUpdate();
                    }

                }

                connection.commit();

            } catch (SQLException e) {

                if (connection != null) {
                    try {
                        System.err.print("Transaction is being rolled back");
                        connection.rollback();
                    } catch (SQLException excep) {
                        System.err.print("Error in ACMReferences extraction");
                    }
                }
            } finally {

                if (authorBulkInsert != null) {
                    authorBulkInsert.close();
                }
                if (categoryBulkInsert != null) {
                    categoryBulkInsert.close();
                }
                if (citationBulkInsert != null) {
                    citationBulkInsert.close();
                }
                connection.setAutoCommit(true);
            }

        } catch (SQLException e) {
            // if the error message is "out of memory", 
            // it probably means no database file is found
            System.err.println(e.getMessage());
        } catch (Exception e) {
            System.err.println("File input error");
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

    public static void main(String[] args) throws Exception {
        Class.forName("org.sqlite.JDBC");
        iMixTopicModelExample trainer = new iMixTopicModelExample();

    }
}
