/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.madgik.dbpediaspotlightclient;

import java.util.List;
import java.util.Set;

/**
 *
 * @author omiros
 */
public class DBpediaResource {

    private DBpediaResourceType type;
    private String uri;
    private int support;
    private String mention;
    private String title;
    private double similarity;
    private double confidence;
    private String wikiAbstract;
    private String wikiId;
    private Set<String> categories;

    public DBpediaResource(DBpediaResourceType type, String URI, String title, int support,  double Similarity, double confidence, String mention, Set<String> categories, String wikiAbstract, String wikiId) {
        this.uri = URI;
        this.support = support;
        this.type = type;
        this.mention = mention;
        this.similarity = Similarity;
        this.confidence = confidence;
        this.title = title;
        this.categories = categories;
        this.wikiAbstract = wikiAbstract;
        this.wikiId = wikiId;
    }

     public Set<String> getCategories() {
        return categories;
    }

    public void setCategories(Set<String> categories) {
        this.categories = categories;
    }
    
    public String getURI() {
        return uri;
    }

    public void setURI(String URI) {
        this.uri = URI;
    }

      public String getWikiId() {
        return wikiId;
    }

    public void setWikiId(String wikiId) {
        this.wikiId = wikiId;
    }
    
 public String getWikiAbstract() {
        return wikiAbstract;
    }

    public void setWikiAbstract(String wikiAbstract) {
        this.wikiAbstract = wikiAbstract;
    }

    
    public void setSimilarity(Double Similarity) {
        this.similarity = Similarity;
    }

    public double getSimilarity() {
        return similarity;
    }
    
     public void setConfidence(Double Confidence) {
        this.confidence = Confidence;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setMention(String Mention) {
        this.mention = Mention;
    }

    public String getMention() {
        return mention;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getTitle() {
        return title;
    }

    
    public void setType(DBpediaResourceType Type) {
        this.type = Type;
    }

    public DBpediaResourceType getType() {
        return type;
    }

    public int getSupport() {
        return support;
    }

    public void setSupport(int Support) {
        this.support = Support;
    }
}
