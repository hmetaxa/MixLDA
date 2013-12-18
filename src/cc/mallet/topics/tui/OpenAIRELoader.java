/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package cc.mallet.topics.tui;

/**
 *
 * @author hmetaxa
 */

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
//import org.sqlite.JDBC;

public class OpenAIRELoader
{
  public static void main(String[] args) throws ClassNotFoundException
  {
    // load the sqlite-JDBC driver using the current class loader
     // String a = System.getProperty("java.class.path");

     //java.class.path 
     
    Class.forName("org.sqlite.JDBC");

    Connection connection = null;
    try
    {
      // create a database connection
      connection = DriverManager.getConnection("jdbc:sqlite:C:/UoA/OpenAire/ArXiv/fundedarxiv.db");
      Statement statement = connection.createStatement();
      statement.setQueryTimeout(30);  // set timeout to 30 sec.

     // statement.executeUpdate("drop table if exists person");
//      statement.executeUpdate("create table person (id integer, name string)");
//      statement.executeUpdate("insert into person values(1, 'leo')");
//      statement.executeUpdate("insert into person values(2, 'yui')");
//      ResultSet rs = statement.executeQuery("select * from person");
      String sql = "select fundedarxiv.file, fundedarxiv.text, GROUP_CONCAT(funds.grantId,\" \") as fundings from fundedarxiv inner join funds on file=filename Group By fundedarxiv.file, fundedarxiv.text LIMIT 10" ;
    // String sql = "select fundedarxiv.file from fundedarxiv inner join funds on file=filename Group By fundedarxiv.file LIMIT 10" ;
              
      ResultSet rs = statement.executeQuery(sql);
      while(rs.next())
      {
        // read the result set
        System.out.println("name = " + rs.getString("file"));
        System.out.println("name = " + rs.getString("fundings"));
        //System.out.println("id = " + rs.getString("text"));
      }
    }
    catch(SQLException e)
    {
      // if the error message is "out of memory", 
      // it probably means no database file is found
      System.err.println(e.getMessage());
    }
    finally
    {
      try
      {
        if(connection != null)
          connection.close();
      }
      catch(SQLException e)
      {
        // connection close failed.
        System.err.println(e);
      }
    }
  }
}



