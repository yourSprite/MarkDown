# HBase Java常用API

常用API

| hbase类                                    | 功能                                                         |
| ------------------------------------------ | ------------------------------------------------------------ |
| org.apache.hadoop.hbase.HBaseConfiguration | HBase配置信息。默认的构造方法会尝试从hbase-default.xml和hbase-site.xml中读取配置。 |
| org.apache.hadoop.hbase.client.HBaseAdmin  | 提供了一个接口来管理HBase数据库的表信息。它提供的方法包括：创建表，删除表，列出表项，使表有效或无效，以及添加或删除表列族成员等。 |
| org.apache.hadoop.hbase.HTableDescriptor   | 包含了表的名字极其对应表的列族。                             |
| org.apache.hadoop.hbase.HColumnDescriptor  | 维护着关于列族的信息，例如版本号，压缩设置等。它通常在创建表或者为表添加列族的时候使用。列族被创建后不能直接修改，只能通过删除然后重新创建的方式。列族被删除的时候，列族里面的数据也会同时被删除。 |
| org.apache.hadoop.hbase.client.HTable      | 可以用来和HBase表直接通信。此方法对于更新操作来说是非线程安全的。 |
| org.apache.hadoop.hbase.client.Put         | 用来对单个行执行添加操作。                                   |
| org.apache.hadoop.hbase.client.Get         | 用来获取单个行的相关信息。                                   |

常用操作

```java
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.*;
import org.apache.hadoop.hbase.client.*;

import java.io.IOException;

public class Template {

    public static Configuration configuration; // 配置信息
    public static Connection connection; // 连接信息
    public static Admin admin; // 对数据库进行管理，用于管理对表的创建删除等

    // 建立连接
    public static void init() {
        configuration = HBaseConfiguration.create();
        configuration.set("hbase.rootdir", "hdfs://192.168.199.105:9000/hbase");
        configuration.set("hbase.zookeeper.quorum", "192.168.199.105");
        configuration.set("hbase.zookeeper.property.clientPort", "2181");

        try {
            connection = ConnectionFactory.createConnection(configuration);
            admin = connection.getAdmin();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    // 关闭连接
    public static void close() {
        try {
            if (admin != null) {
                admin.close();
            }
            if (null != connection) {
                connection.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void createTable(String myTableName, String[] colFamily) throws IOException {
        /**
         * @Description: 建表。HBase的表中会有一个系统默认的属性作为主键，主键无需自行创建，默认为put命令操作中表名后第一个数据
         * @param: [myTableName, colFamily] [表名, 列族名]
         * @return: void
         */

        init();
        TableName tableName = TableName.valueOf(myTableName);

        if (admin.tableExists(tableName)) {
            System.out.println("表已存在!");
        } else {
            HTableDescriptor hTableDescriptor = new HTableDescriptor(tableName); // 对表信息进行管理的类
            for (String str : colFamily) {
                HColumnDescriptor hColumnDescriptor = new HColumnDescriptor(str); // 对列族进行管理的类
                hTableDescriptor.addFamily(hColumnDescriptor);
            }
            admin.createTable(hTableDescriptor);
            System.out.println("表" + myTableName + "创建成功");
        }
        close();
    }


    public static void deleteTable(String tableName) throws IOException {
        /**
         * @Description: 删除指定表
         * @param: [tableName] 表名
         * @return: void
         */
        init();
        TableName tn = TableName.valueOf(tableName);
        if (admin.tableExists(tn)) {
            // 先disable再delete
            admin.disableTable(tn);
            admin.deleteTable(tn);
        }
        close();
    }


    public static void listTables() throws IOException {
        /**
         * @Description: 查看已有表
         * @param: []
         * @return: void
         */
        init();
        HTableDescriptor hTableDescriptors[] = admin.listTables();
        for (HTableDescriptor hTableDescriptor : hTableDescriptors) {
            System.out.println(hTableDescriptor.getNameAsString());
        }
        close();
    }


    public static void insertRow(String tableName, String rowKey, String colFamily, String col, String val) throws IOException {
        /**
         * @Description: 向某一行的某一列插入数据
         * @param: [tableName, rowKey, colFamily, col, val] [表名, 行键, 列族名, 列名(没有为空), 值]
         * @return: void
         */
        init();
        Table table = connection.getTable(TableName.valueOf(tableName));
        Put put = new Put(rowKey.getBytes());
        put.addColumn(colFamily.getBytes(), col.getBytes(), val.getBytes());
        table.put(put);
        table.close();
        close();
    }


    public static void deleteRow(String tableName, String rowKey, String colFamily, String col) throws IOException {
        /**
         * @Description: 删除数据
         * @param: [tableName, rowKey, colFamily, col] [表名, 行键, 列族名, 列名]
         * @return: void
         */
        init();
        Table table = connection.getTable(TableName.valueOf(tableName));
        Delete delete = new Delete(rowKey.getBytes());
        // 删除指定列族的所有数据
        // delete.addFamily(colFamily.getBytes());
        // 删除指定列的数据
        // delete.addColumn(colFamily.getBytes(), col.getBytes());

        table.delete(delete);
        table.close();
        close();
        System.out.println("数据已删除");
    }


    public static void getData(String tableName, String rowKey, String colFamily, String col) throws IOException {
        /**
         * @Description: 根据行键rowkey查找数据
         * @param: [tableName, rowKey, colFamily, col] [tableName, rowKey, colFamily, col]
         * @return: void
         */
        init();
        Table table = connection.getTable(TableName.valueOf(tableName));
        Get get = new Get(rowKey.getBytes());
        get.addColumn(colFamily.getBytes(), col.getBytes());
        Result result = table.get(get);
        showCell(result);
        table.close();
        close();
    }


    public static void showCell(Result result) {
        /**
         * @Description: 格式化输出
         * @param: [result]
         * @return: void
         */
        Cell[] cells = result.rawCells();
        for (Cell cell : cells) {
            System.out.println("RowName:" + new String(CellUtil.cloneRow(cell)) + " ");
            System.out.println("Timetamp:" + cell.getTimestamp() + " ");
            System.out.println("column Family:" + new String(CellUtil.cloneFamily(cell)) + " ");
            System.out.println("row Name:" + new String(CellUtil.cloneQualifier(cell)) + " ");
            System.out.println("value:" + new String(CellUtil.cloneValue(cell)) + " ");
        }
    }


    public static void main(String[] args) throws IOException {
        // 创建一个表，表名为Score，列族为sname,course
        createTable("Score", new String[]{"sname", "course"});

        // 在Score表中插入一条数据，其行键为95001,sname为Mary（因为sname列族下没有子列所以第四个参数为空）
        // 等价命令：put 'Score','95001','sname','Mary'
        // insertRow("Score", "95001", "sname", "", "Mary");

        // 在Score表中插入一条数据，其行键为95001,course:Math为88（course为列族，Math为course下的子列）
        // 等价命令：put 'Score','95001','score:Math','88'
        // insertRow("Score", "95001", "course", "Math", "88");

        // 在Score表中插入一条数据，其行键为95001,course:English为85（course为列族，English为course下的子列）
        // 等价命令：put 'Score','95001','score:English','85'
        // insertRow("Score", "95001", "course", "English", "85");

        // 1、删除Score表中指定列数据，其行键为95001,列族为course，列为Math
        // 执行这句代码前请deleteRow方法的定义中，将删除指定列数据的代码取消注释注释，将删除制定列族的代码注释
        // 等价命令：delete 'Score','95001','score:Math'
        // deleteRow("Score", "95001", "course", "Math");

        // 2、删除Score表中指定列族数据，其行键为95001,列族为course（95001的Math和English的值都会被删除）
        // 执行这句代码前请deleteRow方法的定义中，将删除指定列数据的代码注释，将删除制定列族的代码取消注释
        // 等价命令：delete 'Score','95001','score'
        // deleteRow("Score", "95001", "course", "");

        // 3、删除Score表中指定行数据，其行键为95001
        // 执行这句代码前请deleteRow方法的定义中，将删除指定列数据的代码注释，以及将删除制定列族的代码注释
        // 等价命令：deleteall 'Score','95001'
        // deleteRow("Score", "95001", "", "");

        // 查询Score表中，行键为95001，列族为course，列为Math的值
        // getData("Score", "95001", "course", "Math");

        // 查询Score表中，行键为95001，列族为sname的值（因为sname列族下没有子列所以第四个参数为空）
        // getData("Score", "95001", "sname", "");

        // 删除Score表
        // deleteTable("Score");
    }
}

```

