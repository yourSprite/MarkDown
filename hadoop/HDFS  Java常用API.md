# HDFS  Java常用API

常用API

| hadoop类                                     | 功能                                                         |
| -------------------------------------------- | :----------------------------------------------------------- |
| org.apache.hadoop.fs.FileSystem              | 一个通用文件系统的抽象基类，可以被分布式文件系统继承。所有的可能使用Hadoop文件系统的代码都要使用到这个类 |
| org.apache.hadoop.fs.FileStatus              | 客户端可见的文件状态信息                                     |
| org.apache.hadoop.fs.FSDataInputStream       | 文件输入流，用于读取Hadoop文件                               |
| org.apache.hadoop.fs.FSDataOutputStream      | 文件输出流，用于写Hadoop文件                                 |
| org.apache.hadoop.fs.permission.FsPermission | 文件或者目录的权限                                           |
| org.apache.hadoop.conf.Configuration         | 访问配置项。所有的配置项的值，如果没有专门配置，以core-default.xml为准；否则，以core-site.xml中的配置为准 |

> 备注：创建一个Configuration对象时，其构造方法会默认加载工程项目下两个配置文件，分别是hdfs-site.xml以及core-site.xml，这两个文件中会有访问HDFS所需的参数值，主要是fs.defaultFS，指定了HDFS的地址(比如hdfs://localhost:9000)，有了这个地址客户端就可以通过这个地址访问HDFS了。

常用操作

```java
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;

import java.io.*;
import java.net.URI;

public class Template {

    private static String HDFSUri = "hdfs://localhost:9000";

    public static FileSystem getFileSystem() {
        /**
         * @Description: 获取文件系统
         * @param: []
         * @return: org.apache.hadoop.fs.FileSystem 文件系统
         */

        // 读取配置文件
        Configuration conf = new Configuration();
        // 文件系统
        FileSystem fs = null;
        String hdfsUri = HDFSUri;

        if (StringUtils.isBlank(hdfsUri)) {
            // 返回默认文件系统，如果在hadoop集群下运行，使用此种方法可直接获取默认文件系统；
            try {
                fs = FileSystem.get(conf);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            // 返回指定的文件系统，如果在本地测试，需要此种方法获取文件系统；
            try {
                URI uri = new URI(hdfsUri.trim());
                fs = FileSystem.get(uri, conf);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        return fs;

    }


    public static void mkdir(String path) {
        /**
         * @Description: 创建文件目录
         * @param: [path]要创建的文件路径
         * @return: void
         */

        try {
            FileSystem fs = getFileSystem();
            // 创建目录
            fs.mkdirs(new Path(path));
            System.out.println(HDFSUri+path + "已创建");
            // 释放资源
            fs.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void rmdir(String path) {
        /**
         * @Description: 删除文件或者文件目录
         * @param: [path]删除文件
         * @return: void
         */

        try {
            // 返回FileSystem对象
            FileSystem fs = getFileSystem();

            String hdfsUri = HDFSUri;
            if (StringUtils.isNotBlank(hdfsUri)) {
                path = hdfsUri + path;
            }

            // 删除文件或者文件目录 delete(Path f)此方法已经弃用
            if (fs.delete(new Path(path), true)) {
                System.out.println(path+"已删除");
            } else {
                System.out.println("删除失败");
            }

            fs.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    public static boolean existDir(String filePath) {
        /**
         * @Description: 判断目录是否存在
         * @param: [filePath]目录路径
         * @return: boolean
         */

        boolean flag = false;

        // 传入路径为空直接返回
        if (StringUtils.isEmpty(filePath)) {
            return flag;
        }

        try {
            Path path = new Path(filePath);
            // FileSystem对象
            FileSystem fs = getFileSystem();

            if (fs.exists(path)) {
                flag = true;
            }

        } catch (Exception e) {
            e.printStackTrace();

        }

        return flag;

    }


    public static void uploadFileToHDFS(String srcFile, String destPath) throws Exception {
        /**
         * @Description: 本地文件上传至HDFS
         * @param: [srcFile, destPath]源文件路径,目的文件路径
         * @return: void
         */

        FileInputStream fis = new FileInputStream(new File(srcFile)); //读取本地文件
        Configuration config = new Configuration();
        FileSystem fs = FileSystem.get(URI.create(HDFSUri + destPath), config);
        OutputStream os = fs.create(new Path(destPath));
        // 上传本地文件
        IOUtils.copyBytes(fis, os, 4096, true);
        System.out.println(srcFile+"已上传至"+HDFSUri+destPath);
        fs.close();
    }


    public static void downloadFile(String srcFile, String destPath) throws Exception {
        /**
         * @Description: 从HDFS下载文件到本地
         * @param: [srcFile, destPath]源文件路径,目的文件路径
         * @return: void
         */

        // HDFS文件地址
        String file = HDFSUri + srcFile;
        Configuration config = new Configuration();
        // 构建filesystem
        FileSystem fs = FileSystem.get(URI.create(file), config);
        // 读取文件
        InputStream is = fs.open(new Path(file));
        IOUtils.copyBytes(is, new FileOutputStream(new File(destPath)), 2048, true);
        System.out.println(HDFSUri+srcFile+"已下载到"+destPath);
        fs.close();
    }


    public static void readFile(String filePath) throws IOException {
        /**
         * @Description: 读取文件的内容
         * @param: [filePath]读取文件路径
         * @return: void
         */

        Configuration config = new Configuration();
        String file = HDFSUri + filePath;
        FileSystem fs = FileSystem.get(URI.create(file), config);
        // 读取文件
        InputStream is = fs.open(new Path(file));
        // 读取文件
        IOUtils.copyBytes(is, System.out, 2048, false); //复制到标准输出流
        fs.close();
    }


    public static void main(String[] args) throws Exception {
        /**
         * @Description: 主方法测试
         * @param: [args]
         * @return: void
         */

        // 连接fs
        FileSystem fs = getFileSystem();
        System.out.println("已连接到HDFS");

        // 创建路径
        System.out.println("创建路径测试");
        mkdir("/dit2");

        // 验证是否存在
        if (existDir("/dit2")) {
            System.out.println("路径存在");
        } else {
            System.out.println("路径不存在");
        }

        // 删除路径
        rmdir("/dit2");

        // 验证是否存在
        if (existDir("/dit2")) {
            System.out.println("路径存在");
        } else {
            System.out.println("路径不存在");
        }

        // 上传文件到HDFS
        uploadFileToHDFS("/usr/local/hadoop/myLocalFile.txt", "/dit/myLocalFile.txt");
        // 下载文件到本地
        downloadFile("/dit/myLocalFile.txt", "/usr/local/hadoop/myLocalFile2.txt");

        // 读取文件
        readFile("/dit/myLocalFile.txt");

    }

}
```