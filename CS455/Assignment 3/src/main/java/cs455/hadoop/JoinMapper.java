package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import java.io.IOException;

public class JoinMapper extends Mapper<Text, Text, Text, Text> {
    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        context.write(key, value);
    }
}