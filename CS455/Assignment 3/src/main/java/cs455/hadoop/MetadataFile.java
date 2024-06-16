package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import java.io.IOException;

public class MetadataFile {
    public static class MetadataMapper extends Mapper<LongWritable, Text, Text, Text> {
        private Text songId = new Text();
        private Text metadata = new Text();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split("\\|");
            if (tokens.length >= 9) {
                songId.set(tokens[7]);
                metadata.set(value);
                context.write(songId, metadata);
            }
        }
    }

    public static class MetadataReducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text value : values) {
                context.write(key, value);
            }
        }
    }
}
