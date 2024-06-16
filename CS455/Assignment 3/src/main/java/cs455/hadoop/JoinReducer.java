package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import java.io.IOException;

public class JoinReducer extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        String analysisInfo = null;
        String metadataInfo = null;

        for (Text val : values) {
            String[] tokens = val.toString().split("\\|");
            if (tokens.length > 1) {
                if (tokens.length == 32) {
                    analysisInfo = val.toString();
                } else if (tokens.length == 14) {
                    metadataInfo = val.toString();
                }
            }
        }
        if (analysisInfo != null && metadataInfo != null) {
            context.write(key, new Text(analysisInfo + "|" + metadataInfo));
        }
    }
}
