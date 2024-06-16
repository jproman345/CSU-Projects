package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import java.io.IOException;

class QuestionOneMapper extends Mapper<Object, Text, Text, IntWritable> {
    private final static IntWritable one = new IntWritable(1);

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] tokens = value.toString().split("\\|");
        if (tokens.length >= 39) {
            String artistName = tokens[38];
            context.write(new Text(artistName), one);
        }
    }
}

class QuestionOneReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    private Text topArtist;
    private int maxCount = Integer.MIN_VALUE;

    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        int sum = 0;
        for (IntWritable val : values) {
            sum += val.get();
        }
        if (sum > maxCount) {
            maxCount = sum;
            topArtist = new Text(key);
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        if (topArtist != null) {
            context.write(topArtist, new IntWritable(maxCount));
        }
    }
}
