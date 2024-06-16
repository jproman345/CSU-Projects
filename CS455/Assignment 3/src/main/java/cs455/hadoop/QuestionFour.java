package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import java.io.IOException;

class HighestFadeInMapper extends Mapper<Object, Text, Text, DoubleWritable> {
    private Text artistName = new Text();
    private DoubleWritable totalFadeIn = new DoubleWritable();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] tokens = value.toString().split("\\|");
        if (tokens.length >= 6) {
            String artist = tokens[38];
            double fadeInTime = Double.parseDouble(tokens[5]);
            double fadeOutStartTime = Double.parseDouble(tokens[12]);
            double duration = Double.parseDouble(tokens[4]);
            double fadeOutTime = duration - fadeOutStartTime;
            double totalFadeTime = fadeInTime + fadeOutTime;
            context.write(new Text(artist), new DoubleWritable(totalFadeTime));
        }
    }
}

class HighestFadeInReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
    private double maxTotalFadeIn = Double.MIN_VALUE;
    private Text artistWithMaxFadeIn = new Text();

    public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
        double totalFadeIn = 0;
        for (DoubleWritable val : values) {
            totalFadeIn += val.get();
        }
        if (totalFadeIn > maxTotalFadeIn) {
            maxTotalFadeIn = totalFadeIn;
            artistWithMaxFadeIn.set(key);
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        context.write(artistWithMaxFadeIn, new DoubleWritable(maxTotalFadeIn));
    }
}