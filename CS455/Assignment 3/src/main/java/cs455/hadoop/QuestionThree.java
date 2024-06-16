package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import java.io.IOException;

class HighestHotttnesssMapper extends Mapper<Object, Text, Text, DoubleWritable> {
    private Text artistID = new Text();
    private DoubleWritable hotttnesss = new DoubleWritable();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] tokens = value.toString().split("\\|");
        if (tokens.length >= 35) {
            String id = tokens[40];
            String hotttnesssValue = tokens[1];
            
            if (!hotttnesssValue.equals("nan")) {
                artistID.set(id);
                hotttnesss.set(Double.parseDouble(hotttnesssValue));
                context.write(artistID, hotttnesss);
            } else {
                // Handle invalid or missing value
            }
        }
    }
}

class HighestHotttnesssReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
    private static final double THRESHOLD = 0.001;
    private static final double MIN_HOTTNESSS = 0.0;
    private static final double MAX_HOTTNESSS = 1.0;
    private double maxHotttnesss = MIN_HOTTNESSS;
    private Text maxSongID = new Text();

    public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
        for (DoubleWritable val : values) {
            double hotttnesssValue = val.get();
            if (Double.isNaN(hotttnesssValue) || hotttnesssValue < THRESHOLD || hotttnesssValue < MIN_HOTTNESSS || hotttnesssValue > MAX_HOTTNESSS) {
                continue;
            }
            if (Math.abs(hotttnesssValue - MAX_HOTTNESSS) < Math.abs(maxHotttnesss - MAX_HOTTNESSS)) {
                maxHotttnesss = hotttnesssValue;
                maxSongID.set(key);
            }
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        if (maxSongID.getLength() > 0) {
            context.write(maxSongID, new DoubleWritable(maxHotttnesss));
        }
    }
}