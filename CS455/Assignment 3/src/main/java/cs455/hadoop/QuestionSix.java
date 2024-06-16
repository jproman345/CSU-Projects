package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import java.io.IOException;
import java.util.TreeMap;

class EnergeticDanceableMapper extends Mapper<Object, Text, DoubleWritable, Text> {
    private Text songName = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] tokens = value.toString().split("\\|");
        if (tokens.length >= 4) {
            String name = tokens[40];
            double energyValue = Double.parseDouble(tokens[6]);
            double danceabilityValue = Double.parseDouble(tokens[3]);

            if (energyValue > 0.0 || danceabilityValue > 0.0) {
                double score;
                if (energyValue <= 0.0 || Double.isNaN(energyValue)) {
                    score = danceabilityValue;
                    System.out.println(score);
                } else if (danceabilityValue <= 0.0 || Double.isNaN(danceabilityValue)) {
                    score = energyValue;
                    System.out.println(score);
                } else {
                    score = energyValue * danceabilityValue;
                    System.out.println(score);
                }

                if (score >= 0.0) {
                    songName.set(name);
                    context.write(new DoubleWritable(score), songName);
                }
            }
        }
    }
}

class EnergeticDanceableReducer extends Reducer<DoubleWritable, Text, Text, DoubleWritable> {
    private TreeMap<Double, Text> topSongs = new TreeMap<>();
    private int count = 0;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        // Do nothing in setup for this case
    }

    @Override
    public void reduce(DoubleWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        for (Text val : values) {
            String song = val.toString();
            topSongs.put(key.get(), new Text(song));
            count++;

            if (count > 10) {
                topSongs.remove(topSongs.firstKey()); // Remove the lowest score
            }
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for (double score : topSongs.descendingKeySet()) {
            Text song = topSongs.get(score);
            context.write(song, new DoubleWritable(score));
        }
    }
}
