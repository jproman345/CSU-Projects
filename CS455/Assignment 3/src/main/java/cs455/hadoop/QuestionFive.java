package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import java.io.IOException;
import java.util.TreeMap;

class SongLengthMapper extends Mapper<Object, Text, Text, DoubleWritable> {
    private Text songName = new Text();
    private DoubleWritable duration = new DoubleWritable();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] tokens = value.toString().split("\\|");
        if (tokens.length >= 2) {
            String name = tokens[40];
            double durationValue = Double.parseDouble(tokens[4]);
            songName.set(name);
            duration.set(durationValue);
            context.write(songName, duration);
        }
    }
}

class SongLengthReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
    private TreeMap<Double, Text> songLengths = new TreeMap<>();

    public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
        double totalDuration = 0;
        int count = 0;
        for (DoubleWritable val : values) {
            totalDuration += val.get();
            count++;
        }
        double averageDuration = totalDuration / count;

        songLengths.put(averageDuration, new Text(key));
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        int numSongs = songLengths.size();
        int index = 0;
        double medianDuration = 0;
        double closestDuration = Double.MAX_VALUE;
        Text closestSong = new Text();

        for (double duration : songLengths.keySet()) {
            if (index == 0) {
                context.write(songLengths.get(duration), new DoubleWritable(duration));
            }
            if (index == numSongs / 2) {
                medianDuration = duration;
            }
            if (index == numSongs - 1) {
                context.write(songLengths.get(duration), new DoubleWritable(duration));
            }

            double diff = Math.abs(duration - medianDuration);
            if (diff < closestDuration) {
                closestDuration = diff;
                closestSong = songLengths.get(duration);
            }
            index++;
        }
        context.write(closestSong, new DoubleWritable(medianDuration));
    }
}