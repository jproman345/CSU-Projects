package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import java.io.IOException;
import java.util.TreeMap;

class QuestionTwoMapper extends Mapper<Object, Text, Text, DoubleWritable> {
    private Text artistName = new Text();
    private DoubleWritable loudness = new DoubleWritable();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] tokens = value.toString().split("\\|");
        if (tokens.length >= 39) {
            String artistName = tokens[38];
            double loudness = Double.parseDouble(tokens[9]);
            context.write(new Text(artistName), new DoubleWritable(loudness));
        }
    }
}

class QuestionTwoReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
    private TreeMap<Double, Text> topArtists = new TreeMap<>();

    public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
        double sum = 0;
        int count = 0;
        for (DoubleWritable val : values) {
            sum += val.get();
            count++;
        }
        double averageLoudness = sum / count;

        topArtists.put(-averageLoudness, new Text(key));
        
        if (topArtists.size() > 5) {
            topArtists.remove(topArtists.lastKey());
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for (Double avgLoudness : topArtists.keySet()) {
            Text artist = topArtists.get(avgLoudness);
            context.write(artist, new DoubleWritable(-avgLoudness));
        }
    }
}
