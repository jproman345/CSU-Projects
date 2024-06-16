package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import java.io.IOException;

public class QuestionTen {

    public static class TemporalTrendsMapper extends Mapper<Object, Text, IntWritable, Text> {

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split("\\|");
            if (tokens.length >= 35) {
                int year = Integer.parseInt(tokens[45]);
                if (year > 0) {
                    // Extract relevant musical characteristics: tempo, energy, and loudness
                    double tempo = Double.parseDouble(tokens[13]);
                    double loudness = Double.parseDouble(tokens[9]);

                    // Emit year as key and musical characteristics as value
                    context.write(new IntWritable(year), new Text(tempo + "|" + loudness));
                }
            }
        }
    }

    public static class TemporalTrendsReducer extends Reducer<IntWritable, Text, IntWritable, Text> {

        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double sumTempo = 0.0;
            double sumLoudness = 0.0;
            int count = 0;

            // Iterate through the values and aggregate the musical characteristics for each year
            for (Text val : values) {
                String[] parts = val.toString().split("\\|");
                if (parts.length == 2) {
                    sumTempo += Double.parseDouble(parts[0]);
                    sumLoudness += Double.parseDouble(parts[1]);
                    count++;
                }
            }

            // Calculate the average tempo, energy, and loudness for the year
            double avgTempo = sumTempo / count;
            double avgLoudness = sumLoudness / count;

            // Emit the year and average musical characteristics
            context.write(key, new Text("Avg Tempo: " + avgTempo + " | Avg Loudness: " + avgLoudness));
        }
    }
}