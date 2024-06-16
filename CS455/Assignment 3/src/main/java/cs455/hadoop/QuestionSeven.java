package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import java.io.IOException;

class SegmentDataMapper extends Mapper<LongWritable, Text, Text, Text> {

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String[] fields = value.toString().split("\\|");
        // Assuming the input format is pipeline-separated

        // Remove square brackets from lists of values
        for (int i = 0; i < fields.length; i++) {
            fields[i] = fields[i].replaceAll("\\[|\\]", "");
        }

        String startTime = fields[17];
        String pitch = fields[19];
        String timbre = fields[20];
        String maxLoudness = fields[21];
        String maxLoudnessTime = fields[22];
        String startLoudness = fields[23];

        // Emitting segment features with "Overall" as key
        context.write(new Text("StartTime"), new Text(startTime));
        context.write(new Text("Pitch"), new Text(pitch));
        context.write(new Text("Timbre"), new Text(timbre));
        context.write(new Text("MaxLoudness"), new Text(maxLoudness));
        context.write(new Text("MaxLoudnessTime"), new Text(maxLoudnessTime));
        context.write(new Text("StartLoudness"), new Text(startLoudness));
    }
}

class SegmentDataReducer extends Reducer<Text, Text, Text, Text> {

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        double sum = 0.0;
        int count = 0;

        // Accumulating sums for the segment feature
        for (Text val : values) {
            sum += Double.parseDouble(val.toString());
            count++;
        }

        // Calculating average value
        double average = sum / count;

        // Emitting the average value for the segment feature
        context.write(key, new Text("Average: " + average));
    }
}