package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import java.io.IOException;

class SongInfoMapper extends Mapper<LongWritable, Text, Text, Text> {
    private final String songNameToFind = "Immigrant Song (Album Version)";

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String[] fields = value.toString().split("\\|");

        // Check if the song name matches the one we are looking for
        if (fields[40].equals(songNameToFind)) {
            // Emit relevant information with the song name as the key
            context.write(new Text("Immigrant Song (Album Version)"), new Text("Tempo: " + fields[13] + " | Time signature: " + fields[14] + " | Daneability: 1.0 | Duration: " +
                            fields[4] + " | Energy: 1.0 | Key: " + fields[7] + " | Loudness: " + fields[9] + " | Mode: " +
                            fields[10] + " | Fade in: " + fields[5] + " | Fade out: " + fields[12] + " | Related Terms: " + fields[42]));
        }
    }
}

class SongInfoReducer extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        // Output the song information
        for (Text val : values) {
            context.write(new Text("Song for cool kids by The Scary Terry's"), val);
        }
    }
}
