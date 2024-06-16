package cs455.hadoop;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;

import java.io.IOException;
import java.util.HashSet;

class GenericUniqueMapper extends Mapper<Object, Text, Text, Text> {

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] tokens = value.toString().split("\\|");
        if (tokens.length >= 34) {
            String artistName = tokens[38];
            String artistTerms = tokens[42];
            context.write(new Text(artistName), new Text(artistTerms));
        }
    }
}

class GenericUniqueReducer extends Reducer<Text, Text, Text, IntWritable> {
    private Text mostUniqueArtist = new Text();
    private Text mostGenericArtist = new Text();
    private int maxUniqueTerms = 1;
    private int minUniqueTerms = Integer.MAX_VALUE; // Initialize to max possible value

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        HashSet<String> uniqueTerms = new HashSet<>();
        for (Text val : values) {
            String[] terms = val.toString().split(" ");
            for (String term : terms) {
                uniqueTerms.add(term);
            }
        }
        int uniqueTermCount = uniqueTerms.size();
        if (uniqueTermCount > maxUniqueTerms) {
            maxUniqueTerms = uniqueTermCount;
            mostUniqueArtist.set(key);
        }
        if (uniqueTermCount > 0 && uniqueTermCount <= minUniqueTerms) {
            minUniqueTerms = uniqueTermCount;
            mostGenericArtist.set(key);
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        context.write(mostUniqueArtist, new IntWritable(maxUniqueTerms));
        context.write(mostGenericArtist, new IntWritable(minUniqueTerms));
    }
}
