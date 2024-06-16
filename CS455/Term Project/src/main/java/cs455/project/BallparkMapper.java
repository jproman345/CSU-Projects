package cs455.project;

import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;

public class BallparkMapper extends Mapper<LongWritable, Text, Text, Text> {

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        if (key.get() == 0) {
            return;
        }

        String[] fields = value.toString().split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");

        for (int i = 0; i < fields.length; i++) {
            fields[i] = fields[i].replaceAll("^\"|\"$", "");
        }

        String parkID = fields[0];
        String stadiumKey = parkID.substring(0, Math.min(3, parkID.length()));
        context.write(new Text(stadiumKey), value);
    }
}