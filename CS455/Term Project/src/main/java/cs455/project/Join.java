package cs455.project;

import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import java.util.*;

public class Join {

    public static class JoinMapper extends Mapper<Text, Text, Text, Text> {

        @Override
        protected void map(Text key, Text value, Context context)
                throws IOException, InterruptedException {
            context.write(key, value);
        }
    }

    public static class JoinReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            List<String> stadiumData = new ArrayList<>();
            List<String> hitData = new ArrayList<>();

            for (Text value : values) {
                String[] parts = value.toString().split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
                if (parts.length <= 29) {
                    stadiumData.add(value.toString());
                } else {
                    hitData.add(value.toString());
                }
            }

            for (String hit : hitData) {
                for (String stadium : stadiumData) {
                    String[] parts = stadium.toString().split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
                    String parkName = parts[1];
                    context.write(new Text(parkName), new Text(hit + "," + stadium));
                }
            }
        }
    }
}
