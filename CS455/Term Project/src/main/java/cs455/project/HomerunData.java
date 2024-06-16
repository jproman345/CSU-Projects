package cs455.project;

import java.io.IOException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

public class HomerunData {

    public static class HomerunMapper extends Mapper<LongWritable, Text, Text, Text> {

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
    
            if (key.get() == 0) {
                return;
            }

            String[] parts = value.toString().split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");

            for (int i = 0; i < parts.length; i++) {
                parts[i] = parts[i].replaceAll("^\"|\"$", "");
            }

            String homeTeam = parts[19];

            if (homeTeam.length() < 3) {
                homeTeam = String.format("%-3s", homeTeam).replace(' ', '0');
            }

            context.write(new Text(homeTeam), value);
            
        }
    }

    public static class HomerunReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            
            for (Text value : values) {
                context.write(key, value);
            }
        }
    }
}
