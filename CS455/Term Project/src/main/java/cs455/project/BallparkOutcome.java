package cs455.project;

import java.io.IOException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

public class BallparkOutcome {
    public static class BallparkOutcomeMapper extends Mapper<LongWritable, Text, Text, Text> {

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            String[] fields = value.toString().split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
    
            for (int i = 0; i < fields.length; i++) {
                fields[i] = fields[i].replaceAll("^\"|\"$", "");
            }
    
            String ballparkName = fields[93];
            String outcome = fields[8];
            String hitDistance = fields[52];
            String launchAngle = fields[54];
            String exitVelocity = fields[53];
    
            if (!hitDistance.equals("NA") && !launchAngle.equals("NA") && !exitVelocity.equals("NA")) {
                context.write(new Text(ballparkName), new Text(outcome + "," + hitDistance + "," + exitVelocity + "," + launchAngle));
            }
        }
    }

    public static class BallparkOutcomeReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int totalHomeRuns = 0;
            int totalSingles = 0;
            int totalDoubles = 0;
            int totalTriples = 0;
            int totalHits = 0;
    
            double totalHitDistance = 0;
            double totalExitVelocity = 0;
            double totalLaunchAngle = 0;
            double totalHRDistance = 0;
    
            for (Text value : values) {
                String[] parts = value.toString().split(",");
                String outcome = parts[0];
                double hitDistance = Double.parseDouble(parts[1]);
                double exitVelocity = Double.parseDouble(parts[2]);
                double launchAngle = Double.parseDouble(parts[3]);

                totalExitVelocity += exitVelocity;
                totalLaunchAngle += launchAngle;
    
                if (outcome.contains("home_run")) {
                    totalHomeRuns++;
                    totalHRDistance += hitDistance;
                } else if (outcome.contains("single")) {
                    totalSingles++;
                } else if (outcome.contains("double")) {
                    totalDoubles++;
                } else if (outcome.contains("triple")) {
                    totalTriples++;
                }
                totalHits++;
                totalHitDistance += hitDistance;
    
            }
    
            double avgHitDistance = totalHitDistance / totalHits;
            double avgExitVelocity = totalExitVelocity / totalHits;
            double avgLaunchAngle = totalLaunchAngle / totalHits;
            double avgHRDistance = totalHRDistance / totalHomeRuns;
    
            context.write(key, new Text(
                ", Home Runs: " + totalHomeRuns +
                ", Singles: " + totalSingles +
                ", Doubles: " + totalDoubles +
                ", Triples: " + totalTriples +
                ", Avg Hit Distance: " + avgHitDistance +
                ", Avg Home Run Distance: " + avgHRDistance +
                ", Avg Exit Velocity: " + avgExitVelocity +
                ", Avg Launch Angle: " + avgLaunchAngle));
        }
    }
}