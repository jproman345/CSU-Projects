package cs455.project;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import cs455.project.Join.JoinMapper;
import cs455.project.Join.JoinReducer;

public class HomeRunAnalysis {
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: HomeRunAnalysis <hitDataInputPath> <fieldDataInputPath> <outputPath>");
            System.exit(1);
        }

        Configuration conf = new Configuration();

        // Job to process hit dataset and filter only homeruns
        Job homerunDataJob = Job.getInstance(conf, "Homerun Data");
        homerunDataJob.setJarByClass(HomeRunAnalysis.class);
        homerunDataJob.setMapperClass(HomerunData.HomerunMapper.class);
        homerunDataJob.setReducerClass(HomerunData.HomerunReducer.class);
        homerunDataJob.setOutputKeyClass(Text.class);
        homerunDataJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(homerunDataJob, new Path(args[0]));
        Path tempOutputPath1 = new Path(args[2] + "/homerun_data");
        FileOutputFormat.setOutputPath(homerunDataJob, tempOutputPath1);
        boolean homerunDatasetSuccess = homerunDataJob.waitForCompletion(true);

        // Job to process ballpark dataset
        Job ballparkDataJob = Job.getInstance(conf, "Ballpark Data");
        ballparkDataJob.setJarByClass(HomeRunAnalysis.class);
        ballparkDataJob.setMapperClass(BallparkMapper.class);
        ballparkDataJob.setNumReduceTasks(0);
        ballparkDataJob.setOutputKeyClass(Text.class);
        ballparkDataJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(ballparkDataJob, new Path(args[1]));
        Path tempOutputPath2 = new Path(args[2] + "/ballpark_data");
        FileOutputFormat.setOutputPath(ballparkDataJob, tempOutputPath2);
        boolean ballparkDatasetSuccess = ballparkDataJob.waitForCompletion(true);

        // Job to join stadium data and homerun data
        Job joinJob = Job.getInstance(conf, "Join Stadium and Homerun Data");
        joinJob.setJarByClass(HomeRunAnalysis.class);
        joinJob.setMapperClass(Join.JoinMapper.class);
        joinJob.setReducerClass(Join.JoinReducer.class);
        joinJob.setOutputKeyClass(Text.class);
        joinJob.setOutputValueClass(Text.class);
        joinJob.setInputFormatClass(KeyValueTextInputFormat.class); // Use KeyValueTextInputFormat
        FileInputFormat.setInputPaths(joinJob, tempOutputPath1, tempOutputPath2); // Set input paths
        Path finalOutputPath = new Path(args[2] + "/joined_data");
        FileOutputFormat.setOutputPath(joinJob, finalOutputPath);
        boolean joinSuccess = joinJob.waitForCompletion(true);

        //Analysis
        Job analysisJob = Job.getInstance(conf, "Ballpark Outcome Analysis");
        analysisJob.setJarByClass(HomeRunAnalysis.class);
        analysisJob.setMapperClass(BallparkOutcome.BallparkOutcomeMapper.class);
        analysisJob.setReducerClass(BallparkOutcome.BallparkOutcomeReducer.class);
        analysisJob.setOutputKeyClass(Text.class);
        analysisJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(analysisJob, finalOutputPath);
        FileOutputFormat.setOutputPath(analysisJob, new Path(args[2] + "/Analysis"));
        boolean analysisSuccess = analysisJob.waitForCompletion(true);

        System.exit((joinSuccess & analysisSuccess) ? 0 : 1);

    }
}