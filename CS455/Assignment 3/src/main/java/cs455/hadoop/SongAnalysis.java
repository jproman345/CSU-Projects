package cs455.hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;

public class SongAnalysis {
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: SongAnalysis <analysisInputPath> <metadataInputPath> <outputPath>");
            System.exit(1);
        }

        Configuration conf = new Configuration();

        // Job to process analysis file
        Job analysisJob = Job.getInstance(conf, "Analysis Processing");
        analysisJob.setJarByClass(SongAnalysis.class);
        analysisJob.setMapperClass(AnalysisFile.AnalysisMapper.class);
        analysisJob.setReducerClass(AnalysisFile.AnalysisReducer.class);
        analysisJob.setOutputKeyClass(Text.class);
        analysisJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(analysisJob, new Path(args[0]));
        Path analysisOutputPath = new Path(args[2] + "/analysis_output");
        FileOutputFormat.setOutputPath(analysisJob, analysisOutputPath);
        boolean analysisSuccess = analysisJob.waitForCompletion(true);

        // Job to process metadata file
        Job metadataJob = Job.getInstance(conf, "Metadata Processing");
        metadataJob.setJarByClass(SongAnalysis.class);
        metadataJob.setMapperClass(MetadataFile.MetadataMapper.class);
        metadataJob.setReducerClass(MetadataFile.MetadataReducer.class);
        metadataJob.setOutputKeyClass(Text.class);
        metadataJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(metadataJob, new Path(args[1]));
        Path metadataOutputPath = new Path(args[2] + "/metadata_output");
        FileOutputFormat.setOutputPath(metadataJob, metadataOutputPath);
        boolean metadataSuccess = metadataJob.waitForCompletion(true);

        // Job to join analysis and metadata files
        Job joinJob = Job.getInstance(conf, "Join Analysis and Metadata");
        joinJob.setJarByClass(SongAnalysis.class);
        joinJob.setMapperClass(JoinMapper.class);
        joinJob.setReducerClass(JoinReducer.class);
        joinJob.setOutputKeyClass(Text.class);
        joinJob.setOutputValueClass(Text.class);
        joinJob.setInputFormatClass(KeyValueTextInputFormat.class);
        FileInputFormat.addInputPath(joinJob, analysisOutputPath);
        FileInputFormat.addInputPath(joinJob, metadataOutputPath);
        Path joinedOutputPath = new Path(args[2] + "/joined_output");
        FileOutputFormat.setOutputPath(joinJob, joinedOutputPath);
        boolean joinSuccess = joinJob.waitForCompletion(true);
        
        // Job for question one
        Job questionOneJob = Job.getInstance(conf, "Question One");
        questionOneJob.setJarByClass(SongAnalysis.class);
        questionOneJob.setMapperClass(QuestionOneMapper.class);
        questionOneJob.setCombinerClass(QuestionOneReducer.class);
        questionOneJob.setReducerClass(QuestionOneReducer.class);
        questionOneJob.setOutputKeyClass(Text.class);
        questionOneJob.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(questionOneJob, joinedOutputPath);
        FileOutputFormat.setOutputPath(questionOneJob, new Path(args[2] + "/question_one"));
        boolean questionOneSuccess = questionOneJob.waitForCompletion(true);
        
        // Job for question two
        Job questionTwoJob = Job.getInstance(conf, "Question Two");
        questionTwoJob.setJarByClass(SongAnalysis.class);
        questionTwoJob.setMapperClass(QuestionTwoMapper.class);
        questionTwoJob.setReducerClass(QuestionTwoReducer.class);
        questionTwoJob.setOutputKeyClass(Text.class);
        questionTwoJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(questionTwoJob, joinedOutputPath);
        FileOutputFormat.setOutputPath(questionTwoJob, new Path(args[2] + "/question_two"));
        boolean questionTwoSuccess = questionTwoJob.waitForCompletion(true);
        

        // Job for question three
        Job questionThreeJob = Job.getInstance(conf, "Question Three");
        questionThreeJob.setJarByClass(SongAnalysis.class);
        questionThreeJob.setMapperClass(HighestHotttnesssMapper.class);
        questionThreeJob.setReducerClass(HighestHotttnesssReducer.class);
        questionThreeJob.setOutputKeyClass(Text.class);
        questionThreeJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(questionThreeJob, joinedOutputPath);
        FileOutputFormat.setOutputPath(questionThreeJob, new Path(args[2] + "/question_three"));
        boolean questionThreeSuccess = questionThreeJob.waitForCompletion(true);

        // Job for question four
        Job questionFourJob = Job.getInstance(conf, "Question Four");
        questionFourJob.setJarByClass(SongAnalysis.class);
        questionFourJob.setMapperClass(HighestFadeInMapper.class);
        questionFourJob.setReducerClass(HighestFadeInReducer.class);
        questionFourJob.setOutputKeyClass(Text.class);
        questionFourJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(questionFourJob, joinedOutputPath);
        FileOutputFormat.setOutputPath(questionFourJob, new Path(args[2] + "/question_four"));
        boolean questionFourSuccess = questionFourJob.waitForCompletion(true);
        

        // Job for question five
        Job questionFiveJob = Job.getInstance(conf, "Question Five");
        questionFiveJob.setJarByClass(SongAnalysis.class);
        questionFiveJob.setMapperClass(SongLengthMapper.class);
        questionFiveJob.setReducerClass(SongLengthReducer.class);
        questionFiveJob.setOutputKeyClass(Text.class);
        questionFiveJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(questionFiveJob, joinedOutputPath);
        FileOutputFormat.setOutputPath(questionFiveJob, new Path(args[2] + "/question_five"));
        boolean questionFiveSuccess = questionFiveJob.waitForCompletion(true);
        
        // Job for question eight
        Job questionEightJob = Job.getInstance(conf, "Question Eight");
        questionEightJob.setJarByClass(SongAnalysis.class);
        questionEightJob.setMapperClass(GenericUniqueMapper.class);
        questionEightJob.setReducerClass(GenericUniqueReducer.class);
        questionEightJob.setOutputKeyClass(Text.class);
        questionEightJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(questionEightJob, joinedOutputPath);
        FileOutputFormat.setOutputPath(questionEightJob, new Path(args[2] + "/question_eight"));
        boolean questionEightSuccess = questionEightJob.waitForCompletion(true);

        // Job for question nine
        Job questionNineJob = Job.getInstance(conf, "Question Nine");
        questionNineJob.setJarByClass(SongAnalysis.class);
        questionNineJob.setMapperClass(SongInfoMapper.class);
        questionNineJob.setReducerClass(SongInfoReducer.class);
        questionNineJob.setOutputKeyClass(Text.class);
        questionNineJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(questionNineJob, joinedOutputPath); // Input path should be the joined dataset
        FileOutputFormat.setOutputPath(questionNineJob, new Path(args[2] + "/question_nine"));
        boolean questionNineSuccess = questionNineJob.waitForCompletion(true);
        

        // Job for question ten
        Job temporalTrendsJob = Job.getInstance(conf, "Temporal Trends");
        temporalTrendsJob.setJarByClass(SongAnalysis.class);
        temporalTrendsJob.setMapperClass(QuestionTen.TemporalTrendsMapper.class);
        temporalTrendsJob.setReducerClass(QuestionTen.TemporalTrendsReducer.class);
        temporalTrendsJob.setOutputKeyClass(IntWritable.class);
        temporalTrendsJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(temporalTrendsJob, joinedOutputPath);
        FileOutputFormat.setOutputPath(temporalTrendsJob, new Path(args[2] + "/question_ten"));
        boolean temporalTrendsSuccess = temporalTrendsJob.waitForCompletion(true);

        // Exit with success or failure based on job execution
        System.exit(temporalTrendsSuccess && questionNineSuccess && questionEightSuccess ? 0 : 1);
    }
}