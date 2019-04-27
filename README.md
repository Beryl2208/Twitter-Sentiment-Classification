# Twitter-Sentiment-Classification
Twitter Sentiment Classification using pipelines
Created a Scala class, as a part of a Scala SBT project with all the dependencies needed to be run on AWS. Compiled and packaged the code into a jar.
Input file and jar uploaded to AWS S3 and run on EMR instance.
 
After creating a cluster, we add steps and run jar as a spark application by passing the arguments namely the tweets.csv and the jar-path as in the s3 bucket.
Successfully created output files in output folder in AWS S3 bucket after running the jar.
 






