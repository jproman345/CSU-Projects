import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Main extends App {
  val spark = SparkSession.builder()
    .appName("MovieLens Analysis")
    .getOrCreate()

  import spark.implicits._

  // Setting to show full genre strings in DataFrame show() output
  spark.conf.set("spark.sql.truncate", false)

  // Read datasets
  val moviesDF = spark.read.option("header", "true").csv("/assignment4/movies.csv")
  val ratingsDF = spark.read.option("header", "true").csv("/assignment4/ratings.csv")
  val genomeTagsDF = spark.read.option("header", "true").csv("/assignment4/genome-tags.csv")
  val genomeScoresDF = spark.read.option("header", "true").csv("/assignment4/genome-scores.csv")

  // UDF to extract year from title
  val extractYear = udf((title: String) => "\\((\\d{4})\\)".r.findFirstMatchIn(title).map(_.group(1)).getOrElse(""))

  // Q1: Movies per year
  val withYear = moviesDF.withColumn("year", extractYear($"title"))
  val yearCounts = withYear.filter($"year" =!= "").groupBy("year").count().orderBy("year")
  yearCounts.show(Int.MaxValue)

  // Q2: Average number of genres per movie
  val genreCount = udf((genres: String) => if (genres == "\\N") 0 else genres.split("\\|").length)
  val avgGenres = moviesDF.withColumn("genreCount", genreCount($"genres")).agg(avg("genreCount"))
  avgGenres.show()

  // Q3: Rank genres by average rating
  val genresExploded = moviesDF.withColumn("genre", explode(split($"genres", "\\|")))
  val genresRatings = genresExploded.join(ratingsDF, "movieId")
    .groupBy(lower($"genre").as("genre"))
    .agg(avg("rating").alias("avg_rating"))
    .orderBy(desc("avg_rating"))
  genresRatings.show()

  // Q4: Top-3 genre combinations by average rating
  val genreCombinationRatings = moviesDF.join(ratingsDF, "movieId")
    .groupBy("genres")
    .agg(avg("rating").alias("avg_rating"))
    .orderBy(desc("avg_rating"))
    .limit(3)
  genreCombinationRatings.show(false)

  // Q5: Movies tagged as "comedy"
  val comedyCount = genresExploded.filter(lower($"genre") === "comedy").count()
  println(s"Number of movies tagged as comedy: $comedyCount")

  // Q6: Number of movies per genre
  val movieGenreCounts = genresExploded.groupBy("genre").count().orderBy(desc("count"))
  movieGenreCounts.show()

  // Q7: Effect of tags on ratings
  val tagRatings = genomeScoresDF
    .filter($"relevance" > 0.5)
    .join(genomeTagsDF, "tagId")
    .join(moviesDF, "movieId")
    .join(ratingsDF, "movieId")
    .groupBy("tag")
    .agg(avg("rating").alias("avg_rating"))
    .orderBy(desc("avg_rating"))

  tagRatings.show(50, false)

  spark.stop()
}
