from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()

# Read the CSV data from HDFS
df = spark.read.csv("hdfs://hadoop:9000/input/diabetes_prediction_dataset.csv", header=True, inferSchema=True)

# Data Preprocessing
# Step 1: Drop rows with missing or unknown values in gender
df = df.filter((df.gender == "Female") | (df.gender == "Male"))

# Step 2: Label encoding for gender
gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_encoded")
df = gender_indexer.fit(df).transform(df)

# Step 3: Handle smoking_history
df = df.filter((df.smoking_history.isin(["ever", "never", "former", "not current", "current"])))

# Step 4: One-hot encoding for smoking_history
smoking_indexer = StringIndexer(inputCol="smoking_history", outputCol="smoking_index")
encoder = OneHotEncoder(inputCol="smoking_index", outputCol="smoking_encoded")
df = smoking_indexer.fit(df).transform(df)
df = encoder.transform(df)

# Step 5: Convert bmi to int and normalize it
df = df.withColumn("bmi", df["bmi"].cast("integer"))

# Normalize bmi (assuming it's within the range 0-100)
min_bmi = df.agg({"bmi": "min"}).collect()[0][0]
max_bmi = df.agg({"bmi": "max"}).collect()[0][0]
df = df.withColumn("bmi_normalized", (df["bmi"] - min_bmi) / (max_bmi - min_bmi))

# Step 6: Convert HbA1c_level to int
df = df.withColumn("HbA1c_level", df["HbA1c_level"].cast("integer"))

# Define features for the regression model
feature_cols = ["gender_encoded", "smoking_encoded", "bmi_normalized", "HbA1c_level"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Linear Regression Model
lr = LinearRegression(featuresCol="features", labelCol="diabetes")
model = lr.fit(df)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="diabetes", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(model.transform(df))
print(f"R-squared (R2) Score: {r2}")

# Save the model to HDFS
model.write().overwrite().save("hdfs://hadoop:9000/output/linear_regression_model")

# Stop the Spark session
spark.stop()
