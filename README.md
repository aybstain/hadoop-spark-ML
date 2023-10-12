# Setting up a Hadoop and PySpark Development Environment in Docker

This guide will walk you through setting up a Docker-based development environment for Hadoop and PySpark, including the process of getting data into HDFS and executing PySpark scripts. The environment is self-contained and easy to set up.

## Directory Structure

```plaintext
project_directory/
│
├── hadoop/
│   ├── Dockerfile
│   ├── core-site.xml
│   └── hdfs-site.xml
│
├── pyspark/
│   ├── Dockerfile
│
├── data/
│   ├── your_local_csv_file.csv
│
└── docker-compose.yml
```

- `project_directory`: Your project directory.
- `hadoop`: Contains the Dockerfile for the Hadoop container and Hadoop configuration files.
- `pyspark`: Contains the Dockerfile for the PySpark container.
- `data`: Place your local CSV file in this directory.
- `docker-compose.yml`: Defines the services and dependencies for Docker Compose.

## Steps

1. **Building the Docker Images**

   - **Hadoop Docker Image**

     - Navigate to the `hadoop` directory.

     - Modify the `Dockerfile` to install Hadoop and configure it (e.g., specify Hadoop version).

     - Build the Docker image:

       ```bash
       docker build -t my-hadoop-image .
       ```

   - **PySpark Docker Image**

     - Navigate to the `pyspark` directory.

     - Modify the `Dockerfile` to install Spark and configure it (e.g., specify Spark version).

     - Build the Docker image:

       ```bash
       docker build -t my-pyspark-image .
       ```

2. **Running the Docker Containers**

   - In your project directory, use Docker Compose to start the containers:

     ```bash
     docker-compose up
     ```

3. **Accessing the PySpark Container**

   - Get a shell inside the PySpark container:

     ```bash
     docker exec -it project_directory_pyspark_1 /bin/bash
     ```

4. **Getting Data into HDFS**

   - Inside the PySpark container, start HDFS (if not already started):

     ```bash
     start-dfs.sh
     ```

   - Create an HDFS directory (if it doesn't exist):

     ```bash
     hdfs dfs -mkdir /input
     ```

   - Upload your CSV data to HDFS:

     ```bash
     hdfs dfs -put /path/to/your_local_csv_file.csv /input/
     ```

5. **Running Your PySpark Script**

   - Inside the PySpark container, navigate to the directory where your PySpark script is located (e.g., `/opt`):

     ```bash
     cd /opt
     ```

   - Execute your PySpark script:

     ```bash
     spark-submit your_pyspark_script.py
     ```

6. **Monitoring Output and Results**

   - Monitor the output and results as your PySpark script runs.

   - Examine logs and output for any issues.

7. **Cleanup**

   - Exit the PySpark container:

     ```bash
     exit
     ```

   - Stop and remove the Docker containers:

     ```bash
     docker-compose down
     ```

Your Docker-based Hadoop and PySpark development environment is now set up and ready for use.

## Understanding the PySpark Code

Below is a detailed explanation of the PySpark code:

```python
# Import necessary PySpark libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a SparkSession with an application name
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
model = lr fit(df)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="diabetes", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(model.transform(df))
print(f"R-squared (R2) Score: {r2}")

# Save the model to HDFS
model.write().overwrite().save("hdfs://hadoop:9000/output/linear_regression_model")

# Stop the Spark session
spark stop()
```

**Where to Find the Model:**

The trained linear regression model is saved in HDFS at the specified path: "hdfs://hadoop:9000/output/linear_regression_model". You can access the model in HDFS using Hadoop's HDFS commands or by reading it back into a PySpark application for further use. You can typically find the model's files and metadata within the specified output directory in HDFS.

Your Docker-based Hadoop and PySpark development environment is now set up and ready for use.

