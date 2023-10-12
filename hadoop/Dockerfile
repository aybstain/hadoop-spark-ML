# Use a base image (e.g., Ubuntu) for your Docker image
FROM ubuntu:20.04

# Install dependencies (Java and SSH)
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk openssh-server && \
    apt-get clean

# Set environment variables for Java
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Add the PATH for Hadoop
ENV PATH=$PATH:/opt/hadoop-3.3.1/bin

# Download and configure Hadoop (adjust version as needed)
WORKDIR /opt
ADD https://downloads.apache.org/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz /opt/
RUN tar -xzvf hadoop-3.3.1.tar.gz && \
    rm hadoop-3.3.1.tar.gz && \
    ln -s /opt/hadoop-3.3.1 /opt/hadoop

# Set environment variables for Hadoop
ENV HADOOP_HOME /opt/hadoop
ENV PATH $HADOOP_HOME/bin:$PATH

# Start SSH service (required for Hadoop)
RUN service ssh start

# Set JAVA_HOME for Hadoop
RUN echo "export JAVA_HOME=$JAVA_HOME" >> $HADOOP_HOME/etc/hadoop/hadoop-env.sh

# Expose ports for Hadoop services (adjust as needed)
EXPOSE 9870 8088 8020 9000

# Copy core-site.xml and hdfs-site.xml into the container
COPY core-site.xml $HADOOP_HOME/etc/hadoop/
COPY hdfs-site.xml $HADOOP_HOME/etc/hadoop/

# Add a Hadoop user
RUN useradd -ms /bin/bash hadoop
RUN echo 'hadoop:hadoop' | chpasswd
RUN usermod -aG sudo hadoop

# Entry point to keep the container running (you can customize this)
CMD ["tail", "-f", "/dev/null"]
