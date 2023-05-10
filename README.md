[![Build Status](https://github.com/bioimage-io/pytorch-java-interface-javacpp/actions/workflows/build.yml/badge.svg)](https://github.com/bioimage-io/pytorch-java-interface-javacpp/actions/workflows/build.yml)

# dl-modelrunner-java: pytorch-javacpp

[JAR file](https://maven.scijava.org/service/local/artifact/maven/redirect?r=releases&g=io.bioimage&a=dl-modelrunner-pytorch-javacpp&v=0.2.0&e=jar)


To use with maven:

```
<dependency>
  <groupId>io.bioimage</groupId>
  <artifactId>dl-modelrunner-pytorch-javacpp</artifactId>
  <version>0.2.1-SNAPSHOT</version>
</dependency>
```

and add to `</repositories>` the following:

```
<repository>
  <id>scijava.public</id>
  <url>https://maven.scijava.org/content/groups/public</url>
</repository>
```

In order to have GPU support to the project add the following deps to the pom.xml (regard that this is only for when the code it is executed on its own, not from the modelrunner JDLL):

```
<!-- Additional dependencies required to use CUDA, cuDNN, and NCCL -->
<dependency>
    <groupId>org.bytedeco</groupId>
    <artifactId>pytorch-platform-gpu</artifactId>
    <version>2.0.1-1.5.9-SNAPSHOT</version>
</dependency>

<!-- Additional dependencies to use bundled CUDA, cuDNN, and NCCL -->
<dependency>
    <groupId>org.bytedeco</groupId>
    <artifactId>cuda-platform-redist</artifactId>
    <version>12.1-8.9-1.5.9-SNAPSHOT</version>
</dependency>
```
