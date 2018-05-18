Apache Falcon:
* fault-tolerant feed processing and management
* notifications
* integration with metastore

SnappyData:
* in-memory analytics for OLAP
* integrate Spark and GemiFire XD (in-memory transactional store)

H-Rider:
* view and manipulate HBase data

Apache Arrow
* columnar in-memory
* process and move data fast
* use SIMD (vectorized) processor operations
* single arrow memory as mediator between parquet, pandas, etc.

Apache Tika
* content metadata analysis (?)

= Apache NiFi

* from "NiagraFiles"
* Dataflow management
* need format, schema, protocol, ontology
* challenges after Kafka:
  * difficult when many independent parties involved
  * authorization, schema, interest, prioritization
  * flow control
* features:
  * guaranteed delivery
  * buffering
  * prios
  * QoS setting
  * provenance
  * logs
  * templates
  * pluggable multi-role security

= Hollow
After two years of internal use, Netflix is offering a new open source project as a powerful option to cache data sets that change constantly.

Hollow is a Java library and toolset aimed at in-memory caching of data sets up to several gigabytes in size. Netflix says Hollow's purpose is threefold: It’s intended to be more efficient at storing data; it can provide tools to automatically generate APIs for convenient access to the data; and it can automatically analyze data use patterns to more efficiently synchronize with the back end.

Ditch
=====
* Tez: always buggy
* Oozie: buggy
* Flume: better StreamSets or Kafka
* Storm: Spark, Apex, Flink better
* Hive: slow?
* HDFS: Java memory slow; NameNode bottleneck; better MaprFS, Gluster

Alternatives
============
* Cassandra: Riak, CouchBar
* Kafka: RabbitMQ
* YARN: Mesos, Kubernetes

HBase vs Cassandra
==================
https://hortonworks.com/blog/hbase-cassandra-benchmark/
* HBase faster for read-heavy
* Cassandra faster for write-heavy

= Speed comparison Big Data Databases
* Impala may be faster than Spark; Kognitio even faster?; Greenplum slightly faster?
* Presto ~ Hive < Spark
