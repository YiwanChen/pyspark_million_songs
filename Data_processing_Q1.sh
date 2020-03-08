#/data/msd/
#├─audio
#│	├───attributes
#│  │         └─msd-jmir-area-of-moments-all-v1.0.attributes.csv
#│  │         └─msd-jmir-lpc-all-v1.0.attributes.csv
#│  │         └─msd-jmir-methods-of-moments-all-v1.0.attributes.csv
#│  │         └─msd-jmir-mfcc-all-v1.0.attributes.csv
#│  │         └─msd-jmir-spectral-all-all-v1.0.attributes.csv
#│  │         └─msd-jmir-spectral-derivatives-all-all-v1.0.attributes.csv
#│  │         └─msd-marsyas-timbral-v1.0.attributes.csv
#│  │         └─msd-mvd-v1.0.attributes.csv
#│  │         └─msd-rh-v1.0.attributes.csv
#│  │         └─msd-rp-v1.0.attributes.csv
#│  │         └─msd-ssd-v1.0.attributes.csv
#│  │         └─msd-trh-v1.0.attributes.csv
#│  │         └─msd-tssd-v1.0.attributes.csv
#│	├───features
#│  │         └─msd-jmir-area-of-moments-all-v1.0.csv
#│  │         └─msd-jmir-lpc-all-v1.0.csv
#│  │         └─msd-jmir-methods-of-moments-all-v1.0.csv
#│  │         └─msd-jmir-mfcc-all-v1.0.csv
#│  │         └─msd-jmir-spectral-all-all-v1.0.csv
#│  │         └─msd-jmir-spectral-derivatives-all-all-v1.0.csv
#│  │         └─msd-marsyas-timbral-v1.0.csv
#│  │         └─msd-mvd-v1.0.csv
#│  │         └─msd-rh-v1.0.csv
#│  │         └─msd-rp-v1.0.csv
#│  │         └─msd-ssd-v1.0.csv
#│  │         └─msd-trh-v1.0.csv
#│  │         └─msd-tssd-v1.0.csv
#│	└─statistics
#│		└─sample_properties.csv.gz
#├─genre
#│	├─msd-MAGD-genreAssignment.tsv
#│	├─msd-MASD-styleAssignment.tsv
#│	└─msd-topMAGD-genreAssignment.tsv
#├─main
#│	└─summary
#│		├─analysis.csv.gz
#│		└─metadata.csv.gz
#└─tasteprofile
#	├─mismatches
#	│	├─sid_matches_manually_accepted.txt
#	│	└─sid_mismatches.txt
#	└triplets.tsv

# Dataset structure, and file size, the file are in csv format, but majority of files are compressed csv format. The largest folder is 'audio'. 

# 13167872421  52671489684  /data/msd/audio				12Gb
# 31585889     126343556    /data/msd/genre 			30Mb
# 182869445    731477780    /data/msd/main				174Mb
# 514256719    2057026876   /data/msd/tasteprofile		490Mb


hdfs dfs -cat /data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv | head

hdfs dfs -cat /data/msd/main/summary/analysis.csv.gz |gunzip | head

hdfs dfs -du /data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv | head


text_file = spark.textFile("hdfs:///data/msd/audio")
count = text_file.count();
count.dump();

# count lines
hdfs dfs -cat /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/*.* | wc -l

hdfs dfs -cat /data/msd/main/summary/metadata.csv.gz | gunzip | wc -l   # 1000001
hdfs dfs -cat /data/msd/main/summary/analysis.csv.gz | gunzip | wc -l   # 1000001


hdfs dfs -cat /data/msd/main/summary/metadata.csv.gz | gunzip | head

