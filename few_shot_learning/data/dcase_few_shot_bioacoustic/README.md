Put the Development_Set.zip and Evaluation_Set.zip files in this directory.

- http://dcase.community/challenge2021/task-few-shot-bioacoustic-event-detection#download

run

> sh prepare_data.sh

# Configure config/extraction.yaml
- path:
	data_train: <should point to the train directory> etc

Make sure to also download the Evaluation_Set_Full_Annotations.zip and copy the *.csv files to both the train/ and the respective subdirectory in Test_Set.


