unzip Development_Set.zip
unzip Evaluation_Set.zip
mkdir train
mkdir val
mkdir test
mv Development_Set/Training_Set/*/*.wav train
mv Development_Set/Training_Set/*/*.csv train

mv Development_Set/Validation_Set/*/*.wav val
mv Development_Set/Validation_Set/*/*.csv val

mv Evaluation_Set/*/*.wav test
mv Evaluation_Set/*/*.csv test
mv Development_Set/Validation_Set .
mv Evaluation_Set/ Test_Set
