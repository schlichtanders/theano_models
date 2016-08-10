#! /bin/zsh
folder=$0:a:h
datasetnames=(boston concrete energy kin8nm powerplant winered yacht)
for n in $datasetnames; do
    python $folder/experiment.py parallel $n $n &
done
