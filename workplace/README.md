# Workplace

## crawler
Used to collect player statistics from FIFAINDEX

## data 
Raw match data, processed data (.pt graph files)

## data analyze
Used to do simple analyze on the correlation between players' average statistics & match outcome

## datasets
Include files like player statistic data, conversion table of fixtures to x, y axis, mostly contain files used to do pre-processing

## pre-processed
Contains pre-processed fixture & player statistic files

1. Convert date format to datetime64[ns](i.e. 2022-06-21)
2. Remove duplicate row
3. Remove empty row
4. Remove unknown player
5. Rename player by their corresponding ID using id_conversion_table in /datasets

