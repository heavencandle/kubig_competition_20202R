# kubig_competition_20202R

## 1. Competition theme
The dataset consists of data collected from heavy Scania trucks in everyday usage. The system in focus is the Air Pressure system (APS) which generates pressurized air that are utilized in various functions in a truck, such as braking and gear changes. The dataset’s positive class consists of component failures for a specific component of the APS system. The negative class consists of trucks with failures for components not related to the APS. The data consists of a subset of all available data, selected by experts.  

## 2. Data description
  - Dataset Details : The training set contains 60000 examples in total in which 59000 belong to the negative class and 1000 positive class. The test set contains 16000 examples.  
    • Number of Attributes : 171  
    • Attribute Information: The attribute names of the data have been anonymized for proprietary reasons. It consists of both single numerical counters and histograms consisting of bins with different conditions. Typically the histograms have open ended conditions at each end.   
      For example if we measuring the ambient temperature ‘T’ then the histogram could be defined with 4 bins where:   
      • bin 1 collect values for temperature T < 20   
      • bin 2 collect values for temperature T >= 20 and T < 0   
      • bin 3 collect values for temperature T >= 0 and T < 20   
      • bin 4 collect values for temperature T > 20   
    • The attributes are as follows: class, then anonymized operational data. The operational data have an identifier and a bin id, like ‘ Identifier_Bin ’. In total there are 171 attributes, of which 7 are histogram variables. Missing values are denoted by na  

## 3. MODEL Configuration
- CLASS_WEIGHT = {0:0.015, 1:0.985}  
- optimizer, LEARNING_RATE = Adam, 0.00001
- BATCH SIZE = 200
- EPOCHS = 30
- layers = Dense layers, (100, 100, 50, 50, 50)

## 4. Model Assessment - model name 20201001_234728
![alt text](https://github.com/heavencandle/kubig_competition_20202R/blob/master/graph.PNG)
## 5. How to run train & evaluation
1. run classficatin.py
   - What does classificatin.py do?
      1) Read csv files and process data : replace na's, reorganize histogram columns into aggrevated values(e.g. ag_001~ag_009 -> ag_max, ag_min, ag_actv_bins)  
      2) Split data into train, validation, test data set  
      3) If needed, saves processed data into csv  
2. run evaluation.py  
   - have to change *MODEL_NAME* into h5 file name you want to evaluate  

## 6. Processing & Train details
1. Data processing  
  - NA replacement : replace na's depending on class(i.e. positive/negative)  
    > Why not replace with class-free mean values?  
    > : There was relationship between NA values and 'neg' class, P(neg|NA) was over 90% for most columns. So, if just replace with class-free mean values, it looses its effect on 'neg' class.  
  - Attributes reorganization  
    1) non-histogram columns: no reorganizing  
    2) histogram columns: aggregate rowwise values into max, min, number of activated bins.  
    > Why max and min values?  
    > : most row wise value of histogram columns(e.g. ag_001 ~ ag_009) have similar shpae as a bell, which means abnormal data may have (1) an uncompletely drawn bell shape or (2) too high / low value.  
      
    > Why activated bins?  
    > : Considering data details, each bins may mean time axis. For example, ag_001 ~ ag_009 may mean somewhat pressure change upon time. Upon this assumption, low activated bins value menas radical change(e.g. sudden acceleration) and high activated bins value means retarded change(e.g. not working in time).  
2. Training  
  - Difference in loss per classes(loss = FN * 500 + FT * 10, have to increase Recall) and imbalance between positive data an negative data. More neg data(about 99%) an than pos data.
    > apply class weight
    
  
