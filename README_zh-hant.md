# Soccer Ball Outcome Prediction Project
這項專題的目標是能夠根據足球員的個人能力和球員全體的陣型，利用Graph Neural Network，進行球賽結果的預測。這項工具希望能輔助商業賽事評估／競猜以及球隊進行陣型的分析、球員調度、能力培養，達到利用科技協助運動項目的目標。
發想: https://deepfindr.github.io/

- [Key words](#key-words)
- [Purpose of project](#purpose-of-project)
- [Overview and comparison of existing related research](#overview-and-comparison-of-existing-related-research)
- [Research methods](#research-methods)
- [Implementation](#implementation)
- [Experiment](#experiment)
- [Performance and Result](#performance-and-result)
- [Conclusion](#conclusion)
- [Reference](#reference)
 
## Key words
Graph Neural Network, Transformer, Ensemble

## Purpose of project
這個專題的最初想法來自於我對體育賽事的熱情，以及當時臨近的世界盃賽事。因此我認為，若是有一個機器學習模型可以從賽前公布的隊伍陣型和FIFA的球員能力評級來預測該場賽事的結果應該會很有趣。這項專題的最終目的是希望能藉這項工具來輔助商業賽事評估以及球隊進行陣型的分析、球員調度、能力培養，達到利用科技協助運動項目的目標。

## Overview and comparison of existing related research
相較其他機器學習領域，圖神經網路的研究相對冷門，但圖神經網路現今的應用已相當廣泛，目前多應用於化學、交通、推薦系統，也有應用於圖像、文字的等等的相關研究[1]。近期有相關的論文研究圖神經網路運用於運動賽事的結果預測[2]，但主要研究對象為美式足球及電子競技，且研究專注於為輸入模型的圖新增表徵來改善模型的表現，是與此專題的相異之處。

## Research methods
![Imgur Image](https://i.imgur.com/9CUIwzE.png) 

在實作上，我將整體架構分成三大部分：
1. Graph conversion: GNN的實作上，輸入的資料須為Graph形式，所以在進行模型訓練前，須先將賽事資訊與球員能力資料結合，轉換成Graph。在這篇專題中，我將一次賽事轉換成一個Graph，每張Graph都會有22個Node來代表場上的所有球員（主隊11名與客隊11名）和不同的Edge來呈現足球賽事中的陣型站位。圖資料處理因此又分成兩個部分：
    1. Node: 一個Node代表著球場上的一名球員，Node會有該球員當時的能力評比和出戰時所站的位子等40個Feature (i.e. X, Y, Height, Ball Skills, Defense…)
    2. Edge: 為了呈現球隊的陣型架構，Node和Node之間將會有Edge相連接，Edge會將前後與相鄰的兩個Node連接，而前鋒的Node也將與對手前鋒以Edge連接，如此來完成單場賽事的Graph轉換。
    
    ![Imgur Image](https://i.imgur.com/u2ksw7k.png)

2. Model training: 處理完Graph資料的轉換後，接著會進行模型的訓練。在模型訓練時，我導入了超參數搜索(Hyperparameter Search)，調整模型的深度、神經元數量、learning rate等等不同的參數進行不同initialization的模型的訓練。
3. Ensemble: 訓練完多組模型後，從中取出表現較好的模型進行Ensemble，並運用此Ensemble進行預測（客場勝／平手／主場勝），並得出最終的結果。

## Implementation 
1. GNN: 由於模型的目標是希望藉由球員的能力與之間連結的關聯性(陣型)進行Graph-level的預測，在設計中我採用的是Transformer Convolution Layer和Top-K Pooling來進行Graph Pooling：
    1. Transformers Convolution Layer: 在模型的訓練時，我引用Transformer中的Attention概念。由於Graph如何相連接是最初的人為設定，Edge也不具重要性之分，引入Attention可以讓模型可以自行決定某個Node與對另一個Node之間Edge的重要程度。
    2. Top-K Graph Pooling: 與Node-level預測不同，對於Graph-level預測，模型最終需要將Graph的所有Feature Data(Node Embedding)減小至一個Embedding Vector作為那張Graph的Representation。我引用的是Hierarchical Pooling中的Top-K Pooling，模型會根據一個可學習的vector選擇哪K個Node會留下，並迭代的將Graph中的Node逐漸減少，在message passing後將移除的Node的Feature資料分散給鄰近的Node。最終的Embedding Vector組成為每個階段當時所有的Node的Vector的平均和最大值。

![Imgur Image](https://i.imgur.com/AgzcWhg.png)

模型的整體設計如下圖所示，由Transformer Convolution Layer, Linear Transformation Layer, 與Batch Normalization Layer的數個Block組成，接著會經過Top-K Pooling Layer將Graph進行downsize。而每經Pooling Layer時會取Node Embedding的平均及最大值用於最終Graph Representation的組成。

![Imgur Image](https://i.imgur.com/IhPcMHR.png)


2. Ensemble: 為了提高預測準確率，我將訓練的176組不同initialization模型中取表現最好的前10組(Top10 Highest Accuracy/F1 Score)以Stacking Ensemble的方式堆疊並進行最後的效能評估。此外，我也加入了閾值(Threshold)的概念來模擬實際商業競猜時對沒有把握的賽事便不進行預測的情況。Threshold的實現為設定模型只以所有賽事中最有自信的前N%的賽事進行預估。

## Experiment
1. GNN: 在模型方面，我調整了以下的不同Hyperparameter進行實驗：
    1. Batch Size: 64, 128, 256
    2. Learning Rate: 0.01, 0.02, 0.03
    3. Embedding Size: 16, 32, 64
    4. Model Block Layers: 2, 3, 4, 5
    5. Model Dense Neurons: 32, 64, 128, 256
    6. Model Transformer Attention Heads: 1, 2, 3, 4
    7. Model Transformer Dropout Rate: 0.2, 0.5, 0.9
    8. Model Top K Ratio: 0.2, 0.5, 0.8, 0.9
    9. Model Top K Every N Blocks: 1, 2, 3
    10. Weight Decay: 0.00001, 0.0001, 0.001
    11. SGD Momentum: 0.5, 0.8, 0.9
    12. Scheduler Gamma: 0.5, 0.8, 0.9, 0.995, 1
2. Ensemble: 
    1. Model Selection: Highest Validation Accuracy, Highest Validation F1 Score
    2. Ensemble Output: Voting, Average Output
    3. Threshold: 30%, 50%, 70%, 90%, 100%
    4. Threshold Target: Smallest Vote Variance, Highest Average Output

## Performance and Result
![Imgur Image](https://i.imgur.com/r9Jop3t.png)

最終實驗的結果如表所示。以Dataset中三項Label中（客場勝／平手／主場勝）占比最大的主場勝利作為Baseline，可以看到8組模型皆成功提升了約10%的Accuracy。此外，在30%的閾值下利用F1 Score組成的Voting-Avg Output和Avg-Vote Var的兩個組合皆達到了近80%的Accuracy，而Accuracy組成的Avg-Avg Output則在其餘的閾值下表現亮眼。

## Conclusion
從實驗結果可以看出，GNN模型雖已經達到一定的水準可以協助判斷足球賽事的結果，但在閾值限制較少的情況下的表現仍舊有待提升，且由於足球賽事存在一部份的強弱隊之分，所以在預測上與足球專家的經驗相比是否有明顯更為突出的表現仍須更多的檢驗。此外，因運動賽事具有一定的隨機性，預測時會受到不小的限制，模型的表現也會受平手或爆冷門的影響。即便如此，這個系統的能力依舊能為足球賽事的結果替人們提供較為快速、自動化且便利的預測，達到輔助商業賽事評估／競猜以及球隊進行陣型的分析、球員調度、能力培養的目標。
未來的研究目標或可朝加入具能力處理Data隨機性的Normalizing Flow等等作為改善的方向。

## Reference
[1] Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C., & Sun, M. (2020). Graph neural networks: A review of methods and applications. AI Open, 1, 57-81. https://doi.org/10.1016/j.aiopen.2021.01.001

[2] Xenopoulos, P., & Silva, C. (2022). Graph Neural Networks to Predict Sports Outcomes. arXiv. https://doi.org/10.1109/BigData52589.2021.9671833
