# transfer_learning
The project about transfer learning.

And the machine parameters used are as follows：

| CPU             | 内存 | 显卡         |
| --------------- | ---- | ------------ |
| i7-5930K (6/12) | 128G | GTX 1080 * 4 |

(Also read codes of other algorithm, but some try it myself, others are not. )



## Non-deep migration learning

### TCA

Download the data set of Office-31, the input format is `.mat` (if it is not, please use Matlab to convert it). 

You can see the result with the command: `python TCA.py`.

The results is as follows:

(**The accuracy is up to 91.08%**)

![](./results/TCA.png)



### JDA

I used the data set provided by the author Long Mingsheng source code. 

10 common categories are picked out from 4 datasets of object images: `caltech, amazon, webcam, and dslr`. Then follow the previously reported protocols for preparing features, i.e., extracting SURF features and quantizing them into an 800-bin histogram with codebooks computed via K-means on a subset of images from amazon. 
The file name: `*_SURF_L10.mat` is about features and labels.

------

I write two version of the JDA.
One is the code of *Long Mingsheng*'s' MATLAB version of the code. According to the structure of the paper, after the reappearance, the results are as follows:

 (**The best accuracy is 79.62%,** about webcam and dslr; but **the worst accuracy is 24.76%**, about webcam and caltech)

![](./results/JDA-1-1.png)

![](./results/JDA-1-2.png)

![](./results/JDA-1-3.png)

![](./results/JDA-1-4.png)



The other is *Wang Jindong*'s Python version of the code. He changed the structure of the code, reduced the function call, converted to a loop, and used the data already obtained in the previous loop, avoiding the recalculation. Therefore, the error is also reduced. The results are as follows:

(**The accuracy is 46.56%** about caltech and amazon. And in last version is only 27.78%)

![](./results/JDA-2.png)





