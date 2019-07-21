# transfer_learning
The project about transfer learning.

And the machine parameters used are as follows：

| CPU             | 内存 | 显卡         |
| --------------- | ---- | ------------ |
| i7-5930K (6/12) | 128G | GTX 1080 * 4 |

(Also read codes of other algorithm, but some try it myself, others are not. )



## Non-deep transfer learning

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

 (**The best accuracy is 79.62%,** about webcam and dslr; but **the worst accuracy is 24.76%**, about webcam and caltech. And except `A->W` and `A->D`, all are better than 1 NN.)

![](./results/JDA-1-1.png)

![](./results/JDA-1-2.png)

![](./results/JDA-1-3.png)

![](./results/JDA-1-4.png)

First, I think the low accuracy due to my programming, but after I see the paper result, it may because the data or algorithm.

The best in paper is `D->W = 89.49`, and in my test is `W->D = 79.62`(and `D->W =73.85% `).

The worst in paper is `W->C = 31.17%`, and in my test is `W->C = 24.76%`.

![](./results/JDA-1-5.png)

------

The other is *Wang Jindong*'s Python version of the code. He changed the structure of the code, reduced the function call, converted to a loop, and used the data already obtained in the previous loop, avoiding the recalculation. Therefore, the error is also reduced. The results are as follows:

(**The accuracy is 46.56%** about caltech and amazon. And in last version is only 27.78%, in paper is 44.78%)

![](./results/JDA-2.png)



## Deep transfer learning

### ResNet50

The datasets are the same.

The structure of ResNet50 is as follows:

![](./image/ResNet50.png)

We can build it layer by layer according to its structure; it can also be used `torchvision.models.resnet50(pretrained=False, progress=True, **kwargs)`. I choose the latter, and use the pertained parameters (the parameters are in `./codes/ResNet50/resnet50-19c8e357.pth`, you can use it).

But the images is so large and I first set the `batches_size = 256`. So the docker stop, and then I forced the container size as 8 GB and decreased the batches size to 100.

And the part of the result as follows:

![](./results/ResNet50.png)

Obviously, the accuracy is increase. And only after 4 iteration, the target accuracy can be up to 93.58%. So, the network is good. And you can set up the batches size as 10, also can even lower.



### Digit Network

After reading Appendix D of "Self-Ensembling for Visual Domain Adaptation". I began to reproduce his digital migration network. I chose the simplest one. That is, the transfer learning from MNIST to USPS is realized. Its network structure is set as follows:

![](./image/MNISTtoUSPS.png)

The code of network are as follows:

``` python
def __init__(self, n_classes):
  			self.conv1_1 = nn.Conv2d(1, 32, (5, 5))
        self.conv1_1_bn = nn.BatchNorm2d(32)
      
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3))
        self.conv2_1_bn = nn.BatchNorm2d(64)

        self.conv2_2 = nn.Conv2d(64, 64, (3, 3))
        self.conv2_2_bn = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d((2, 2))

        self.drop1 = nn.Dropout()

        self.fc3 = nn.Linear(1024, 256)

        self.fc4 = nn.Linear(256, n_classes)
```

And the result is:

![](./results/MNIST2USPS-1.png)

![](./results/MNIST2USPS-2.png)









