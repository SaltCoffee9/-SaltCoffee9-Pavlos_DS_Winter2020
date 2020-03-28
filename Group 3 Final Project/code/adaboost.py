import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn
from sklearn.datasets import make_moons, make_circles
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.externals.six import StringIO  
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from mpl_toolkits.mplot3d import Axes3D
import os
import imageio
import graphviz
import pydot

class Adaboost:
    """
    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
    """
    def __init__(self, X, y, base_classfier=DecisionTreeClassifier,learning_rate=1.,**kwargs):
        # initialize
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.num_samples = len(self.X)
        #initialize sample weight
        self.sample_weight = np.full(self.num_samples, 1 / self.num_samples)
        #store attributes
        self.classifiers = []
        self.scores = []
        self.errors_list = []
        self.alphas = []
        self.predictions = []
        #define base classfier
        self.base_classfier=base_classfier
        self.kwargs=kwargs

    def predict(self, data=None, labels=None, reduction="sign"):

        if data is None:
            data = self.X
            labels = self.y
        if reduction=='vote':
            #C(\boldsymbol{x})=\arg \max _{k} \sum_{m=1}^{M} \alpha^{(m)} \cdot \mathbb{I}\left(T^{(m)}(\boldsymbol{x})=k\right)
            predictions=np.argmax(np.array([
                sum([alpha*(classifier.predict(data)==k) for classifier, alpha in zip(self.classifiers, self.alphas)]) for k in range(self.n_classes)]),axis=0)
        else:
            predictions = np.zeros([len(data)]).astype("float")
            for classifier, alpha in zip(self.classifiers, self.alphas):
                predictions += alpha * classifier.predict(data)
            if reduction == "sign":
                predictions = np.sign(predictions)
            elif reduction == "mean":
                predictions /= len(self.classifiers)
 
        if labels is not None:
            if self.n_classes>2:
                #caculate precison score when multi-class classification
                score = precision_score(predictions, labels,average='micro')
            else:
                #else f1 score
                score = f1_score(predictions,labels)
            return predictions, score
        else:
            return predictions

    def contour_plot(self, data=None, labels=None, interval=0.2, title="adaboost", name=None, make_gif=False,
                     mode="3d"):
        if data is None:
            data = self.X
            labels = self.y
        if labels is None:
            labels = np.ones([len(data)])
        x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
        y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
        #xx, yy, Z_grid are used for plotting surface 
        xx, yy = np.meshgrid(np.arange(x_min, x_max, interval), np.arange(y_min, y_max, interval))
        X_grid = np.concatenate([np.expand_dims(np.ravel(xx), axis=-1),
                                 np.expand_dims(np.ravel(yy), axis=-1)], axis=-1)

        Z_grid = self.predict(data=X_grid, reduction="mean")
        Z_grid = Z_grid.reshape(xx.shape)
        if mode == "3d":
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_zlim(Z_grid.min()*1.1,Z_grid.max()*1.1)
            ax.view_init(azim=120)
            plt.title(title)
            ax.plot_surface(xx,yy,Z_grid,cmap=plt.cm.summer,alpha=0.6)
            ax.scatter(xs=data[labels==1][:,0], ys=data[labels==1][:, 1],zs=self.predict(data=data, reduction="mean")[labels==1],c='#00CED1')
            ax.scatter(xs=data[labels==-1][:,0], ys=data[labels==-1][:, 1],zs=self.predict(data=data, reduction="mean")[labels==-1],c='#DC143C')
            if not os.path.exists('temp'):
                os.mkdir('temp')
            if name is None:
                plt.savefig(title+'.png')
                plt.close()
            else:
                plt.savefig('temp/'+name+'.png')
                plt.close()

        if mode == "2d":
            plt.contourf(xx, yy, Z_grid, cmap=plt.cm.RdBu, alpha=.8)
            plt.scatter(data[:, 0], data[:, 1], c=labels,
                        cmap=ListedColormap(['#FF0000', '#0000FF']), edgecolors='k')
            plt.title(title)
            if name is None:
                plt.savefig(title+'.png')
                plt.close()
            else:
                plt.savefig('temp/'+name+'.png')
                plt.close()
        if make_gif == True:
            imgs=[]
            for i in range(len(os.listdir('temp'))):
                imgs.append(imageio.imread('temp/'+str(i+1)+'.png'))
            imageio.mimsave('AdaBoost.gif',imgs,'GIF',duration=0.1)

    def __next__(self, reduction="vote", plot=True, plot_mode="2d"):
        self.n_classes = len(np.array(sorted(list(set(self.y)))))
        if self.n_classes==2:
            reduction="sign"
        classifier = self.base_classfier(**self.kwargs)
        classifier.fit(self.X, self.y, sample_weight=self.sample_weight)
        predictions = classifier.predict(self.X)
        self.predictions.append((predictions==self.y).sum()/len(predictions))
        # err^{(m)}=\sum_{i=1}^{n} w_{i} \mathbb{I}\left(c_{i} \neq T^{(m)}\left(\boldsymbol{x}_{i}\right)\right) / \sum_{i=1}^{n} w_{i}
        error_rate = np.dot(predictions != self.y, self.sample_weight) / np.sum(self.sample_weight, axis=0)
        # \alpha^{(m)}=\log \frac{1-e r r^{(m)}}{e r r^{(m)}}+\log (K-1)
        alpha = self.learning_rate * (
            np.log((1. - error_rate)/error_rate) +
            np.log(self.n_classes - 1.)
        )
        # w_{i} \leftarrow w_{i} \cdot \exp \left(\alpha^{(m)} \cdot \mathbb{I}\left(c_{i} \neq T^{(m)}\left(\boldsymbol{x}_{i}\right)\right)\right)
        self.sample_weight *= np.exp(alpha * (predictions != self.y) *((self.sample_weight > 0) | (alpha < 0)))
        self.sample_weight /= np.sum(self.sample_weight)
        self.classifiers.append(classifier)
        self.alphas.append(alpha)
        _, f1 = self.predict(reduction=reduction)
        self.scores.append(f1)
        if plot:
            self.contour_plot(
                title="adaboost step " + str(len(self.classifiers)) + " f1 score {:.2f}".format(f1), name=str(len(self.classifiers)), mode=plot_mode)
            return f1
        else:
            return f1

    def print_DT(self):
        # print decesion trees by graphviz and pydot
        for i, classifier in enumerate(self.classifiers):
            dot_data = StringIO() 
            sklearn.tree.export_graphviz(classifier,out_file=dot_data)
            graph = pydot.graph_from_dot_data(dot_data.getvalue())  
            graph[0].write_pdf(str(i)+".pdf") 

    def plot_attrs(self,attr,xr,yr):
        # For example: model.plot_attrs('predictions',(0,100),(0,1))
        attrs=getattr(self,attr)
        if type(attrs)==type(np.arange(1)):
            n=attrs.shape[0]
        else:
            n=len(attrs)
        x=np.arange(n)
        plt.plot(x,attrs)
        plt.xlim(xr)
        plt.ylim(yr)
        if not os.path.exists(attr):
            os.mkdir(attr)
        plt.savefig(attr+'/'+str(len(self.alphas)-1)+'.png')
        plt.close()        
    


if __name__ == '__main__':
    #test
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
    score_0_2=[]
    score_0_4=[]
    score_0_6=[]
    score_0_8=[]
    score_1_0=[]
    for n in range(1,50):
        abc_dt = AdaBoostClassifier(n_estimators=n,
                                learning_rate=0.2)
        model = abc_dt.fit(X, y)
        y_pred = model.predict(X)
        score_0_2.append(metrics.accuracy_score(y, y_pred))
        
        abc_dt = AdaBoostClassifier(n_estimators=n,
                                learning_rate=0.2)
        model = abc_dt.fit(X, y)
        y_pred = model.predict(X)
        score_0_4.append(metrics.accuracy_score(y, y_pred))

        abc_dt = AdaBoostClassifier(n_estimators=n,
                                learning_rate=0.4)
        model = abc_dt.fit(X, y)
        y_pred = model.predict(X)
        score_0_6.append(metrics.accuracy_score(y, y_pred))
        
        abc_dt = AdaBoostClassifier(n_estimators=n,
                                learning_rate=0.6)
        model = abc_dt.fit(X, y)
        y_pred = model.predict(X)
        score_0_8.append(metrics.accuracy_score(y, y_pred))
        
        abc_dt = AdaBoostClassifier(n_estimators=n,
                                learning_rate=0.8)
        model = abc_dt.fit(X, y)
        y_pred = model.predict(X)
        score_1_0.append(metrics.accuracy_score(y, y_pred))
        
        
        


    xx=list(range(len(score_0_2)))
    plt.plot(xx,score_0_2,label='learning rate = 0.2')
    plt.plot(xx,score_0_4,label='learning rate = 0.4')
    plt.plot(xx,score_0_6,label='learning rate = 0.6')
    plt.plot(xx,score_0_8,label='learning rate = 0.8')
    plt.plot(xx,score_1_0,label='learning rate = 1.0')

    plt.legend() 
    plt.savefig('learing rate.png')

    # # model = Adaboost(X, y, base_classfier=LogisticRegression,penalty='l2',multi_class='auto',solver='lbfgs',learning_rate=0.6)
    # # model = Adaboost(X, y, learning_rate=0.08, base_classfier=SVC, gamma='auto',kernel='linear')
    # model = Adaboost(X, y, max_depth=1)
    # for i in range(100):
    #     print(model.__next__(plot=False))
    #     # model.plot_alphas(100,3)
    #     # model.plot_scores(100,1)
    #     model.plot_attrs('alphas',(0,100),(1.3,2.8))
    #     model.plot_attrs('scores',(0,100),(0.6,1))
    #     model.plot_attrs('predictions',(0,100),(0,1))
    #     model.plot_attrs('sample_weight',(0,105),(0,0.3))
    # imgs=[]
    # for i in range(len(os.listdir('alphas'))):
    #     imgs.append(imageio.imread('alphas/'+str(i)+'.png'))
    # imageio.mimsave('alpha.gif',imgs,'GIF',duration=0.1)
    # imgs=[]
    # for i in range(len(os.listdir('scores'))):
    #     imgs.append(imageio.imread('scores/'+str(i)+'.png'))
    # imageio.mimsave('scores.gif',imgs,'GIF',duration=0.1)
    # imgs=[]
    # for i in range(len(os.listdir('predictions'))):
    #     imgs.append(imageio.imread('predictions/'+str(i)+'.png'))
    # imageio.mimsave('predictions.gif',imgs,'GIF',duration=0.1)
    # imgs=[]
    # for i in range(len(os.listdir('sample_weight'))):
    #     imgs.append(imageio.imread('sample_weight/'+str(i)+'.png'))
    # imageio.mimsave('sample_weight.gif',imgs,'GIF',duration=0.1)
    # # model.print_DT()

    # print(model.alphas)



    # model.contour_plot(make_gif=False,mode="3d")

    # X, y = make_moons(n_samples=300, noise=0.2, random_state=3)
    # y[np.where(y == 0)] = -1
    # model = Adaboost(X, y, max_depth=1)
    # # model = Adaboost(X, y, base_classfier=LogisticRegression,penalty='l2')
    # # model = Adaboost(X, y, learning_rate=0.4, base_classfier=SVC)
    # for i in range(100):
    #     print(model.__next__(plot=True,plot_mode="3d"))
    # model.contour_plot(make_gif=True,mode="3d")
    # svc=SVC(probability=True, kernel='linear')

    # Create adaboost classifer object
    # abc =AdaBoostClassifier(n_estimators=100, base_estimator=svc,learning_rate=1)
    # lr=LogisticRegression(penalty='l2')
    # abc =AdaBoostClassifier(n_estimators=100, base_estimator=lr,learning_rate=1)
    # # Train Adaboost Classifer
    # model = abc.fit(X, y)

    # #Predict the response for test dataset
    # y_pred = model.predict(X)
    # print(f1_score(y,y_pred))

    # abc =AdaBoostClassifier(n_estimators=2, base_estimator=lr,learning_rate=1)
    # # Train Adaboost Classifer
    # model = abc.fit(X, y)

    # #Predict the response for test dataset
    # y_pred = model.predict(X)
    # print(f1_score(y,y_pred))