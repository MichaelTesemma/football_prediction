import pandas as pd

data = pd.read_csv('stack_data_with_mlp.csv')


class accuracy:
    def __init__(self, model, name):
        self.name = name
        self.model = model
        self.right_count = 0
        self.wrong_count = 0
        self.result = data['result']
    def check(self):
        for i in self.result:
            if self.model[i] == self.result[i]:
                self.right_count += 1
            else:
                self.wrong_count += 1

        print(f'The {self.name} got {self.right_count} predictions right and {self.wrong_count} predictions wrong')
        percentage = self.right_count * 100 / len(data)
        print(f'{percentage}%') 

knn = accuracy(model=data['knn'], name='KMeans')
knn.check()
rfc = accuracy(model=data['rfc'], name='Random Forest')
rfc.check()
log = accuracy(model=data['log'], name='Logistic Regression')
log.check()
gb = accuracy(model=data['gb'], name='Gausian')
gb.check()
sgd = accuracy(model=data['sgd'], name='SGD')
sgd.check()
lda = accuracy(model=data['lda'], name='LDA')
lda.check()
svc = accuracy(model=data['svc'], name='Support Vector machine')
svc.check()
mlp = accuracy(model=data['mlp'], name='Neural Network')
mlp.check()
