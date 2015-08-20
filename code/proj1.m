%clear;
clc;
%Read data
data = csvread('train.csv',1,0);
test_data = csvread('test.csv',1,0);
reduce_dim = false;
X = data(20000:40000,2:size(data,2));
y= data(20000:40000,1);

Xt = data(40000:42000,2:size(data,2));
yt = data(40000:42000,1);
figure(1)
for ii = 1:60
    subplot(6,10,ii)
    rand_num = randperm(500,1);
    image(rot90(rot90(reshape(X(rand_num,:),28,28))'))
    title((y(rand_num)),'FontSize',20)
    axis off
end
colormap gray
%visualize the first column of data
bin=0:10;
figure,
histogram(y,bin,'Normalization','count','FaceAlpha',0.6,'FaceColor','red');

%Cut the dataset
cv = cvpartition(y, 'holdout', .5);
Xtrain = X(cv.training,:);
Ytrain = y(cv.training,1);
Xtest = X(cv.test,:);
Ytest = y(cv.test,1);

%classification tree
tic
mdl_ctree = ClassificationTree.fit(Xtrain,Ytrain);
ypred = predict(mdl_ctree,Xtest);
Confmat_ctree = confusionmat(Ytest,ypred);
tctree = toc

%combine
tic
mdl = fitensemble(Xtrain,Ytrain,'bag',200,'Discriminant','type','Classification');
ypred = predict(mdl,Xtest);
Confmat_bag = confusionmat(Ytest,ypred);
tcom = toc

%combine tree
tic
mdlt = fitensemble(Xtrain,Ytrain,'bag',200,'tree','type','Classification');
ypred = predict(mdlt,Xtest);
Confmat_bagt = confusionmat(Ytest,ypred);
tcomt = toc

%svm
tic
[ypred1,svm_models] = multisvm(Xtrain,Ytrain,Xtest);
Confmat_svm = confusionmat(Ytest,ypred1);
tsvm = toc


%usint test.csv and output predictions from 3 classifiers.
% test_result = zeros(size(test_data,1),3);
% test(:,1) = predict(mdl_ctree,test_data);
% test(:,2) = predict(mdl,test_data);
% test(:,3) = svmpredict(svm_models,test_data);

%combine running time
t = [tctree,tcom,tsvm]
%calculate accuracy
acc_ctree = sum(diag(Confmat_ctree))/sum(sum(Confmat_ctree));
acc_ccom = sum(diag(Confmat_bag))/sum(sum(Confmat_bag));
acc_svm = sum(diag(Confmat_svm))/sum(sum(Confmat_svm));
acc = [acc_ctree,acc_ccom,acc_svm]

%test
ypred2 = predict(mdl,Xt);
Confmat_test = confusionmat(yt,ypred2);

%test.csv
result = predict(mdlt,test_data);

%label of row and column for heatmap
rlb = 9:-1:0;
clb = 0:9;
 
svmhm = HeatMap(rot90(Confmat_svm),'RowLabels',rlb,'ColumnLabels',clb);
addTitle(svmhm,'Confusion Matrix: SVM');

ctreehm = HeatMap(rot90(Confmat_ctree),'RowLabels',rlb,'ColumnLabels',clb);
addTitle(ctreehm,'Confusion Matrix: Single Classification Tree');


ccomhm = HeatMap(rot90(Confmat_bag),'RowLabels',rlb,'ColumnLabels',clb);
addTitle(ccomhm,'Confusion Matrix: Ensemble of Bagged Classification Trees');