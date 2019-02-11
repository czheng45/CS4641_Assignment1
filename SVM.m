clear;clc;close all;


% load specified dataset
set = 'car';


% set reference to training function
if strcmpi(set,'car')
    trainsvm = @carSVM; 
    
elseif strcmpi(set,'pima')
    trainsvm = @pimaSVM;
    
else
    error('Please use either the ''car'' set or the ''pima'' set');
    
end

[train,test] = loadData(set,4641);

[svmres(1).model,svmres(1).valacc,svmres(1).traintime] = trainsvm(train,'linear',[]);
[svmres(2).model,svmres(2).valacc,svmres(2).traintime] = trainsvm(train,'polynomial',2);
[svmres(3).model,svmres(3).valacc,svmres(3).traintime] = trainsvm(train,'rbf',[]);

for q = 1:3
    res = svmres(q).model.predictFcn(test);
    svmres(q).testtime = toc;
    svmres(q).testacc = sum(res == test{:,1})/size(test,1);

    res = svmres(q).model.predictFcn(train);
    svmres(q).trainacc = sum(res == train{:,1})/size(train,1);
end


for train_percentage = 1:20
   tr_idx = size(train,1)*(train_percentage*5/100);
   numdata(train_percentage) = round(tr_idx);
   for p = 1:10
       train = train(randperm(length(train{:,1})),:);
       train_limit = train(1:round(tr_idx),:);

       [currmodel{1},data_amt_valacc(train_percentage,p,1),currtime(1)] = trainsvm(train_limit,'linear',[]);
       [currmodel{2},data_amt_valacc(train_percentage,p,2),currtime(2)] = trainsvm(train_limit,'polynomial',2);
       [currmodel{3},data_amt_valacc(train_percentage,p,3),currtime(3)] = trainsvm(train_limit,'rbf',[]);

       data_amt_testacc(train_percentage,p,1) = sum(currmodel{1}.predictFcn(test) == test{:,1})/size(test,1);
       data_amt_testacc(train_percentage,p,2) = sum(currmodel{2}.predictFcn(test) == test{:,1})/size(test,1);
       data_amt_testacc(train_percentage,p,3) = sum(currmodel{3}.predictFcn(test) == test{:,1})/size(test,1);
       
       data_amt_trainacc(train_percentage,p,1) = sum(currmodel{1}.predictFcn(train_limit) == train_limit{:,1})/size(train_limit,1);
       data_amt_trainacc(train_percentage,p,2) = sum(currmodel{2}.predictFcn(train_limit) == train_limit{:,1})/size(train_limit,1);
       data_amt_trainacc(train_percentage,p,3) = sum(currmodel{3}.predictFcn(train_limit) == train_limit{:,1})/size(train_limit,1);
   end
end


figure
hold on
plot(numdata,mean(data_amt_trainacc(:,:,1),2),'r:')
plot(numdata,mean(data_amt_valacc(:,:,1),2),'r--')
plot(numdata,mean(data_amt_testacc(:,:,1),2),'r')

plot(numdata,mean(data_amt_trainacc(:,:,2),2),'b:')
plot(numdata,mean(data_amt_valacc(:,:,2),2),'b--')
plot(numdata,mean(data_amt_testacc(:,:,2),2),'b')

plot(numdata,mean(data_amt_trainacc(:,:,3),2),'k:')
plot(numdata,mean(data_amt_valacc(:,:,3),2),'k--')
plot(numdata,mean(data_amt_testacc(:,:,3),2),'k')
grid minor
legend('Linear training accuracy','Linear validation accuracy','Linear test accuracy', ...
    'Quadratic training accuracy','Quadratic validation accuracy','Quadratic test accuracy', ...
    'RBF training accuracy','RBF validation accuracy','RBF test accuracy');
xlabel('Amount of training data')
ylabel('Accuracy')

save(['svm_',set,'.mat'],'train','test','svmres','data_amt_testacc','data_amt_valacc','data_amt_trainacc');

