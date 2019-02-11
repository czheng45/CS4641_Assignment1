clear;clc;close all;


% load specified dataset
set = 'car';


% set reference to training function
if strcmpi(set,'car')
    trainknn = @carKNN;
    best_k = 9;
    
elseif strcmpi(set,'pima')
    trainknn = @pimaKNN;
    best_k = 27;
    
else
    error('Please use either the ''car'' set or the ''pima'' set');
    
end

kmax = 100;

for q = 1:kmax
    for r = 1:10
        [train,test] = loadData(set,r*4641);
        
        [currmodel,curracc(r),currtime(r)] = trainknn(train,q);
        models{r} = currmodel;
        tic
        testacc(r) = sum(currmodel.predictFcn(test) == test{:,1})/size(test,1);
        testtime(r) = toc;
        
    end
    knnres(q).model = models;
    knnres(q).valacc = mean(curracc);
    knnres(q).traintime = mean(currtime);
    knnres(q).k = q;
    knnres(q).testacc = mean(testacc);
    knnres(q).testtime = mean(testtime);
    
end


for train_percentage = 1:20
   tr_idx = size(train,1)*(train_percentage*5/100);
   numdata(train_percentage) = round(tr_idx);
   for p = 1:10
       train = train(randperm(length(train{:,1})),:);
       train_limit = train(1:round(tr_idx),:);

       [currmodel,data_amt_valacc(train_percentage,p),currtime] = trainknn(train_limit,best_k);

       data_amt_testacc(train_percentage,p) = sum(currmodel.predictFcn(test) == test{:,1})/size(test,1);
   end
   
   
end

figure
plot([knnres.k],[knnres.testacc])
hold on
plot([knnres.k],[knnres.valacc])
grid minor
legend('Test accuracy','Validation accuracy')
xlabel('k')
ylabel('Accuracy')

figure
plot(numdata,mean(data_amt_testacc,2))
hold on
plot(numdata,mean(data_amt_valacc,2))
grid minor
legend('Test accuracy','Validation accuracy')
xlabel('Amount of training data')
ylabel('Accuracy')

save(['knn_',set,'.mat'],'knnres','kmax','data_amt_testacc','data_amt_valacc');

