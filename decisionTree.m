clear;clc;close all;


% load specified dataset
set = 'pima';
[train,test] = loadData(set,4641);

% set reference to training function
if strcmpi(set,'car')
    best_nodes = 79;
    traintree = @carTreefit;
    
elseif strcmpi(set,'pima')
    best_nodes = 15;
    traintree = @pimaTreefit;
    
else
    error('Please use either the ''pima'' set or the ''car'' set');
    
end

maxsplits = 100;
parents = 1;

for q = 1:maxsplits
    [currmodel,curracc,currtime] = traintree(train,q);
    currsplit = sum(~cellfun(@isempty,currmodel.ClassificationTree.CutType));
    if q == 1
        dtres(parents).model = currmodel;
        dtres(parents).valacc = curracc;
        dtres(parents).traintime = currtime;
        dtres(parents).rsplit = currsplit;
    else
        if currsplit > dtres(parents).rsplit
            parents = parents+1;
            dtres(parents).model = currmodel;
            dtres(parents).valacc = curracc;
            dtres(parents).traintime = currtime;
            dtres(parents).rsplit = currsplit;
        end 
    end
end

for q = 1:length(dtres)
    tic;
    res = dtres(q).model.predictFcn(test);
    dtres(q).testtime = toc;
    dtres(q).testacc = sum(res == test{:,1})/size(test,1);
    
    res = dtres(q).model.predictFcn(train);
    dtres(q).trainacc = sum(res == train{:,1})/size(train,1);
end

for train_percentage = 1:20
   tr_idx = size(train,1)*(train_percentage*5/100);
   numdata(train_percentage) = round(tr_idx);
   for p = 1:10
       train = train(randperm(length(train{:,1})),:);
       train_limit = train(1:numdata(train_percentage),:);

       [currmodel,data_amt_valacc(train_percentage,p),currtime] = traintree(train_limit,best_nodes);

       data_amt_testacc(train_percentage,p) = sum(currmodel.predictFcn(test) == test{:,1})/size(test,1);
       data_amt_trainacc(train_percentage,p) = sum(currmodel.predictFcn(train_limit) == train_limit{:,1})/size(train_limit,1);
   end
   
   
end

figure
plot([dtres.rsplit],[dtres.testacc])
hold on
plot([dtres.rsplit],[dtres.valacc])
plot([dtres.rsplit],[dtres.trainacc])
grid minor
legend('Test accuracy','Validation accuracy','Training accuracy')
xlabel('Number of decision nodes')
ylabel('Accuracy')

figure
plot(numdata,mean(data_amt_valacc,2))
hold on
plot(numdata,mean(data_amt_testacc,2))
plot(numdata,mean(data_amt_trainacc,2))
grid minor
legend('Validation accuracy','Test accuracy','Training accuracy')
xlabel('Amount of training data')
ylabel('Accuracy')



save(['dtree_',set,'.mat'],'train','test','dtres','maxsplits','data_amt_testacc','data_amt_valacc');